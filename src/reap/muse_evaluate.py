"""CPU teacher-forced loss and baseline divergence for streamed Muse pruning.

These diagnostics measure token prediction, not executable coding success.
No model weights are exported. Each candidate reroutes through physically sliced
expert banks with the original top-k and native Muse output softcap.
"""

from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from reap.models.muse_spark import prune_muse_experts
from reap.muse_streaming import (
    MuseShardSource,
    _activation_hashes,
    _activation_path,
    _batches_digest,
    _file_digest,
    _prepare_batches,
    _save_json,
)


def build_pruning_plan(source, calibration, metric, keep):
    """Rank every sparse layer independently, with deterministic lower-ID ties."""
    if metric not in ("reap", "weighted_ean_sum"):
        raise ValueError("metric must be reap or weighted_ean_sum")
    c = source.config
    if not c.num_experts_per_tok <= keep <= c.num_local_experts:
        raise ValueError("keep must be between top-k and the original expert count")
    calibration = Path(calibration)
    state = json.loads((calibration / "progress.json").read_text())
    if (
        not state["complete"]
        or state["next_layer"] != c.num_hidden_layers
        or state["identity"]["source"] != source.identity
        or not state["identity"].get("collect_metrics", True)
        or [r["layer"] for r in state["layers"]] != list(range(c.num_hidden_layers))
    ):
        raise ValueError("plan requires complete matching baseline calibration")
    retained, coverage = {}, {}
    for index, row in enumerate(state["layers"]):
        path = calibration / "scores" / f"layer-{index:05d}.safetensors"
        if _file_digest(path) != row["score_sha256"]:
            raise ValueError("calibration score checksum differs")
        if c.mlp_layer_types[index] != "sparse":
            continue
        scores = load_file(path)
        values, counts = scores[metric], scores["expert_frequency"]
        if (
            values.shape != (c.num_local_experts,)
            or counts.shape != values.shape
            or not torch.isfinite(values).all()
            or (counts < 0).any()
            or int(scores["total_tokens"]) != state["tokens"]
            or int(counts.sum()) != state["tokens"] * c.num_experts_per_tok
        ):
            raise ValueError(f"invalid scores or coverage at layer {index}")
        retained[str(index)] = (
            values.argsort(descending=True, stable=True)[:keep].sort().values.tolist()
        )
        coverage[str(index)] = {
            "unobserved_experts": int((counts == 0).sum()),
            "experts_below_ten_assignments": int((counts < 10).sum()),
            "minimum_assignments": int(counts.min()),
            "retained_unobserved": int((counts[retained[str(index)]] == 0).sum()),
        }
    return {
        "schema": 1,
        "source": source.identity,
        "metric": metric,
        "keep": keep,
        "calibration_progress_sha256": _file_digest(calibration / "progress.json"),
        "calibration_tokens": state["tokens"],
        "retained_experts": retained,
        "coverage": coverage,
    }


class PrunedMuseShardSource(MuseShardSource):
    """Load one source layer and slice its banks before replaying the candidate."""

    def __init__(self, root, plan):
        super().__init__(root)
        if plan.get("schema") != 1 or plan.get("source") != self.identity:
            raise ValueError("pruning plan source identity differs")
        sparse = {
            str(i)
            for i, kind in enumerate(self.config.mlp_layer_types)
            if kind == "sparse"
        }
        retained = plan["retained_experts"]
        keep = plan["keep"]
        if (
            type(keep) is not int
            or not self.config.num_experts_per_tok
            <= keep
            <= self.config.num_local_experts
        ):
            raise ValueError("invalid pruning plan expert count")
        if set(retained) != sparse:
            raise ValueError("pruning plan must cover every sparse layer")
        for ids in retained.values():
            if (
                not isinstance(ids, list)
                or len(ids) != keep
                or any(type(i) is not int for i in ids)
                or ids != sorted(set(ids))
                or min(ids) < 0
                or max(ids) >= self.config.num_local_experts
            ):
                raise ValueError("invalid retained expert IDs")
        # Copy the plan: caller mutations must not change a resumed candidate.
        self.retained = {int(i): list(ids) for i, ids in retained.items()}
        self.base_identity = dict(self.identity)
        self.identity = {
            **self.identity,
            "pruning_plan_sha256": hashlib.sha256(
                json.dumps(plan, sort_keys=True).encode()
            ).hexdigest(),
            "evaluation_runner_sha256": _file_digest(Path(__file__)),
        }

    @contextmanager
    def layer(self, index):
        with super().layer(index) as layer:
            if index in self.retained:
                prune_muse_experts(layer.mlp, self.retained[index])
            yield layer


def muse_logits(head, hidden, config):
    """Match native output dtype, multiplication order, and tanh softcapping."""
    logits = head(hidden)
    logits = logits * config.output_multiplier
    logits = logits / config.final_logit_softcapping
    logits = torch.tanh(logits)
    return logits * config.final_logit_softcapping


def _checked_final_run(output, batches, source_identity):
    state = json.loads((output / "progress.json").read_text())
    if (
        not state["complete"]
        or state["stage"] != "final"
        or state["identity"]["source"] != source_identity
        or state["identity"]["batches_sha256"] != _batches_digest(batches)
        or state["batches"] != len(batches)
    ):
        raise ValueError("evaluation requires a matching complete final activation run")
    if _activation_hashes(output, "final", len(batches)) != state["activation_hashes"]:
        raise ValueError("final activation checksum differs")
    return state


@torch.inference_mode()
def evaluate_final_activations(
    source, batches, output, *, baseline=None, chunk_tokens=32
):
    """Score valid adjacent token pairs; optionally compare against baseline.

    KL is baseline || candidate. All valid next tokens are scored, including prompt
    tokens; this is not assistant-only loss. Left/right padding and transitions
    across padding are excluded. Logits are never persisted or held for a corpus.
    """
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be a positive integer")
    if source.config.final_logit_softcapping <= 0:
        raise ValueError("positive native output softcap required")
    batches = _prepare_batches(batches, source.config)
    output = Path(output).resolve()
    baseline = None if baseline is None else Path(baseline).resolve()
    totals = {
        "tokens": 0,
        "nll_sum": 0.0,
        "baseline_nll_sum": 0.0,
        "kl_sum": 0.0,
        "top1_agreements": 0,
        "logit_squared_error": 0.0,
    }
    per_batch = []
    with ExitStack() as stack:
        for directory in sorted({output} | ({baseline} if baseline else set())):
            lock = stack.enter_context((directory / ".lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("evaluation run is being written") from exc
        state = _checked_final_run(output, batches, source.identity)
        if baseline:
            _checked_final_run(
                baseline, batches, getattr(source, "base_identity", source.identity)
            )
        with source.lm_head() as head:
            for i, batch in enumerate(batches):
                hidden = load_file(_activation_path(output, "final", i))["hidden"]
                ref = (
                    None
                    if baseline is None
                    else load_file(_activation_path(baseline, "final", i))["hidden"]
                )
                valid = batch["attention_mask"][:, :-1] & batch["attention_mask"][:, 1:]
                rows, positions = valid.nonzero(as_tuple=True)
                report = {key: 0 for key in totals}
                for offset in range(0, len(rows), chunk_tokens):
                    r, p = (
                        rows[offset : offset + chunk_tokens],
                        positions[offset : offset + chunk_tokens],
                    )
                    targets = batch["input_ids"][r, p + 1]
                    logits = muse_logits(head, hidden[r, p], source.config).float()
                    log_probs = F.log_softmax(logits, dim=-1)
                    nll = -log_probs.gather(1, targets[:, None]).sum(
                        dtype=torch.float64
                    )
                    if not torch.isfinite(nll):
                        raise ValueError("nonfinite evaluation loss")
                    report["tokens"] += len(r)
                    report["nll_sum"] += float(nll)
                    if ref is not None:
                        ref_logits = muse_logits(head, ref[r, p], source.config).float()
                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        report["baseline_nll_sum"] += float(
                            -ref_log_probs.gather(1, targets[:, None]).sum(
                                dtype=torch.float64
                            )
                        )
                        report["kl_sum"] += float(
                            (ref_log_probs.exp() * (ref_log_probs - log_probs)).sum(
                                dtype=torch.float64
                            )
                        )
                        report["top1_agreements"] += int(
                            (ref_logits.argmax(-1) == logits.argmax(-1)).sum()
                        )
                        report["logit_squared_error"] += float(
                            (ref_logits - logits).square().sum(dtype=torch.float64)
                        )
                per_batch.append(report)
                for key in totals:
                    totals[key] += report[key]
        if not totals["tokens"]:
            raise ValueError("evaluation requires at least one valid next-token pair")
        n = totals["tokens"]
        nll = totals["nll_sum"] / n
        result = {
            "schema": 1,
            "scored_tokens": n,
            "mean_nll": nll,
            "perplexity": math.exp(nll),
            "chunk_tokens": chunk_tokens,
            "run_progress_sha256": _file_digest(output / "progress.json"),
            "evaluator_sha256": _file_digest(Path(__file__)),
            "batches_sha256": state["identity"]["batches_sha256"],
            "scope": "teacher-forced text; all valid adjacent tokens, not coding accuracy",
            "per_batch": per_batch,
        }
        if baseline:
            result.update(
                baseline_progress_sha256=_file_digest(baseline / "progress.json"),
                baseline_mean_nll=totals["baseline_nll_sum"] / n,
                delta_mean_nll=(totals["nll_sum"] - totals["baseline_nll_sum"]) / n,
                baseline_to_candidate_kl=totals["kl_sum"] / n,
                top1_agreement=totals["top1_agreements"] / n,
                logit_rmse=math.sqrt(
                    totals["logit_squared_error"] / (n * source.config.vocab_size)
                ),
            )
        if any(not math.isfinite(v) for v in result.values() if isinstance(v, float)):
            raise ValueError("nonfinite evaluation summary")
    return result


def load_corpus_batches(path, split):
    """Read pretokenized, provenance-bearing corpus records without retokenizing."""
    corpus = json.loads(Path(path).read_text())
    records = corpus[split]
    batches = [
        {"input_ids": torch.tensor([r["input_ids"]], dtype=torch.long)} for r in records
    ]
    return batches, records


def write_evaluation(path, result, records):
    """Add category/record identity to the per-batch diagnostics."""
    if len(records) != len(result["per_batch"]):
        raise ValueError("record count differs from evaluation batches")
    for report, record in zip(result["per_batch"], records):
        report.update(record_id=record["id"], category=record["category"])
    _save_json(Path(path), result)
