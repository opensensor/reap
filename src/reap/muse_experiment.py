"""Resumable coding screening: calibration, baseline loss, pruned loss/KL.

One process, CPU only. Completion means the requested teacher-forced comparisons
finished; it does not authorize export or certify coding quality or device fit.
"""

import argparse
import fcntl
import json
from pathlib import Path
import time

import torch

from reap.muse_evaluate import (
    PrunedMuseShardSource,
    build_pruning_plan,
    evaluate_final_activations,
    load_corpus_batches,
    write_evaluation,
)
from reap.muse_streaming import (
    MuseShardSource,
    _file_digest,
    _save_json,
    run_calibration,
)


def run_experiment(
    source_root,
    corpus_path,
    output,
    *,
    keeps=(128, 144),
    metrics=("reap", "weighted_ean_sum"),
    resume=False,
    stop_after_calibration_layer=None,
    progress=print,
):
    output = Path(output).resolve()
    source = MuseShardSource(source_root)
    if output == source.root or output.is_relative_to(source.root):
        raise ValueError("experiment output must be outside source checkpoint")
    keeps, metrics = list(keeps), list(metrics)
    if (
        not keeps
        or len(set(keeps)) != len(keeps)
        or any(
            type(k) is not int
            or not source.config.num_experts_per_tok
            <= k
            <= source.config.num_local_experts
            for k in keeps
        )
    ):
        raise ValueError("invalid or duplicate retained expert counts")
    if (
        not metrics
        or len(set(metrics)) != len(metrics)
        or set(metrics) - {"reap", "weighted_ean_sum"}
    ):
        raise ValueError("invalid or duplicate scoring metrics")
    calibration, cal_records = load_corpus_batches(corpus_path, "calibration")
    validation, val_records = load_corpus_batches(corpus_path, "validation")
    if not calibration or not validation:
        raise ValueError("both corpus splits must be nonempty")
    for key in ("id", "group", "token_sha256"):
        if {r[key] for r in cal_records} & {r[key] for r in val_records}:
            raise ValueError(f"calibration/validation {key} overlap")
    # Check the tokens themselves as well as the corpus's claimed digests.
    if {tuple(b["input_ids"].flatten().tolist()) for b in calibration} & {
        tuple(b["input_ids"].flatten().tolist()) for b in validation
    }:
        raise ValueError("calibration/validation token overlap")
    identity = {
        "source": source.identity,
        "corpus_sha256": _file_digest(Path(corpus_path)),
        "keeps": keeps,
        "metrics": metrics,
        "experiment_sha256": _file_digest(Path(__file__)),
        "evaluator_sha256": _file_digest(Path(__file__).with_name("muse_evaluate.py")),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another experiment writer is active") from exc
        path = output / "experiment.json"
        if path.exists():
            if not resume:
                raise ValueError("existing experiment requires --resume")
            state = json.loads(path.read_text())
            if state["identity"] != identity:
                raise ValueError("experiment resume identity differs")
        else:
            if resume:
                raise ValueError("no experiment to resume")
            if any(p.name != ".experiment.lock" for p in output.iterdir()):
                raise ValueError("new experiment needs an empty directory")
            state = {
                "identity": identity,
                "phase": "calibration",
                "complete": False,
                "started_unix": time.time(),
                "results": {},
            }
            _save_json(path, state)

        def replay(model_source, batches, name, observe=False, stop=None):
            state.update(phase=name, complete=False, updated_unix=time.time())
            _save_json(path, state)
            directory = output / name
            return run_calibration(
                model_source,
                batches,
                directory,
                collect_metrics=observe,
                stop_after_layer=stop,
                resume=(directory / "progress.json").exists(),
                progress=lambda row: progress(json.dumps({"phase": name, **row})),
                batch_progress=lambda row: (
                    progress(json.dumps({"phase": name, "event": "batch", **row}))
                    if row["batches_completed"] % 8 == 0
                    or row["batches_completed"] == row["batches"]
                    else None
                ),
            )

        cal_state = replay(
            source, calibration, "calibration", True, stop_after_calibration_layer
        )
        if not cal_state["complete"]:
            return state
        # Freeze every candidate before touching validation data.
        plans = {}
        for metric in metrics:
            for keep in keeps:
                name = f"{metric}-{keep}"
                plan = build_pruning_plan(source, output / "calibration", metric, keep)
                plan_path = output / "plans" / f"{name}.json"
                if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
                    raise ValueError("saved pruning plan differs")
                _save_json(plan_path, plan)
                plans[name] = plan
        replay(source, validation, "baseline")
        baseline = evaluate_final_activations(source, validation, output / "baseline")
        write_evaluation(output / "baseline" / "evaluation.json", baseline, val_records)
        state["results"]["baseline"] = baseline
        _save_json(path, state)
        progress(
            json.dumps(
                {
                    "phase": "baseline",
                    "mean_nll": baseline["mean_nll"],
                    "perplexity": baseline["perplexity"],
                }
            )
        )
        for name, plan in plans.items():
            candidate = PrunedMuseShardSource(source_root, plan)
            replay(candidate, validation, name)
            result = evaluate_final_activations(
                candidate, validation, output / name, baseline=output / "baseline"
            )
            write_evaluation(output / name / "evaluation.json", result, val_records)
            state["results"][name] = result
            _save_json(path, state)
            progress(
                json.dumps(
                    {
                        "phase": name,
                        "mean_nll": result["mean_nll"],
                        "delta_mean_nll": result["delta_mean_nll"],
                        "kl": result["baseline_to_candidate_kl"],
                    }
                )
            )
        state.update(phase="complete", complete=True, updated_unix=time.time())
        _save_json(path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--keeps", type=int, nargs="+", default=[128, 144])
    parser.add_argument("--metrics", nargs="+", default=["reap", "weighted_ean_sum"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-calibration-layer", type=int)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    state = run_experiment(
        args.source,
        args.corpus,
        args.output,
        keeps=args.keeps,
        metrics=args.metrics,
        resume=args.resume,
        stop_after_calibration_layer=args.stop_after_calibration_layer,
        progress=lambda message: print(message, flush=True),
    )
    print(
        json.dumps({"phase": state["phase"], "complete": state["complete"]}), flush=True
    )


if __name__ == "__main__":
    main()
