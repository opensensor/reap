"""Whole native Muse text/logit parity, candidate slicing, and artifact checks."""

import copy
import importlib.util
import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

if importlib.util.find_spec("transformers.models.muse_spark") is None:
    pytest.skip("requires the Muse-capable environment", allow_module_level=True)

from transformers.models.muse_spark.configuration_muse_spark import MuseSparkConfig
from transformers.models.muse_spark.modeling_muse_spark import (
    MuseSparkForConditionalGeneration,
)

from tests.test_muse_streaming import checkpoint, cpu_threads  # noqa: F401
from reap.models.muse_spark import prune_muse_experts
from reap.muse_streaming import MuseShardSource, run_calibration
from reap.muse_evaluate import (
    PrunedMuseShardSource,
    build_pruning_plan,
    evaluate_final_activations,
    muse_logits,
)


def native_checkpoint(tmp_path, dtype):
    text, root, batches = checkpoint(tmp_path, dtype)
    config = MuseSparkConfig(
        text_config=text.config.to_dict(),
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "pos_emb_height": 2,
            "pos_emb_width": 2,
        },
        audio_config={
            "hidden_sizes": [16],
            "depths": [1],
            "patch_sizes": [2],
            "num_hidden_layers": 1,
            "out_hidden_size": 16,
        },
        out_hidden_size=64,
        projector_hidden_size=16,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = MuseSparkForConditionalGeneration(config).to(dtype).eval()
    model.model.language_model = text
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["lm_head.weight"] = "head.safetensors"
    save_file(
        {"lm_head.weight": model.lm_head.weight.contiguous()}, root / "head.safetensors"
    )
    index_path.write_text(json.dumps(index))
    return model, root, batches


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_logits_loss_and_pruned_forward_match_native(tmp_path, dtype):
    model, root, batches = native_checkpoint(tmp_path, dtype)
    source = MuseShardSource(root)
    calibration, baseline, candidate = [
        tmp_path / name for name in ("calibration", "baseline", "candidate")
    ]
    run_calibration(source, batches, calibration)
    run_calibration(source, batches, baseline, collect_metrics=False)
    plan = build_pruning_plan(source, calibration, "weighted_ean_sum", 3)
    pruned_source = PrunedMuseShardSource(root, plan)
    run_calibration(
        pruned_source, batches, candidate, collect_metrics=False, stop_after_layer=1
    )
    run_calibration(
        PrunedMuseShardSource(root, plan),
        batches,
        candidate,
        collect_metrics=False,
        resume=True,
    )
    base_logits, candidate_logits, labels = [], [], []
    with torch.inference_mode():
        for i, batch in enumerate(batches):
            logits = model(**batch, use_cache=False).logits
            hidden = load_file(
                baseline / "activations-final" / f"batch-{i:05d}.safetensors"
            )["hidden"]
            torch.testing.assert_close(
                muse_logits(model.lm_head, hidden, source.config),
                logits,
                rtol=0,
                atol=0,
            )
            mask = batch.get(
                "attention_mask", torch.ones_like(batch["input_ids"])
            ).bool()
            valid = mask[:, :-1] & mask[:, 1:]
            base_logits.append(logits[:, :-1][valid].float())
            labels.append(batch["input_ids"][:, 1:][valid])
        for index, layer in enumerate(model.model.language_model.layers):
            prune_muse_experts(layer.mlp, plan["retained_experts"][str(index)])
        for i, batch in enumerate(batches):
            logits = model(**batch, use_cache=False).logits
            hidden = load_file(
                candidate / "activations-final" / f"batch-{i:05d}.safetensors"
            )["hidden"]
            torch.testing.assert_close(
                muse_logits(model.lm_head, hidden, source.config),
                logits,
                rtol=0,
                atol=0,
            )
            mask = batch.get(
                "attention_mask", torch.ones_like(batch["input_ids"])
            ).bool()
            candidate_logits.append(logits[:, :-1][mask[:, :-1] & mask[:, 1:]].float())
    p, q, labels = (
        torch.cat(base_logits),
        torch.cat(candidate_logits),
        torch.cat(labels),
    )
    expected_nll = F.cross_entropy(q, labels).item()
    expected_kl = (
        (p.log_softmax(-1).exp() * (p.log_softmax(-1) - q.log_softmax(-1)))
        .sum(-1)
        .mean()
        .item()
    )
    for chunk in (1, 4, 128):
        result = evaluate_final_activations(
            pruned_source, batches, candidate, baseline=baseline, chunk_tokens=chunk
        )
        assert result["scored_tokens"] == len(labels) == 18
        assert result["mean_nll"] == pytest.approx(expected_nll, abs=2e-6)
        assert result["baseline_to_candidate_kl"] == pytest.approx(
            expected_kl, abs=2e-6
        )
        assert result["baseline_mean_nll"] == pytest.approx(
            F.cross_entropy(p, labels).item(), abs=2e-6
        )
        assert result["top1_agreement"] == pytest.approx(
            (p.argmax(-1) == q.argmax(-1)).float().mean().item()
        )
        assert result["logit_rmse"] == pytest.approx(
            (p - q).square().mean().sqrt().item(), abs=2e-6
        )
    same = evaluate_final_activations(source, batches, baseline, baseline=baseline)
    assert (
        same["delta_mean_nll"]
        == same["baseline_to_candidate_kl"]
        == same["logit_rmse"]
        == 0
    )
    assert same["top1_agreement"] == 1
    assert source.active is None


@pytest.mark.parametrize(
    "change",
    [
        "incomplete",
        "disabled",
        "score_corruption",
        "source",
        "missing_ids",
        "duplicate_ids",
    ],
)
def test_invalid_calibration_or_plan_rejected(tmp_path, change):
    _, root, batches = native_checkpoint(tmp_path, torch.float32)
    source = MuseShardSource(root)
    output = tmp_path / "cal"
    run_calibration(
        source,
        batches,
        output,
        collect_metrics=change != "disabled",
        stop_after_layer=1 if change == "incomplete" else None,
    )
    if change == "score_corruption":
        path = output / "scores" / "layer-00000.safetensors"
        scores = load_file(path)
        scores["reap"] += 1
        save_file(scores, path)
    if change in ("incomplete", "disabled", "score_corruption"):
        with pytest.raises(ValueError, match="calibration|checksum"):
            build_pruning_plan(source, output, "reap", 3)
        return
    plan = build_pruning_plan(source, output, "reap", 3)
    if change == "source":
        plan["source"] = {"bad": True}
    elif change == "missing_ids":
        del plan["retained_experts"]["0"]
    else:
        plan["retained_experts"]["0"] = [0, 0, 1]
    with pytest.raises(ValueError, match="identity|cover|IDs"):
        PrunedMuseShardSource(root, plan)


def test_plan_change_invalidates_resume_and_evaluation_checks_inputs(tmp_path):
    _, root, batches = native_checkpoint(tmp_path, torch.float32)
    source = MuseShardSource(root)
    calibration = tmp_path / "cal"
    run_calibration(source, batches, calibration)
    plan = build_pruning_plan(source, calibration, "reap", 3)
    original = PrunedMuseShardSource(root, plan)
    candidate = tmp_path / "candidate"
    run_calibration(original, batches, candidate, collect_metrics=False)
    changed = copy.deepcopy(plan)
    changed["retained_experts"]["0"] = next(
        x for x in ([0, 1, 2], [1, 2, 3]) if x != plan["retained_experts"]["0"]
    )
    with pytest.raises(ValueError, match="identity"):
        run_calibration(
            PrunedMuseShardSource(root, changed),
            batches,
            candidate,
            collect_metrics=False,
            resume=True,
        )
    batches[0]["input_ids"][0, 2] = 14
    with pytest.raises(ValueError, match="matching"):
        evaluate_final_activations(original, batches, candidate)


def test_no_valid_next_token_pairs_rejected(tmp_path):
    _, root, _ = native_checkpoint(tmp_path, torch.float32)
    batches = [
        {
            "input_ids": torch.tensor([[1, 0, 3]]),
            "attention_mask": torch.tensor([[1, 0, 1]]),
        }
    ]
    source = MuseShardSource(root)
    output = tmp_path / "run"
    run_calibration(source, batches, output, collect_metrics=False)
    with pytest.raises(ValueError, match="next-token"):
        evaluate_final_activations(source, batches, output)
