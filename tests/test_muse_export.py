"""ModelSlim arithmetic, real native shapes, sliced metadata and export recovery."""

import importlib.util
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

pytest.importorskip("msmodelslim")
if importlib.util.find_spec("transformers.models.muse_spark") is None:
    pytest.skip("requires native Muse", allow_module_level=True)

from transformers.models.muse_spark.modeling_muse_spark import MuseSparkTextModel
from tests.test_muse_evaluate import native_checkpoint
from tests.test_muse_streaming import cpu_threads  # noqa: F401
from reap import muse_export
from reap.muse_export import quantize_weight, dequantize_weight, export_checkpoint
from reap.muse_evaluate import build_pruning_plan
from reap.muse_streaming import MuseShardSource, run_calibration


@pytest.mark.parametrize("bits", [4, 8])
def test_modelslim_quantization_matches_independent_reference(bits):
    torch.manual_seed(361)
    weight = torch.randn(9, 256) * 0.03
    weight[0] = 0
    values = quantize_weight(weight, bits=bits)
    shape = (9, 2, 128) if bits == 4 else (9, 1, 256)
    groups = weight.reshape(shape)
    scale = (groups.abs().amax(-1) / (2 ** (bits - 1) - 1)).clamp_min(
        torch.finfo(torch.float32).eps
    )
    codes = (
        (groups / scale.unsqueeze(-1))
        .round()
        .clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    )
    stored_scale = scale.half() if bits == 4 else scale
    expected = (codes * stored_scale.float().unsqueeze(-1)).reshape(9, 256)
    actual = dequantize_weight(values["weight"], values["weight_scale"], bits=bits)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert values["weight"].shape == (9, 128 if bits == 4 else 256)
    if bits == 4:
        raw = values["weight"].view(torch.uint8)
        integer = codes.reshape(9, 256).to(torch.int8)
        torch.testing.assert_close(raw & 15, (integer[:, ::2] & 15).to(torch.uint8))
        torch.testing.assert_close(raw >> 4, (integer[:, 1::2] & 15).to(torch.uint8))


def export_fixture(tmp_path):
    original, root, batches = native_checkpoint(tmp_path, torch.float32)
    c = original.config.text_config
    c.moe_hidden_size = c.moe_intermediate_size = 128
    model = MuseSparkTextModel(c).eval()
    values = {"model.language_model." + k: v for k, v in model.state_dict().items()}
    values["lm_head.weight"] = original.lm_head.weight
    # Real source contains mixed BF16 and FP32, including FP32 router/gates.
    values = {
        k: v.contiguous()
        if ".mlp.gate." in k or "residual_gate" in k
        else v.bfloat16().contiguous()
        for k, v in values.items()
    }
    save_file(values, root / "fixture.safetensors")
    (root / "config.json").write_text(
        json.dumps({"model_type": "muse_spark", "text_config": c.to_dict()})
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "fixture.safetensors" for k in values}})
    )
    calibration = tmp_path / "calibration"
    source = MuseShardSource(root)
    run_calibration(source, batches, calibration)
    plan = build_pruning_plan(source, calibration, "weighted_ean_sum", 2)
    return root, plan, values


def test_export_slices_routes_preserves_norm_and_resumes(tmp_path):
    root, plan, original = export_fixture(tmp_path)
    output = tmp_path / "export"
    first = export_checkpoint(
        root, plan, output, stop_after_layer=1, progress=lambda _: None
    )
    assert not first["complete"] and first["next_layer"] == 1
    assert not (output / "model.safetensors.index.json").exists()
    state = export_checkpoint(root, plan, output, resume=True, progress=lambda _: None)
    assert state["complete"] and state["next_layer"] == 3
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["num_local_experts"] == 2
    assert config["text_config"]["num_experts_per_tok"] == 2
    index = json.loads((output / "model.safetensors.index.json").read_text())
    count = 0
    for filename, receipt in state["shards"].items():
        tensors = load_file(output / filename)
        for name, value in tensors.items():
            assert index["weight_map"][name] == filename
            count += value.numel() * value.element_size()
    assert count == index["metadata"]["total_size"]
    prefix = "model.language_model.layers.0."
    dense = load_file(output / "dense-000.safetensors")
    ids = plan["retained_experts"]["0"]
    torch.testing.assert_close(
        dense[prefix + "mlp.gate.weight"],
        original[prefix + "mlp.gate.weight"][ids],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        dense[prefix + "mlp.gate.e_score_correction_bias"],
        original[prefix + "mlp.gate.e_score_correction_bias"][ids],
        atol=0,
        rtol=0,
    )
    norm = prefix + "mlp.experts.post_expert_norm.weight"
    torch.testing.assert_close(dense[norm], original[norm].half(), atol=0, rtol=0)
    for new_id, old_id in enumerate(ids):
        expert = load_file(output / f"experts-000-ep{new_id}.safetensors")
        for projection, weight in (
            ("gate_proj", original[prefix + "mlp.experts.gate_up_proj"][old_id, :128]),
            ("up_proj", original[prefix + "mlp.experts.gate_up_proj"][old_id, 128:]),
            ("down_proj", original[prefix + "mlp.experts.down_proj"][old_id]),
        ):
            expected = quantize_weight(weight)
            for kind, value in expected.items():
                torch.testing.assert_close(
                    expert[f"{prefix}mlp.experts.{new_id}.{projection}.{kind}"],
                    value,
                    atol=0,
                    rtol=0,
                )
    assert (
        json.loads((output / "deployment.json").read_text())["serving_ready"] is False
    )
    with pytest.raises(ValueError, match="resume"):
        export_checkpoint(root, plan, output)


def test_export_recovers_partial_layer_and_rejects_corruption(tmp_path, monkeypatch):
    root, plan, _ = export_fixture(tmp_path)
    output = tmp_path / "export"
    real_save = muse_export._save_tensors

    def fail(path, values):
        if path.name == "experts-000-ep1.safetensors":
            raise RuntimeError("interrupted export")
        real_save(path, values)

    monkeypatch.setattr(muse_export, "_save_tensors", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        export_checkpoint(root, plan, output, progress=lambda _: None)
    assert not (output / "model.safetensors.index.json").exists()
    monkeypatch.setattr(muse_export, "_save_tensors", real_save)
    export_checkpoint(
        root, plan, output, resume=True, stop_after_layer=1, progress=lambda _: None
    )
    path = output / "experts-000-ep0.safetensors"
    values = load_file(path)
    key = next(iter(values))
    values[key] = values[key] + 1
    save_file(values, path)
    with pytest.raises(ValueError, match="checksum"):
        export_checkpoint(root, plan, output, resume=True)


def test_export_source_and_plan_guards(tmp_path):
    root, plan, _ = export_fixture(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        export_checkpoint(root, plan, root / "output")
    plan["retained_experts"]["0"] = [0, 0]
    with pytest.raises(ValueError, match="IDs"):
        export_checkpoint(root, plan, tmp_path / "export")
    with pytest.raises(ValueError, match="nonfinite"):
        quantize_weight(torch.full((2, 128), float("nan")))
