"""Native Muse integration tests; run in the documented Muse environment."""

import fcntl
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

if importlib.util.find_spec("transformers.models.muse_spark") is None:
    pytest.skip(
        "requires the Muse-capable Transformers test environment",
        allow_module_level=True,
    )

from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
from transformers.models.muse_spark.modeling_muse_spark import MuseSparkTextModel

from reap import muse_streaming as streaming
from reap.muse_streaming import MuseShardSource, run_calibration
from reap.observer import MoETransformerObserver, MuseSparkObserverHookConfig


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def checkpoint(tmp_path, dtype=torch.float32):
    torch.manual_seed(531)
    config = MuseSparkTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        moe_hidden_size=8,
        moe_intermediate_size=12,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        layer_rope_theta=[500000.0, 0, 500000.0],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = MuseSparkTextModel(config).to(dtype).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("mlp.gate.weight") or "residual_gate.gate" in name:
                p.data = p.data.float()
        for layer in model.layers:
            layer.mlp.gate.e_score_correction_bias = torch.tensor(
                [-0.12, 0.04, -0.01, 0.08]
            )
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "muse_spark", "text_config": config.to_dict()})
    )
    tensors = {"model.language_model." + k: v for k, v in model.state_dict().items()}
    tensors["model.vision_model.unrelated.weight"] = torch.ones(7, 9)
    shards = [{}, {}, {}]
    index = {}
    for i, (key, value) in enumerate(tensors.items()):
        shard = i % 3
        shards[shard][key] = value.contiguous()
        index[key] = f"model-{shard:03d}.safetensors"
    for i, values in enumerate(shards):
        save_file(values, root / f"model-{i:03d}.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )
    batches = [
        {
            "input_ids": torch.tensor(
                [[0, 0, 1, 4, 7, 8, 9, 10, 2], [1, 5, 8, 7, 2, 0, 0, 0, 0]]
            ),
            "attention_mask": torch.tensor(
                [[0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0]]
            ),
        },
        {
            "input_ids": torch.tensor([[1, 8, 6, 3, 10, 12, 5, 8, 2]]),
            "position_ids": torch.arange(11, 20).unsqueeze(0),
        },
    ]
    return model, root, batches


def native_reference(model, batches):
    observer = MoETransformerObserver(model, MuseSparkObserverHookConfig())
    outputs = []
    with torch.inference_mode():
        for batch in batches:
            with observer.set_attention_mask(batch.get("attention_mask")):
                outputs.append(
                    model(**batch, use_cache=False).last_hidden_state.clone()
                )
    scores = observer.report_state()
    observer.close_hooks()
    return outputs, scores


def assert_results(output, expected, scores):
    for i, value in enumerate(expected):
        actual = load_file(output / "activations-final" / f"batch-{i:05d}.safetensors")[
            "hidden"
        ]
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    for layer, values in scores.items():
        actual = load_file(output / "scores" / f"layer-{layer:05d}.safetensors")
        for key, value in values.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streaming_matches_native_full_forward_and_metrics(tmp_path, dtype):
    model, root, batches = checkpoint(tmp_path, dtype)
    expected, scores = native_reference(model, batches)
    source = MuseShardSource(root)
    output = tmp_path / "run"
    state = run_calibration(source, batches, output)
    assert state["complete"] and state["next_layer"] == 3
    assert (
        state["tokens"] == 21
    )  # Original padding mask, not the final sliding-window row.
    assert source.active is None
    assert len(source.load_log) == 5  # embedding, three layers, final norm
    assert all("vision" not in row["prefix"] for row in source.load_log)
    assert [p.name for p in output.glob("activations-*")] == ["activations-final"]
    assert_results(output, expected, scores)


def test_shard_reader_preserves_dtypes_and_releases_before_next_layer(tmp_path):
    _, root, _ = checkpoint(tmp_path, torch.bfloat16)
    source = MuseShardSource(root)
    with source.layer(0) as layer:
        assert layer.mlp.experts.down_proj.dtype == torch.bfloat16
        assert layer.mlp.gate.weight.dtype == torch.float32
        assert layer.post_attention_residual_gate.gate.dtype == torch.float32
        with pytest.raises(RuntimeError, match="release"), source.layer(1):
            pass
    assert all(p.device.type == "meta" for p in layer.parameters())
    assert source.active is None
    with source.layer(1) as second:
        assert second.mlp.gate.weight.device.type == "cpu"
    assert all(p.device.type == "meta" for p in second.parameters())


def test_resume_loads_only_remaining_layers_and_matches_uninterrupted(tmp_path):
    model, root, batches = checkpoint(tmp_path)
    expected, scores = native_reference(model, batches)
    output = tmp_path / "run"
    first = run_calibration(MuseShardSource(root), batches, output, stop_after_layer=1)
    assert first["next_layer"] == 1 and not first["complete"]
    source = MuseShardSource(root)
    state = run_calibration(source, batches, output, resume=True)
    assert state["complete"]
    assert [row["prefix"] for row in source.load_log] == [
        "model.language_model.layers.1.",
        "model.language_model.layers.2.",
        "model.language_model.norm.",
    ]
    assert_results(output, expected, scores)
    source = MuseShardSource(root)
    assert run_calibration(source, batches, output, resume=True)["complete"]
    assert not source.load_log


@pytest.mark.parametrize("stage", ["input", "layer-00001", "final"])
def test_interruption_recomputes_uncommitted_stage(tmp_path, monkeypatch, stage):
    model, root, batches = checkpoint(tmp_path)
    expected, scores = native_reference(model, batches)
    output = tmp_path / "run"
    source = MuseShardSource(root)
    save = streaming._save_tensors

    def fail(path, values):
        if (
            f"activations-{stage}" in str(path)
            and path.name == "batch-00001.safetensors"
        ):
            raise RuntimeError("simulated interruption")
        save(path, values)

    monkeypatch.setattr(streaming, "_save_tensors", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        run_calibration(source, batches, output)
    assert source.active is None
    monkeypatch.setattr(streaming, "_save_tensors", save)
    assert run_calibration(MuseShardSource(root), batches, output, resume=True)[
        "complete"
    ]
    assert_results(output, expected, scores)


@pytest.mark.parametrize("change", ["tokens", "source", "activation", "scores"])
def test_resume_rejects_changed_inputs_or_artifacts(tmp_path, change):
    _, root, batches = checkpoint(tmp_path)
    output = tmp_path / "run"
    run_calibration(MuseShardSource(root), batches, output, stop_after_layer=1)
    if change == "tokens":
        batches[0]["input_ids"][0, 2] = 12
    elif change == "source":
        path = root / "config.json"
        path.write_text(path.read_text() + " ")
    else:
        path = (
            output / "activations-layer-00000" / "batch-00000.safetensors"
            if change == "activation"
            else output / "scores" / "layer-00000.safetensors"
        )
        tensors = load_file(path)
        key = next(iter(tensors))
        tensors[key] = tensors[key] + 1
        save_file(tensors, path)
    with pytest.raises(ValueError, match="identity|checksum"):
        run_calibration(MuseShardSource(root), batches, output, resume=True)


def test_output_guard_and_concurrent_writer_lock(tmp_path):
    _, root, batches = checkpoint(tmp_path)
    source = MuseShardSource(root)
    with pytest.raises(ValueError, match="outside"):
        run_calibration(source, batches, root / "output")
    output = tmp_path / "run"
    output.mkdir()
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer"):
            run_calibration(source, batches, output)
    assert not source.load_log


def test_shard_paths_cannot_escape_checkpoint(tmp_path):
    _, root, _ = checkpoint(tmp_path)
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"]["model.language_model.norm.weight"] = "../outside.safetensors"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="within"):
        MuseShardSource(root)


def test_crash_after_commit_retires_old_stage_and_partial_files(tmp_path, monkeypatch):
    model, root, batches = checkpoint(tmp_path)
    expected, scores = native_reference(model, batches)
    output = tmp_path / "run"
    retire = streaming._retire_stage

    def fail(*args):
        raise RuntimeError("crash after commit")

    monkeypatch.setattr(streaming, "_retire_stage", fail)
    with pytest.raises(RuntimeError, match="after commit"):
        run_calibration(MuseShardSource(root), batches, output)
    assert json.loads((output / "progress.json").read_text())["next_layer"] == 1
    partial = output / "activations-input" / ".partial-ab123_cd"
    partial.write_bytes(b"interrupted file write")
    monkeypatch.setattr(streaming, "_retire_stage", retire)
    run_calibration(MuseShardSource(root), batches, output, resume=True)
    assert not partial.exists()
    assert [p.name for p in output.glob("activations-*")] == ["activations-final"]
    assert_results(output, expected, scores)
