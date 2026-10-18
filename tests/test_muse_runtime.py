"""Compare the serving equations and cache to independent native Muse forwards."""

import importlib.util
import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

if importlib.util.find_spec("transformers.models.muse_spark") is None:
    pytest.skip("requires native Muse", allow_module_level=True)
# ModelSlim patches Pydantic globally; defer importing it until quantization.
if importlib.util.find_spec("msmodelslim") is None:
    pytest.skip("requires ModelSlim", allow_module_level=True)

from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
from transformers.models.muse_spark.modeling_muse_spark import (
    MuseSparkTextModel,
    MuseSparkTextRotaryEmbedding,
)
from reap.muse_export import export_checkpoint, dequantize_weight
from reap.muse_runtime import MuseRuntime, configuration, placement_plan, w8_linear
from tests.test_muse_export import export_fixture
from tests.test_muse_streaming import cpu_threads  # noqa: F401


@pytest.fixture
def exported(tmp_path):
    source, plan, _ = export_fixture(tmp_path)
    output = tmp_path / "export"
    export_checkpoint(source, plan, output, progress=lambda _: None)
    return output


def native_reference(root):
    raw = json.loads((root / "config.json").read_text())
    c = MuseSparkTextConfig.from_dict(raw["text_config"])
    c._attn_implementation = c._experts_implementation = "eager"
    with torch.device("meta"):
        native = MuseSparkTextModel(c)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    exported = {}
    for shard in set(index.values()):
        exported.update(load_file(root / shard))
    values = {}
    for name in native.state_dict():
        key = "model.language_model." + name
        if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
            prefix, bank = key.rsplit(".", 1)
            banks = []
            for i in range(c.num_local_experts):
                projections = (
                    ["gate_proj", "up_proj"]
                    if bank == "gate_up_proj"
                    else ["down_proj"]
                )
                banks.append(
                    torch.cat(
                        [
                            dequantize_weight(
                                exported[f"{prefix}.{i}.{p}.weight"],
                                exported[f"{prefix}.{i}.{p}.weight_scale"],
                            ).half()
                            for p in projections
                        ],
                        dim=0,
                    )
                )
            values[name] = torch.stack(banks)
        elif key + "_scale" in exported:
            values[name] = dequantize_weight(
                exported[key], exported[key + "_scale"], bits=8
            ).half()
        else:
            values[name] = exported[key]
    native.load_state_dict(values, assign=True, strict=True)
    native.rotary_emb = MuseSparkTextRotaryEmbedding(c)
    native.eval()
    return native, exported["lm_head.weight"]


def logits(native, head, ids):
    with torch.inference_mode():
        hidden = native(torch.tensor([ids]), use_cache=False).last_hidden_state
        raw = F.linear(hidden, head) * native.config.output_multiplier
        return (
            torch.tanh(raw / native.config.final_logit_softcapping)
            * native.config.final_logit_softcapping
        )


def test_all_layers_and_bounded_cache_match_native(exported):
    native, head = native_reference(exported)
    runtime = MuseRuntime(exported, max_context=32, progress=lambda _: None)
    ids = [1, 4, 7, 8, 9, 10, 12, 3, 2]
    expected = logits(native, head, ids)
    actual = runtime.forward(ids, all_logits=True)
    torch.testing.assert_close(actual, expected, rtol=0.004, atol=0.0004)
    runtime.reset()
    chunks = []
    start = 0
    for size in (3, 1, 4, 1):
        chunks.append(runtime.forward(ids[start : start + size], all_logits=True))
        start += size
    torch.testing.assert_close(
        torch.cat(chunks, dim=1), expected, rtol=0.004, atol=0.0004
    )
    assert [cache.key.shape[-2] for cache in runtime.cache] == [3, 9, 3]
    assert all(cache.seen == 9 for cache in runtime.cache)
    runtime.reset()
    assert runtime.seen == 0 and all(c.key is None for c in runtime.cache)


def test_generation_matches_native_greedy_and_resets(exported):
    native, head = native_reference(exported)
    runtime = MuseRuntime(
        exported, max_context=16, prefill_chunk=2, progress=lambda _: None
    )
    prompt = [1, 7, 4, 9, 3]
    ids = prompt[:]
    for _ in range(3):
        ids.append(int(logits(native, head, ids)[0, -1].argmax()))
    assert runtime.generate(prompt, 3) == ids[len(prompt) :]
    assert runtime.generate(prompt, 3) == ids[len(prompt) :]
    assert (
        runtime.generate(prompt, 3, eos=[ids[len(prompt)]])
        == ids[len(prompt) : len(prompt) + 1]
    )
    with pytest.raises(ValueError, match="context"):
        runtime.generate(prompt, 20)


def test_placement_and_integrity_guards(exported):
    plan = placement_plan(exported, ["cpu", "cpu"], max_context=8, reserve_gib=0)
    assert plan[0]["first_layer"] == 0 and plan[-1]["last_layer_exclusive"] == 3
    assert plan[0]["last_layer_exclusive"] == plan[1]["first_layer"]
    assert all(p["required_bytes"] > 0 for p in plan)
    path = exported / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"].pop(next(iter(index["weight_map"])))
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index"):
        MuseRuntime(exported)
    raw = json.loads((exported / "config.json").read_text())
    raw["text_config"]["layer_rope_theta"][0] = 1234
    with pytest.raises(ValueError, match="attention"):
        configuration(raw)


def test_w8_chunking_matches_direct_dequantization():
    torch.manual_seed(4)
    x = torch.randn(1, 3, 32).half()
    codes = torch.randint(-127, 128, (27, 32), dtype=torch.int8)
    scales = torch.rand(27, 1) * 0.01
    expected = F.linear(x, (codes.float() * scales).half())
    torch.testing.assert_close(
        w8_linear(x, codes, scales, row_chunk=7), expected, rtol=0, atol=0
    )


def test_nonfinite_logits_fail_instead_of_generating(exported):
    runtime = MuseRuntime(exported, max_context=16, progress=lambda _: None)
    runtime.globals["lm_head.weight"] = torch.full_like(
        runtime.globals["lm_head.weight"], float("nan")
    )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        runtime.forward([1, 4])
    runtime.reset()
    assert runtime.seen == 0 and all(cache.key is None for cache in runtime.cache)


def test_optimized_placement_is_available_offline_and_accounts_for_weights(exported):
    plain = placement_plan(exported, ["npu:0", "npu:1"], max_context=8, reserve_gib=0)
    fast = placement_plan(
        exported, ["npu:0", "npu:1"], max_context=8, reserve_gib=0, optimized=True
    )
    assert sum(x["required_bytes"] for x in fast) > sum(
        x["required_bytes"] for x in plain
    )
    with pytest.raises(ValueError, match="only once"):
        placement_plan(exported, ["npu:00", "npu:0"])
