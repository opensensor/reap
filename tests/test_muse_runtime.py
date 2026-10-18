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
    emitted = []
    assert (
        runtime.generate(prompt, 3, on_token=emitted.append)
        == emitted
        == ids[len(prompt) :]
    )
    import threading

    cancel = threading.Event()

    def stop(token):
        cancel.set()

    with pytest.raises(InterruptedError):
        runtime.generate(prompt, 3, on_token=stop, cancel=cancel)
    runtime.reset()
    assert runtime.seen == 0
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


def test_bounded_placement_includes_sliding_append_capacity(exported):
    config = configuration(json.loads((exported / "config.json").read_text()))
    common = dict(max_context=16, reserve_gib=0, optimized=True)
    plain = placement_plan(exported, ["cpu"], **common)[0]["required_bytes"]
    fused = placement_plan(
        exported, ["cpu"], bounded_attention=True, prefill_chunk=4, **common
    )[0]["required_bytes"]
    sliding_layers = config.layer_types.count("sliding_attention")
    extra_tokens = min(16, config.sliding_window - 1 + 4) - min(
        16, config.sliding_window - 1
    )
    assert (
        fused - plain
        == sliding_layers
        * extra_tokens
        * 2
        * config.num_key_value_heads
        * config.head_dim
        * 2
    )
    with pytest.raises(ValueError, match="invalid placement"):
        placement_plan(
            exported,
            ["cpu"],
            max_context=65537,
            optimized=False,
            bounded_attention=True,
        )


def test_large_prefill_requires_bounded_optimized_execution(exported):
    plan = placement_plan(
        exported,
        ["npu:0"],
        max_context=1024,
        prefill_chunk=512,
        optimized=True,
        bounded_attention=True,
    )
    assert plan[0]["last_layer_exclusive"] == 3
    for optimized, bounded, chunk in (
        (False, False, 512),
        (True, False, 512),
        (True, True, 513),
    ):
        with pytest.raises(ValueError, match="invalid placement"):
            placement_plan(
                exported,
                ["npu:0"],
                prefill_chunk=chunk,
                optimized=optimized,
                bounded_attention=bounded,
            )


def test_prefix_reuse_preserves_outputs_after_sliding_rollover(exported):
    from types import MethodType
    from reap.muse_fast import FastMuseLayer

    def make(enabled):
        runtime = MuseRuntime(
            exported, max_context=32, prefill_chunk=4, progress=lambda _: None
        )
        # Use the real bounded attention/cache path with the CPU reference
        # projections; optimized packed kernels are separately tested on NPUs.
        runtime.bounded_attention = True
        runtime.prefix_cache_enabled = enabled
        for layer in runtime.layers:
            layer.bounded_attention = True
            layer.max_context = 32
            layer.prefill_chunk = 4
            layer.attention = MethodType(FastMuseLayer.attention, layer)
        runtime.reset()
        runtime.logit_trace = []
        original_forward = runtime.forward

        def traced_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            runtime.logit_trace.append(result.clone())
            return result

        runtime.forward = traced_forward
        return runtime

    cached, cold = make(True), make(False)
    cases = [
        ([1, 4, 7, 8, 9, 10, 12, 3, 2], 0),
        ([1, 4, 7, 8, 9, 10, 12, 3, 5, 6], 8),
        ([1, 4, 7, 8, 9, 10, 12, 3], 8),
        ([1, 5, 7, 8, 9, 10, 12, 3], 0),
        ([1, 4, 7], 0),
        ([1, 4, 7], 3),
        ([1, 4, 7, 8], 0),
    ]
    for prompt, reused in cases:
        cached.logit_trace.clear()
        cold.logit_trace.clear()
        assert cached.generate(prompt, 6) == cold.generate(prompt, 6)
        actual_logits = cached.logit_trace[-6:]
        if len(actual_logits) == 5:
            actual_logits = [cached.prefix_snapshot.logits, *actual_logits]
        for actual, expected in zip(actual_logits, cold.logit_trace[-6:], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert cached.generation_stats["cached_prompt_tokens"] == reused
        assert cold.generation_stats["cached_prompt_tokens"] == 0
        # Every request decodes far enough to move the small sliding window.
        assert all(c.seen > cached.c.sliding_window for c in cached.cache)
        cached.release_request_cache()
        assert cached.seen == 0 and all(c.key is None for c in cached.cache)
        assert cached.prefix_snapshot is not None
    cached.reset()
    assert cached.prefix_snapshot is None
    assert cached.generate(cases[-1][0], 6) == cold.generate(cases[-1][0], 6)
    assert cached.generation_stats["cached_prompt_tokens"] == 0


def test_prefix_placement_reserves_both_sliding_snapshots(exported):
    config = configuration(json.loads((exported / "config.json").read_text()))
    common = dict(
        max_context=16,
        reserve_gib=0,
        optimized=True,
        bounded_attention=True,
        prefill_chunk=4,
    )
    plain = placement_plan(exported, ["cpu"], **common)[0]["required_bytes"]
    cached = placement_plan(exported, ["cpu"], prefix_cache=True, **common)[0][
        "required_bytes"
    ]
    expected = (
        2
        * config.layer_types.count("sliding_attention")
        * min(16, config.sliding_window - 1)
        * 2
        * config.num_key_value_heads
        * config.head_dim
        * 2
    )
    assert cached - plain == expected
    with pytest.raises(ValueError, match="bounded attention"):
        placement_plan(exported, ["cpu"], prefix_cache=True)


def test_fused_layer_graphs_preserve_static_dependencies_across_requests(
    exported, monkeypatch
):
    from pathlib import Path
    from types import SimpleNamespace
    from reap.muse_attention import AppendCache
    from reap.muse_fast import FastMuseLayer
    from reap.muse_tensor import DecodeGraph

    class Replay:
        # Emulate graph storage identities so stale cross-graph dependencies
        # fail when inputs/routes change, rather than simply rerunning helpers.
        def __init__(self, function, inputs, routes):
            self.function = function
            self.inputs, self.routes = inputs.clone(), routes.clone()

            def clone(value):
                return (
                    tuple(clone(v) for v in value)
                    if isinstance(value, tuple)
                    else value.clone()
                )

            self.output = clone(function(self.inputs, self.routes))

        def run(self, inputs, routes):
            self.inputs.copy_(inputs)
            self.routes.copy_(routes)
            result = self.function(self.inputs, self.routes)

            def copy(a, b):
                if isinstance(a, tuple):
                    for av, bv in zip(a, b, strict=True):
                        copy(av, bv)
                else:
                    a.copy_(b)

            copy(self.output, result)
            return self.output

    monkeypatch.setattr(DecodeGraph, "capture", Replay)
    scope = {}
    exec(
        (
            Path(__file__).parents[1]
            / "scripts/muse_resident_patches/layer_decode_graphs.py"
        ).read_text(),
        scope,
    )
    scope["pipelined_project"] = lambda experts, inputs, ids: experts.project(
        inputs, ids=ids
    )
    runtime = MuseRuntime(exported, max_context=32, progress=lambda _: None)

    def make(original):
        layer = FastMuseLayer(
            original.c, original.index, original.w, SimpleNamespace(backend=True), None
        )
        layer.bounded_attention, layer.max_context, layer.prefill_chunk = True, 32, 4

        def project(inputs, *, ids=None, ends=None):
            if ends is not None:
                ids = torch.repeat_interleave(
                    torch.arange(len(ends)),
                    ends.diff(prepend=torch.zeros(1, dtype=ends.dtype)),
                )
            values = []
            for i, expert in enumerate(ids.tolist()):
                prefix = f"mlp.experts.{expert}."
                x = inputs[i : i + 1]
                gate = original.linear(prefix + "gate_proj", x, expert=True)
                up = original.linear(prefix + "up_proj", x, expert=True)
                values.append(
                    original.linear(
                        prefix + "down_proj", F.silu(gate) * up, expert=True
                    )
                )
            return torch.cat(values)

        layer.tensor_experts = SimpleNamespace(graphs={}, project=project)
        return layer

    torch.manual_seed(731)
    with torch.inference_mode():
        for original in runtime.layers:
            baseline, candidate = make(original), make(original)
            for request in range(3):
                if request == 2:
                    candidate.tensor_experts.graphs.clear()
                caches = AppendCache(), AppendCache()
                for start, count in [(0, 4), *[(i, 1) for i in range(4, 13)]]:
                    x = (torch.randn(1, count, original.c.hidden_size) * 0.1).half()
                    expected = baseline(x, start, caches[0])
                    actual = scope["fused_layer"](candidate, x, start, caches[1])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
