from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from reap.muse_fast import TiledBank, pack_tiles
from reap.muse_tensor import TensorExperts, make_pool, partitions, slice_bank
from tests.test_muse_fast import decode
from tests.test_muse_resident import runtime
from reap.muse_resident import Resident


def test_balanced_partitions_cover_channels_and_rotate_larger_shards():
    totals = [0] * 6
    for layer in range(6):
        cuts = partitions(4096, 6, layer)
        assert cuts[0][0] == 0 and cuts[-1][1] == 4096
        assert all(a[1] == b[0] for a, b in zip(cuts, cuts[1:]))
        assert sorted(hi - lo for lo, hi in cuts) == [640] * 4 + [768] * 2
        totals = [n + hi - lo for n, (lo, hi) in zip(totals, cuts)]
    assert totals == [4096] * 6
    with pytest.raises(ValueError):
        partitions(256, 6)


def test_sharding_preserves_physical_nz_codes_scales_and_gate_up_order():
    torch.manual_seed(73)
    codes = torch.randint(-128, 128, (8192, 128), dtype=torch.int8)
    scales = (torch.rand(8192, 2) * 0.02).half()
    q, s = pack_tiles(codes)[None], pack_tiles(scales, scales=True)[None]
    bank = TiledBank([(q, s, torch.full_like(s, 8, dtype=torch.int8))], 256, 8192)
    expected = decode(q[0], s[0])
    for lo, hi in partitions(4096, 6, 4):
        shard = slice_bank(bank, [(lo, hi), (4096 + lo, 4096 + hi)], "cpu")
        a, b, offsets = shard.parts[0]
        assert a.is_contiguous() and b.is_contiguous() and (offsets == 8).all()
        assert a.data_ptr() != q.data_ptr()
        torch.testing.assert_close(
            decode(a[0], b[0]),
            torch.cat((expected[lo:hi], expected[4096 + lo : 4096 + hi])),
            rtol=0,
            atol=0,
        )
    with pytest.raises(ValueError, match="NZ"):
        slice_bank(bank, [(1, 129)], "cpu")


@pytest.mark.parametrize("grouped", [False, True])
def test_parallel_gather_preserves_expert_rows_and_uneven_column_order(
    monkeypatch, grouped
):
    # Keep the reference independent of physical W4 packing and NPU dispatch.
    torch.manual_seed(37)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device=lambda _: nullcontext()), raising=False
    )
    x = torch.randn(8, 12)
    gate, up, down = [torch.randn(4, 12, 12) for _ in range(3)]
    ids = torch.tensor([0, 0, 0, 2, 2, 2, 3, 3], dtype=torch.int32)
    ends = torch.tensor([3, 3, 6, 8], dtype=torch.int64)

    class Bank:
        backend = True

        def __init__(self, weight):
            self.weight = weight

        def linear(self, value, *, ids=None, ends=None):
            if ends is not None:
                ids = torch.repeat_interleave(
                    torch.arange(4), ends.diff(prepend=torch.zeros(1, dtype=ends.dtype))
                )
            return torch.cat(
                [F.linear(value[i : i + 1], self.weight[e]) for i, e in enumerate(ids)]
            )

    ranks = []
    for lo, hi in partitions(12, 5, alignment=1):
        ranks.append(
            SimpleNamespace(
                device="cpu",
                gate_up=Bank(torch.cat((gate[:, lo:hi], up[:, lo:hi]), dim=1)),
                down=Bank(down[:, lo:hi]),
            )
        )
    expected = torch.cat(
        [
            F.linear(
                F.silu(F.linear(x[i : i + 1], gate[e])) * F.linear(x[i : i + 1], up[e]),
                down[e],
            )
            for i, e in enumerate(ids)
        ]
    )
    pool = make_pool(ranks)
    try:
        tp = TensorExperts(ranks, pool, decode_graphs=False)
        actual = tp.project(x, **({"ends": ends} if grouped else {"ids": ids}))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    finally:
        pool.shutdown()


def test_hot_switch_includes_nonowner_shards_and_invalidates_graphs(monkeypatch):
    r = runtime()
    extra = runtime().layers[0]
    r.resident_layers = [r.layers[0], extra]
    events = []
    r.invalidate_graphs = lambda: events.append("invalidate")
    resident = Resident(r)
    monkeypatch.setattr(resident, "synchronize", lambda: None)
    backend = object()
    resident.loaded["native"] = ("identity", backend)
    resident.switch(native="native")
    assert all(layer.gate_up.backend is backend for layer in r.resident_layers)
    assert events == ["invalidate"]
    before = resident.storage_identity()
    q, s, o = extra.down.parts[0]
    extra.down.parts[0] = (q.clone(), s, o)
    assert resident.storage_identity() != before


def test_memory_plan_conserves_weights_offsets_globals_and_full_context():
    from reap.muse_tensor import tensor_plan

    c = SimpleNamespace(
        num_hidden_layers=18,
        moe_hidden_size=768,
        moe_intermediate_size=768,
        sliding_window=128,
        num_key_value_heads=2,
        head_dim=64,
        layer_types=["sliding_attention", "full_attention"] * 9,
    )
    stages = [
        dict(device=f"npu:{r}", first_layer=r * 3, last_layer_exclusive=(r + 1) * 3)
        for r in range(6)
    ]
    shards = {}
    for i in range(18):
        shards[f"dense-{i:03d}.safetensors"] = {
            "tensors": {
                "attn.weight": {"bytes": 768 * 768},
                "attn.weight_scale": {"bytes": 768 * 4},
            }
        }
        for ep in (0, 1):
            shards[f"experts-{i:03d}-ep{ep}.safetensors"] = {
                "tensors": {
                    f"mlp.experts.{ep}.down_proj.weight": {"bytes": 768 * 384},
                    f"mlp.experts.{ep}.down_proj.weight_scale": {"bytes": 768 * 6 * 2},
                }
            }
    shards["global-0.safetensors"] = {
        "tensors": {
            "model.language_model.embed_tokens.weight": {"bytes": 1000000},
            "lm_head.weight": {"bytes": 1000000},
        }
    }
    plan = tensor_plan(c, {"shards": shards}, stages, 98304, 512, 3)
    expert_bytes = 18 * 2 * (768 * 384 + 768 * 6 * 3)
    dense_bytes = 18 * 768 * 768 * 2 + 2000000
    kv_bytes = 9 * (98304 + 128 - 1 + 512) * 2 * 2 * 64 * 2
    assert sum(p["expert_shard_bytes"] for p in plan) == expert_bytes
    assert sum(p["dense_and_kv_bytes"] for p in plan) == dense_bytes + kv_bytes
    assert (
        sum(p["required_bytes"] for p in plan)
        == expert_bytes + dense_bytes + kv_bytes + 6 * 3 * 2**30
    )
    assert plan[0]["first_layer"] == 0 and plan[-1]["last_layer_exclusive"] == 18
    assert all(
        a["last_layer_exclusive"] == b["first_layer"] for a, b in zip(plan, plan[1:])
    )


def test_decode_captures_share_device_pool_and_keep_independent_outputs(monkeypatch):
    import sys
    from reap import muse_tensor as module

    allocations, captures = [], []

    def pool_handle():
        value = object()
        allocations.append(value)
        return value

    def capture(graph, *, pool, stream):
        captures.append((pool, stream))
        return nullcontext()

    fake = SimpleNamespace(
        device=lambda _: nullcontext(),
        synchronize=lambda _: None,
        graph_pool_handle=pool_handle,
        Stream=lambda **_: object(),
        NPUGraph=lambda: SimpleNamespace(replay=lambda: None, reset=lambda: None),
        graph=capture,
    )
    monkeypatch.setattr(torch, "npu", fake, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(module, "_GRAPH_CONTEXTS", {})
    x, routes = torch.ones(8, 128), torch.arange(8)
    a = module.DecodeGraph.capture(lambda x, _: x * 2, x, routes)
    b = module.DecodeGraph.capture(lambda x, _: x * 3, x, routes)
    assert len(allocations) == 1 and captures[0] == captures[1]
    assert a.output.data_ptr() != b.output.data_ptr()
    assert (a.output == 2).all() and (b.output == 3).all()
    tp = TensorExperts([], None)
    tp.graphs = {"a": a, "b": b}
    tp.invalidate_graphs()
    module.DecodeGraph.capture(lambda x, _: x * 4, x, routes)
    assert len(allocations) == 2 and captures[2][0] is not captures[0][0]


def test_native_geometry_survives_lru_eviction_until_graph_reset(monkeypatch):
    import weakref
    from reap import muse_tensor as module

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(device=lambda _: nullcontext(), synchronize=lambda _: None),
        raising=False,
    )
    backend = SimpleNamespace(
        geometry={"native": torch.tensor([8, 1280, 2048, 128, 0])}
    )
    ref = weakref.ref(backend.geometry["native"])

    def capture(function, inputs, routes):
        return SimpleNamespace(
            inputs=inputs, graph=SimpleNamespace(reset=lambda: None), retained=()
        )

    monkeypatch.setattr(module.DecodeGraph, "capture", capture)
    rank = SimpleNamespace(
        device="cpu",
        gate_up=SimpleNamespace(backend=backend),
        down=SimpleNamespace(backend=backend),
    )
    tp = TensorExperts([rank], None)
    tp.prepare_graphs(torch.ones(8, 256), torch.arange(8))
    backend.geometry.clear()
    assert ref() is not None
    tp.invalidate_graphs()
    assert ref() is None


def test_decode_candidate_reapplication_does_not_nest_graph_wrappers(monkeypatch):
    import importlib
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "scripts/muse_resident_patches/tensor_decode_graphs.py"
    ).read_text()
    first = {}
    exec(source, first)
    for name, function in first["replacements"]().items():
        module, path = name.split(":")
        cls, method = path.split(".")
        monkeypatch.setattr(
            getattr(importlib.import_module(module), cls), method, function
        )
    second = {}
    exec(source, second)
    for name, function in second["replacements"]().items():
        assert function._muse_original is first["replacements"]()[name]._muse_original


def test_rank_pool_keeps_each_device_on_one_distinct_host_thread():
    import threading

    pool = make_pool(["cpu"] * 6)
    try:
        visits = []
        for _ in range(4):
            futures = [
                pool.submit(lambda rank: (rank, threading.get_ident()), r)
                for r in range(6)
            ]
            visits.append([f.result() for f in futures])
        assert len({identity for _, identity in visits[0]}) == 6
        assert all(row == visits[0] for row in visits)
    finally:
        pool.shutdown()
