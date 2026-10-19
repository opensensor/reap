from contextlib import nullcontext
from types import SimpleNamespace
from concurrent.futures import Future

import pytest
import torch
import torch.nn.functional as F

from reap.muse_dense import QG_NAMES, TensorAttention
from reap.muse_tensor import DecodeGraph, TensorExperts, make_pool


def test_changed_inputs_prefill_decode_and_recapture(monkeypatch):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(device=lambda _: nullcontext(), synchronize=lambda _: None),
        raising=False,
    )
    captures = []

    class Graph:
        def __init__(self, fn, x, routes):
            self.fn, self.inputs, self.routes = fn, x.clone(), routes.clone()
            self.output = fn(self.inputs, self.routes).clone()
            self.graph = SimpleNamespace(reset=lambda: None)
            captures.append(self)

        def run(self, x, routes):
            self.inputs.copy_(x)
            self.routes.copy_(routes)
            self.output.copy_(self.fn(self.inputs, self.routes))
            return self.output

    monkeypatch.setattr(DecodeGraph, "capture", Graph)
    torch.manual_seed(19)
    weights = {name: torch.randn(224, 64).half() for name in QG_NAMES}
    pool = make_pool(["cpu"] * 6)
    experts = TensorExperts([SimpleNamespace(device="cpu") for _ in range(6)], pool)
    tp = TensorAttention.from_layer(SimpleNamespace(w=weights), experts)
    try:
        with torch.inference_mode():
            for request in range(3):
                if request == 2:
                    experts.invalidate_graphs()
                for rows in (7, 1, 1, 32, 1):
                    x = torch.randn(1, rows, 64).half()
                    expected = torch.cat(
                        [F.linear(x, weights[n]) for n in QG_NAMES], dim=-1
                    )
                    torch.testing.assert_close(tp.project(x), expected, rtol=0, atol=0)
        assert len(captures) == 12
        assert sum(r.attention_qg_weight.numel() for r in experts.ranks) == 2 * 224 * 64
        assert len(pool.attention_staging) == 7
    finally:
        pool.shutdown()


def test_rank_failure_drains_remaining_futures():
    visited = []

    class Tracked(Future):
        def result(self):
            visited.append(self.rank)
            return super().result()

    class Pool:
        def submit(self, function, rank):
            f = Tracked()
            f.rank = rank
            if rank == 0:
                f.set_exception(RuntimeError("injected DMA failure"))
            else:
                f.set_result((torch.zeros(1, 32), torch.zeros(1, 32)))
            return f

    experts = TensorExperts(
        [SimpleNamespace(device="cpu")] * 6, Pool(), decode_graphs=False
    )
    with pytest.raises(RuntimeError, match="injected DMA"):
        TensorAttention(experts).project(torch.ones(1, 64))
    assert visited == list(range(6))


def test_loader_excludes_owner_copies_and_tracks_every_shard(monkeypatch):
    from reap import muse_tensor
    from reap.muse_fast import FastMuseLayer
    from reap.muse_resident import Resident

    # Device-tagged CPU substitutes exercise the actual loader without requiring
    # torch_npu. The graph/math test above separately checks the gathered values.
    weights = {n: torch.randn(224, 64).half() for n in QG_NAMES}
    weights["input_layernorm.weight"] = torch.zeros(64).half()
    cpu = SimpleNamespace(
        w=weights,
        c=SimpleNamespace(),
        index=0,
        gate_up=SimpleNamespace(inputs=768, outputs=1536),
        down=SimpleNamespace(inputs=768, outputs=768),
    )
    monkeypatch.setattr(muse_tensor, "slice_bank", lambda *a: SimpleNamespace(parts=[]))
    monkeypatch.setattr(muse_tensor, "native_weight", lambda x, d: x)
    from reap import muse_fast

    monkeypatch.setattr(muse_fast, "native_weight", lambda x, d: x)
    pool = make_pool(["cpu"] * 6)
    try:
        layer = muse_tensor.shard_layer(
            cpu,
            ["cpu", "cpu:1", "cpu:2", "cpu:3", "cpu:4", "cpu:5"],
            "cpu",
            pool,
            tensor_attention=True,
        )
        assert isinstance(layer, FastMuseLayer)
        assert not any(n in layer.w for n in QG_NAMES)
        resident = Resident(
            SimpleNamespace(
                optimized=True,
                globals={},
                layers=[layer],
                resident_layers=layer.tensor_experts.ranks,
            )
        )
        before = resident.storage_identity()
        rank = layer.tensor_experts.ranks[-1]
        rank.attention_qg_weight = rank.attention_qg_weight.clone()
        assert resident.storage_identity() != before
    finally:
        pool.shutdown()


@pytest.mark.parametrize("rope", [False, True])
def test_attention_prefill_and_cached_decode_without_owner_qg(rope):
    from reap.muse_fast import FastMuseLayer

    torch.manual_seed(829)
    c = SimpleNamespace(
        head_dim=32,
        rms_norm_eps=1e-5,
        qk_scale_factor=0.7,
        rope_parameters={"rope_theta": 10000},
        layer_rope_theta=[10000 if rope else None],
        sliding_window=16,
        layer_types=["full_attention"],
    )
    weights = {
        "input_layernorm.weight": torch.zeros(224).half(),
        **{
            f"self_attn.{n}_proj.weight": torch.randn(rows, 224).half() * 0.03
            for n, rows in (("q", 224), ("gate", 224), ("k", 32), ("v", 32), ("o", 224))
        },
    }
    base = FastMuseLayer(c, 0, weights, None, None)
    sharded = FastMuseLayer(
        c, 0, {n: w for n, w in weights.items() if n not in QG_NAMES}, None, None
    )
    pool = make_pool(["cpu"] * 6)
    experts = TensorExperts(
        [SimpleNamespace(device="cpu") for _ in range(6)], pool, decode_graphs=False
    )
    sharded.tensor_attention = TensorAttention.from_layer(base, experts)
    for layer in (base, sharded):
        layer.bounded_attention = True
        layer.max_context = 64
        layer.prefill_chunk = 16
    from reap.muse_attention import AppendCache

    caches = [AppendCache(), AppendCache()]
    try:
        with torch.inference_mode():
            start = 0
            for count in (7, 1, 1, 16, 1):
                x = torch.randn(1, count, 224).half()
                expected = base.attention(x, start, caches[0])
                actual = sharded.attention(x, start, caches[1])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                start += count
    finally:
        pool.shutdown()


def test_affinity_override_validates_all_masks_and_restores_threads(monkeypatch):
    import os
    import threading
    from pathlib import Path

    scope = {}
    exec(
        (
            Path(__file__).parents[1]
            / "scripts/muse_resident_patches/layer_decode_graphs.py"
        ).read_text(),
        scope,
    )
    masks = {i: set(range(8)) for i in range(7)}
    monkeypatch.setattr(os, "sched_getaffinity", lambda tid: masks[tid].copy())
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda tid, cpus: masks.__setitem__(tid, set(cpus))
    )
    monkeypatch.setattr(threading, "get_native_id", lambda: 6)
    pool = SimpleNamespace(
        workers=[
            SimpleNamespace(_threads=[SimpleNamespace(native_id=i)]) for i in range(6)
        ]
    )
    experts = SimpleNamespace(pool=pool)
    bind = scope["bind_rank_threads"]
    with pytest.raises(ValueError, match="outside original"):
        bind(experts, (0, 1, 2, 3, 4, 5, 8))
    assert all(v == set(range(8)) for v in masks.values())
    bind(experts, tuple(range(7)))
    assert all(v == {i} for i, v in masks.items())
    bind(experts, tuple(range(7)))
    bind(experts, ())
    assert all(v == set(range(8)) for v in masks.values())
    assert pool._muse_affinity_original == {}


def test_affinity_syscall_failure_restores_already_changed_masks(monkeypatch):
    import os
    import threading
    from pathlib import Path

    scope = {}
    exec(
        (
            Path(__file__).parents[1]
            / "scripts/muse_resident_patches/layer_decode_graphs.py"
        ).read_text(),
        scope,
    )
    masks = {i: set(range(8)) for i in range(7)}

    def apply(tid, cpus):
        if tid == 3 and len(cpus) == 1:
            raise OSError("injected affinity failure")
        masks[tid] = set(cpus)

    monkeypatch.setattr(os, "sched_getaffinity", lambda tid: masks[tid].copy())
    monkeypatch.setattr(os, "sched_setaffinity", apply)
    monkeypatch.setattr(threading, "get_native_id", lambda: 6)
    pool = SimpleNamespace(
        workers=[
            SimpleNamespace(_threads=[SimpleNamespace(native_id=i)]) for i in range(6)
        ]
    )
    with pytest.raises(OSError, match="injected affinity"):
        scope["bind_rank_threads"](SimpleNamespace(pool=pool), tuple(range(7)))
    assert all(v == set(range(8)) for v in masks.values())


def test_affinity_applies_only_to_decode_and_restores_prefill():
    from pathlib import Path

    scope = {}
    exec(
        (
            Path(__file__).parents[1]
            / "scripts/muse_resident_patches/layer_decode_graphs.py"
        ).read_text(),
        scope,
    )
    visits = []
    scope["CPU_AFFINITY"] = tuple(range(7))
    scope["bind_rank_threads"] = lambda experts, cpus: visits.append(cpus)
    scope["original_call"] = lambda self, x, start, cache: x
    layer = SimpleNamespace(index=0, tensor_experts=object(), bounded_attention=False)
    for count in (7, 1, 1, 512):
        x = torch.zeros(1, count, 16)
        assert scope["fused_layer"](layer, x) is x
    assert visits == [(), tuple(range(7)), tuple(range(7)), ()]
