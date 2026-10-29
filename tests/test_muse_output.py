from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from reap.muse_dense import TensorOutput
from reap.muse_resident import Resident
from reap.muse_runtime import MuseRuntime, PREFIX
from reap.muse_tensor import DecodeGraph, make_pool, tensor_plan
from tests.test_muse_resident import runtime


def test_output_rows_softcap_retention_and_recapture(monkeypatch):
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
            self.output.copy_(self.fn(self.inputs, self.routes))
            return self.output

    monkeypatch.setattr(DecodeGraph, "capture", Graph)
    torch.manual_seed(719)
    weight = torch.randn(224, 64).half()
    pool = make_pool(["cpu"] * 6)
    head = TensorOutput(weight, ["cpu"] * 6, pool, multiplier=0.7, softcap=2.0)
    try:
        with torch.inference_mode():
            retained = []
            for rows in (4, 1, 1, 7, 1):
                x = torch.randn(1, rows, 64).half()
                expected = torch.tanh((F.linear(x, weight) * 0.7) / 2.0) * 2.0
                actual = head.logits(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for previous, saved in retained:
                    torch.testing.assert_close(previous, saved, rtol=0, atol=0)
                retained.append((actual, actual.clone()))
            assert len(head.buffers) == 7
            r = runtime()
            r.output_head = head
            resident = Resident(r)
            before = resident.identity
            head.ranks[-1].weight = head.ranks[-1].weight.clone()
            assert resident.storage_identity() != before
            MuseRuntime.invalidate_graphs(r)
            assert not head.execution.graphs
            torch.testing.assert_close(head.logits(x), expected, rtol=0, atol=0)
            assert len(captures) == 12
            assert sum(v.weight.numel() for v in head.ranks) == weight.numel()
    finally:
        pool.shutdown()


def test_output_failure_drains_every_rank():
    visited = []

    class Tracked(Future):
        def result(self):
            visited.append(self.rank)
            return super().result()

    class Pool:
        def submit(self, fn, rank):
            f = Tracked()
            f.rank = rank
            if rank == 0:
                f.set_exception(RuntimeError("head rank failed"))
            else:
                f.set_result(torch.zeros(1, 4, 32))
            return f

    head = TensorOutput(
        torch.ones(224, 64).half(), ["cpu"] * 6, Pool(), multiplier=1, softcap=2
    )
    with pytest.raises(RuntimeError, match="head rank failed"):
        head.logits(torch.ones(1, 4, 64).half())
    assert visited == list(range(6))


def test_output_placement_moves_bytes_without_duplicates():
    c = SimpleNamespace(
        num_hidden_layers=6,
        layer_types=["full_attention"] * 6,
        num_key_value_heads=1,
        head_dim=16,
        moe_hidden_size=192,
        moe_intermediate_size=192,
    )
    shards = {}
    for i in range(6):
        shards[f"dense-{i:03d}.safetensors"] = {"tensors": {}}
        for ep in (0, 1):
            shards[f"experts-{i:03d}-ep{ep}.safetensors"] = {"tensors": {}}
    shards["global-head.safetensors"] = {
        "tensors": {"lm_head.weight": {"bytes": 224 * 64 * 2, "shape": [224, 64]}}
    }
    shards["global-embedding.safetensors"] = {
        "tensors": {
            PREFIX + "embed_tokens.weight": {"bytes": 224 * 64 * 2, "shape": [224, 64]}
        }
    }
    args = (
        c,
        {"shards": shards},
        [{"device": f"npu:{i}"} for i in range(6)],
        32,
        16,
        3,
    )
    before = tensor_plan(*args)
    after = tensor_plan(*args, tensor_output=True)
    assert sum(x["required_bytes"] for x in before) == sum(
        x["required_bytes"] for x in after
    )
    assert [
        b["required_bytes"] - a["required_bytes"] for a, b in zip(before, after)
    ] == [64 * 64 * 2] + [32 * 64 * 2] * 4 + [32 * 64 * 2 - 224 * 64 * 2]
