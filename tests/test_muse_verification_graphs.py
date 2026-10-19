"""CPU graph structure/math tests; NPU qualification is a separate hardware gate."""

from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from reap.muse_fast import FastMuseLayer
from reap.muse_tensor import TensorExperts, make_pool, partitions


PATCH = (
    Path(__file__).parents[1]
    / "scripts/muse_resident_patches/layer_verification_graphs_v1.py"
)
MEASURED_SHA256 = "b8fe5b57a99ff40f1aeef17285a3a60a142fb190bed359283c216d0307662e7f"


class Layer:
    def __init__(self):
        self.index = 0
        self.c = SimpleNamespace(
            num_experts_per_tok=8,
            num_local_experts=16,
            output_multiplier=0.75,
            routed_scaling_factor=0.5,
        )
        self.w = {
            "mlp.gate.weight": torch.randn(16, 16),
            "mlp.gate.e_score_correction_bias": torch.randn(16) * 0.1,
        }
        self.gate_up = SimpleNamespace(backend=True)
        self.pre = torch.randn(16, 16).half() * 0.2
        self.post = torch.randn(16, 16).half() * 0.2
        self.up = torch.randn(16, 32, 16).half() * 0.2
        self.down = torch.randn(16, 16, 16).half() * 0.2

    def norm(self, name, x, input_scaled=False):
        scale = 1.25 if input_scaled else 0.875
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
            * scale
        ).half()

    def linear(self, name, x):
        return F.linear(x, self.pre if name == "mlp.pre_expert_proj" else self.post)

    def expert_projection(self, routes, *, ids=None, ends=None):
        if ids is None:
            ids = torch.repeat_interleave(
                torch.arange(16), ends.diff(prepend=torch.zeros(1, dtype=ends.dtype))
            )
        output = []
        for row, expert in zip(routes, ids):
            gate, up = F.linear(row[None], self.up[expert]).chunk(2, dim=-1)
            output.append(F.linear(F.silu(gate) * up, self.down[expert]))
        return torch.cat(output)


class Bank:
    def __init__(self, weight):
        self.weight = weight
        self.inputs = weight.shape[-1]
        self.backend = SimpleNamespace(geometry={"marker": torch.zeros(1)})

    def linear(self, x, *, ids=None, ends=None):
        assert ids is not None and ends is None
        return torch.cat(
            [F.linear(row[None], self.weight[expert]) for row, expert in zip(x, ids)]
        )


class FakeGraph:
    captures = 0

    @classmethod
    def capture(cls, function, inputs, routes):
        cls.captures += 1
        item = cls()
        item.function = function
        item.inputs, item.routes = inputs.clone(), routes.clone()
        item.output = function(item.inputs, item.routes)
        item.retained = ()
        item.fail = False
        return item

    def run(self, inputs, routes):
        if self.fail:
            raise RuntimeError("injected rank replay failure")
        assert inputs.shape == self.inputs.shape and routes.shape == self.routes.shape
        self.inputs.copy_(inputs)
        self.routes.copy_(routes)
        result = self.function(self.inputs, self.routes)
        # Retain static output addresses, as real graph replay does.
        if isinstance(result, tuple):
            for old, new in zip(self.output, result):
                old.copy_(new)
        else:
            self.output.copy_(result)
        return self.output


@pytest.fixture
def probe(monkeypatch):
    scope = {}
    exec(compile(PATCH.read_text(), str(PATCH), "exec"), scope)
    references = {2: 0, 4: 0, 8: 0}

    def grouped_reference(layer, x):
        if x.shape[1] in references:
            references[x.shape[1]] += 1
        return FastMuseLayer.moe(layer, x)

    scope["_baseline_graph_moe"] = grouped_reference
    scope["DecodeGraph"] = FakeGraph
    FakeGraph.captures = 0
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(device=lambda _: nullcontext(), synchronize=lambda _: None),
        raising=False,
    )
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    torch.manual_seed(1919)
    layer = Layer()
    ranks = [
        SimpleNamespace(
            device="cpu",
            gate_up=Bank(
                torch.cat((layer.up[:, lo:hi], layer.up[:, 16 + lo : 16 + hi]), dim=1)
            ),
            down=Bank(layer.down[:, lo:hi]),
        )
        for lo, hi in partitions(16, 5, alignment=1)
    ]
    pool = make_pool(ranks)
    experts = TensorExperts(ranks, pool)
    layer.tensor_experts = experts
    try:
        yield SimpleNamespace(
            layer=layer,
            experts=experts,
            pool=pool,
            scope=scope,
            references=references,
            candidate=scope["replacements"]()["reap.muse_fast:FastMuseLayer.moe"],
        )
    finally:
        pool.shutdown()


def test_versioned_patch_matches_hardware_qualified_source():
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == MEASURED_SHA256


def test_alternating_block_graphs_preserve_grouped_math_and_metadata(probe):
    for repeat in range(8):
        for count in (1, 8, 2, 4, 8, 3, 16, 2):
            x = (torch.randn(1, count, 16) * (repeat + 1) / 5).half()
            expected = FastMuseLayer.moe(probe.layer, x)
            actual = probe.candidate(probe.layer, x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert FakeGraph.captures == 36 == len(probe.experts.graphs)
    qualification = probe.scope["routed_block_qualification_status"](expected_layers=1)
    assert qualification["completed"] and len(qualification["records"]) == 3
    assert all(record["qualified_calls"] == 2 for record in qualification["records"])
    assert probe.references == {2: 2, 4: 2, 8: 2}
    for key, item in probe.experts.graphs.items():
        if key[0] == "verification":
            bank = getattr(
                probe.experts.ranks[key[2]], "gate_up" if key[3] == "up" else "down"
            )
            assert len(item.retained) == 1
            assert item.retained[0] is bank.backend.geometry["marker"]


def test_guard_qualifies_only_observed_shapes_and_fails_sticky(probe):
    x = torch.randn(1, 4, 16).half()
    probe.candidate(probe.layer, x)
    status = probe.scope["routed_block_qualification_status"](expected_layers=1)
    assert not status["completed"] and status["records"][0]["qualified_calls"] == 1
    unguarded = probe.scope["_unguarded_routed_block_graph_moe"]
    probe.scope["_unguarded_routed_block_graph_moe"] = lambda layer, x: (
        unguarded(layer, x) + 0.125
    )
    with pytest.raises(AssertionError):
        probe.candidate(probe.layer, x)
    with pytest.raises(RuntimeError, match="previously failed"):
        probe.candidate(probe.layer, x)
    status = probe.scope["routed_block_qualification_status"](expected_layers=1)
    assert not status["completed"] and not status["records"][0]["exact"]
    assert probe.references[4] == 2


def test_rank_replay_failure_drains_workers_and_can_recover(probe):
    x = torch.randn(1, 4, 16).half()
    for _ in range(2):
        probe.candidate(probe.layer, x)
    failed = probe.experts.graphs["verification", 4, 2, "up"]
    failed.fail = True
    with pytest.raises(RuntimeError, match="injected rank replay failure"):
        probe.candidate(probe.layer, x)
    # FIFO queues show every worker has escaped its barrier and drained.
    for rank in range(len(probe.experts.ranks)):
        assert probe.pool.submit(lambda rank: rank, rank).result(timeout=2) == rank
    failed.fail = False
    torch.testing.assert_close(
        probe.candidate(probe.layer, x),
        FastMuseLayer.moe(probe.layer, x),
        rtol=0,
        atol=0,
    )
    assert FakeGraph.captures == 12


@pytest.mark.parametrize("count", [1, 3, 16])
def test_other_row_counts_keep_existing_path_without_verification_graphs(probe, count):
    x = torch.randn(1, count, 16).half()
    torch.testing.assert_close(
        probe.candidate(probe.layer, x),
        FastMuseLayer.moe(probe.layer, x),
        rtol=0,
        atol=0,
    )
    assert not probe.experts.graphs
    assert not probe.scope["routed_block_qualification_status"](expected_layers=1)[
        "records"
    ]
