"""Independent tiled ABI reconstruction and full routed MoE parity."""

from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from reap.muse_fast import FastMuseLayer, TiledBank, pack_tiles
from reap.muse_runtime import MuseLayer


def decode(codes, scales):
    n, packed_k = codes.shape
    k = packed_k * 2
    # Reconstruct by coordinates, independently of the packing reshape sequence.
    physical = codes.reshape(n // 32, k // 16, 16, 16)
    rows = torch.arange(n)[:, None]
    cols = torch.arange(k)[None, :]
    raw = physical[rows // 32, cols // 16, cols % 16, rows % 16].to(torch.int16)
    biased = (raw >> torch.where(rows % 32 < 16, 0, 4)) & 15
    metadata = scales.reshape(n // 32, k // 128, 32)
    scale = metadata[rows // 32, cols // 128, rows % 32]
    return ((biased - 8).float() * scale.float()).half()


@pytest.mark.parametrize("n,k", [(128, 256), (256, 4096)])
def test_tile_encoding_preserves_every_weight_and_scale(n, k):
    g = torch.Generator().manual_seed(717)
    integers = torch.randint(-8, 8, (n, k), dtype=torch.int8, generator=g)
    codes = (integers[:, ::2] & 15) | ((integers[:, 1::2] & 15) << 4)
    scales = (torch.rand(n, k // 128, generator=g) * 0.02).half()
    actual = decode(pack_tiles(codes), pack_tiles(scales, scales=True))
    expected = (integers.float() * scales.float().repeat_interleave(128, dim=-1)).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 7, 64, 257])
def test_fast_moe_matches_independent_dequantized_experts(monkeypatch, tokens):
    torch.manual_seed(81)
    c = SimpleNamespace(
        num_local_experts=4,
        num_experts_per_tok=2,
        output_multiplier=0.7,
        routed_scaling_factor=1.1,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-5,
    )
    values = {
        "input_layernorm.weight": torch.zeros(128).half(),
        "mlp.gate.weight": torch.randn(4, 128) * 0.01,
        "mlp.gate.e_score_correction_bias": torch.tensor([0.01, -0.02, 0.03, 0]),
        "mlp.pre_expert_proj.weight": (torch.randn(256, 128) * 0.02).half(),
        "mlp.post_expert_proj.weight": (torch.randn(128, 256) * 0.02).half(),
        "mlp.pre_expert_norm.weight": torch.zeros(256).half(),
        "mlp.experts.post_expert_norm.weight": torch.zeros(256).half(),
    }
    for e in range(4):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"mlp.experts.{e}.{proj}"
            values[key + ".weight"] = torch.randint(
                -128, 128, (256, 128), dtype=torch.int8
            )
            values[key + ".weight_scale"] = (torch.rand(256, 2) * 0.01).half()
    gate_up = TiledBank.from_weights(values, 4, ("gate_proj", "up_proj"), "cpu")
    down = TiledBank.from_weights(values, 4, ("down_proj",), "cpu")

    def group(x, q, s, o, tiled):
        assert tiled and (o == 8).all()
        return F.linear(x, decode(q, s))

    def routed(x, q, s, o, ids):
        assert ids.dtype == torch.int32 and (o == 8).all()
        return torch.cat(
            [
                F.linear(x[i : i + 1], decode(q[e], s[e]))
                for i, e in enumerate(ids.tolist())
            ]
        )

    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_qwen_w4_group_matmul_310", group, raising=False
    )
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_qwen_w4_routed_matmul_310", routed, raising=False
    )
    reference = MuseLayer(c, 0, values)
    fast = FastMuseLayer(c, 0, values, gate_up, down)
    x = (torch.randn(1, tokens, 128) * 0.1).half()
    with torch.inference_mode():
        torch.testing.assert_close(fast.moe(x), reference.moe(x), rtol=0, atol=0)


def test_tiled_split_k_matches_reference_partial_rounding(monkeypatch):
    torch.manual_seed(52)
    values = {}
    for e in range(2):
        key = f"mlp.experts.{e}.gate_proj"
        values[key + ".weight"] = torch.randint(
            -128, 128, (128, 2048), dtype=torch.int8
        )
        values[key + ".weight_scale"] = (torch.rand(128, 32) * 0.01).half()
    bank = TiledBank.from_weights(values, 2, ("gate_proj",), "cpu")
    calls = []

    def group(x, q, s, o, tiled):
        calls.append(x.shape)
        return F.linear(x, decode(q, s))

    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_qwen_w4_group_matmul_310", group, raising=False
    )
    x = (torch.randn(129, 4096) * 0.1).half()
    parts = []
    from reap.muse_export import dequantize_weight

    for a, b in ((0, 2048), (2048, 4096)):
        weight = dequantize_weight(
            values["mlp.experts.1.gate_proj.weight"][:, a // 2 : b // 2],
            values["mlp.experts.1.gate_proj.weight_scale"][:, a // 128 : b // 128],
        ).half()
        parts.append(F.linear(x[:, a:b], weight).float())
    expected = (parts[0] + parts[1]).half()
    torch.testing.assert_close(bank.linear(x, expert=1), expected, rtol=0, atol=0)
    assert len(calls) == 4 and max(shape[0] for shape in calls) == 128


def test_stage_positions_reuse_only_identical_geometry_and_chunk():
    from reap.muse_fast import StepPositions

    cache = StepPositions()
    c = SimpleNamespace(head_dim=64, rope_parameters={"rope_theta": 500000})
    x = torch.empty(1, 128, 8192).half()
    for start, count in ((0, 128), (128, 1), (98303, 1), (0, 7)):
        values = cache.get(c, x, start, count)
        assert cache.get(c, x, start, count) is values
        positions, cos, sin = values
        torch.testing.assert_close(positions, torch.arange(start, start + count))
        inv = 1.0 / (500000 ** (torch.arange(0, 64, 2).float() / 64))
        angles = positions[:, None].float() * inv
        phase = torch.cat((angles, angles), -1)
        torch.testing.assert_close(cos, phase.cos().half(), rtol=0, atol=0)
        torch.testing.assert_close(sin, phase.sin().half(), rtol=0, atol=0)
    c.rope_parameters["rope_theta"] = 10000
    changed = cache.get(c, x, 0, 7)
    assert not torch.equal(changed[1], values[1])
