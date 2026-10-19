from types import SimpleNamespace

import pytest
import torch

from reap.muse_fast import pack_tiles
from reap.muse_int4 import NativeA8Bank, canonical_codes, pack_part


@pytest.mark.parametrize("n,k", [(32, 256), (672, 2048), (704, 512), (1408, 1024)])
@pytest.mark.parametrize("alignment", [32, 64, 128])
def test_native_layout_preserves_weights_and_padding(n, k, alignment):
    torch.manual_seed(n + k)
    canonical = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8)
    scales = torch.randn(n, k // 128).half()
    tiled = pack_tiles(canonical)
    assert torch.equal(canonical_codes(tiled), canonical)
    q, s, offsets, sums = pack_part(
        tiled[None],
        pack_tiles(scales, scales=True)[None],
        torch.full((1, n, k // 128), 8, dtype=torch.int8),
        alignment=alignment,
    )
    padded = q.shape[1]
    decoded = q[0].reshape(padded // 16, k // 128, 2, 16, 32)
    decoded = decoded.permute(0, 3, 1, 2, 4).contiguous().reshape(padded, k // 2)
    assert torch.equal(decoded[:n], canonical)
    assert not decoded[n:].any()
    restored_scale = s[0].reshape(padded // 16, k // 128, 16).transpose(1, 2)
    restored_scale = restored_scale.reshape(padded, k // 128)
    assert torch.equal(restored_scale[:n], scales)
    assert not restored_scale[n:].any()
    assert not offsets.any()
    restored_sum = sums[0].reshape(padded // 16, k // 128, 16).transpose(1, 2)
    restored_sum = restored_sum.reshape(padded, k // 128)
    fields = torch.stack((canonical & 15, (canonical >> 4) & 15), -1)
    signed = torch.where(fields >= 8, fields - 16, fields)
    assert torch.equal(restored_sum[:n], signed.reshape(n, -1, 128).sum(-1).half())
    assert not restored_sum[n:].any()


def test_native_layout_rejects_asymmetric_and_nonfinite_weights():
    q = torch.zeros(1, 32, 128, dtype=torch.int8)
    s = torch.ones(1, 32, 2, dtype=torch.float16)
    o = torch.full_like(s, 8, dtype=torch.int8)
    o[0, 0, 0] = 7
    with pytest.raises(ValueError, match="symmetric"):
        pack_part(q, s, o)
    o.fill_(8)
    s[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pack_part(q, s, o)


def test_native_adapter_rejects_invalid_split_geometry():
    part = (
        torch.zeros(1, 32, 128, dtype=torch.int8),
        torch.ones(1, 32, 2, dtype=torch.float16),
        torch.full((1, 32, 2), 8, dtype=torch.int8),
    )
    with pytest.raises(ValueError, match="width mismatch"):
        NativeA8Bank(SimpleNamespace(inputs=512, outputs=32, parts=[part]), "cpu")
    with pytest.raises(ValueError, match="output geometry"):
        NativeA8Bank(SimpleNamespace(inputs=256, outputs=64, parts=[part]), "cpu")


def test_native_adapter_preserves_split_sum_and_crops_padding(monkeypatch):
    part = (
        torch.zeros(1, 32, 128, dtype=torch.int8),
        torch.ones(1, 32, 2, dtype=torch.float16),
        torch.full((1, 32, 2), 8, dtype=torch.int8),
    )
    bank = NativeA8Bank(
        SimpleNamespace(inputs=768, outputs=32, parts=[part, part, part]), "cpu"
    )
    x = torch.arange(768, dtype=torch.float16).expand(8, -1)
    routes = torch.zeros(8, dtype=torch.int32)
    packed_inputs = []
    values = iter((2048, 1, -2048))

    def pack(value):
        packed_inputs.append(value.clone())
        return (value,) * 4

    def matmul(*arguments):
        assert arguments[-1] is routes
        result = torch.full((8, 128), next(values), dtype=torch.float16)
        result[:, 32:] = float("nan")
        return result

    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_qwen_w4_a8_pack_310", pack, raising=False
    )
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_qwen_w4_a8_int4_matmul_310", matmul, raising=False
    )
    actual = bank.linear(x, ids=routes)
    assert torch.equal(actual, torch.ones(8, 32, dtype=torch.float16))
    assert all(
        torch.equal(value, x[:, i * 256 : (i + 1) * 256])
        for i, value in enumerate(packed_inputs)
    )
    with pytest.raises(ValueError, match="route geometry"):
        bank.linear(x, ids=routes.long())
