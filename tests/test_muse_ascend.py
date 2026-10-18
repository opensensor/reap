import pytest
import torch
import torch.nn.functional as F

from reap.muse_ascend import w4_linear


@pytest.mark.parametrize("m,k", [(1, 4096), (129, 4096), (17, 2176)])
def test_split_projection_matches_unsplit_reference(m, k):
    generator = torch.Generator().manual_seed(71)
    x = (torch.randn(m, k, generator=generator) * 0.1).half()
    integers = torch.randint(-7, 8, (128, k), generator=generator, dtype=torch.int8)
    codes = ((integers[:, ::2] & 15) | ((integers[:, 1::2] & 15) << 4)).contiguous()
    scales = torch.full((128, k // 128), 0.02, dtype=torch.float16)
    calls = []

    def operator(inputs, packed, scale, offset, tiled):
        calls.append(inputs.shape)
        assert (
            inputs.is_contiguous() and packed.is_contiguous() and scale.is_contiguous()
        )
        assert inputs.shape[0] <= 128 and 256 <= inputs.shape[1] <= 2560
        assert not tiled and not offset.any()
        raw = packed.to(torch.uint8)
        nibbles = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(128, -1).short()
        signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
        weight = (signed.float() * scale.float().repeat_interleave(128, dim=1)).half()
        return F.linear(inputs, weight)

    actual = w4_linear(x, codes, scales, op=operator)
    expected = F.linear(
        x, (integers.float() * scales.float().repeat_interleave(128, dim=1)).half()
    )
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.003)
    assert len(calls) == ((m + 127) // 128) * 2


def test_invalid_contract_fails_before_operator_call():
    x = torch.zeros(1, 4096, dtype=torch.float16)
    codes = torch.zeros(128, 2048, dtype=torch.int8)
    scales = torch.ones(128, 32, dtype=torch.float16)

    def fail(*args):
        raise AssertionError("operator must not run")

    for options in ({"token_chunk": 129}, {"k_chunk": 4096}):
        with pytest.raises(ValueError, match="chunk"):
            w4_linear(x, codes, scales, op=fail, **options)
    with pytest.raises(ValueError, match="geometry"):
        w4_linear(x, codes[:, :-1], scales, op=fail)
    with pytest.raises(ValueError, match="FP16"):
        w4_linear(x.float(), codes, scales, op=fail)
