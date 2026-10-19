"""Bounded W4 projection adapter for the existing Ascend 310P Qwen operator.

The installed operator accepts M <= 128 and K <= 2560. Muse has K=4096,
so split along K, accumulate partial products in FP32, and chunk token rows.
This is a correctness bring-up path, not a complete Muse serving runtime.
"""

import torch


def w4_linear(x, codes, scales, *, op=None, token_chunk=128, k_chunk=2048):
    """Evaluate symmetric group-128 W4 weights with a bounded-size operator.

    Only temporary zero offsets are allocated. Stored codes/scales remain packed.
    Each partial GEMM produces FP16, so split-K adds rounding versus a single GEMM.
    """
    if not 1 <= token_chunk <= 128 or not 256 <= k_chunk <= 2560 or k_chunk % 128:
        raise ValueError("operator chunk sizes outside supported bounds")
    if x.ndim != 2 or codes.ndim != 2 or scales.ndim != 2:
        raise ValueError("expected matrix inputs")
    m, k = x.shape
    n = codes.shape[0]
    if m < 1 or n < 1 or n % 128 or k < 256 or k % 128:
        raise ValueError("unsupported matrix geometry")
    if codes.shape != (n, k // 2) or scales.shape != (n, k // 128):
        raise ValueError("packed weight geometry differs")
    if (
        x.dtype != torch.float16
        or codes.dtype != torch.int8
        or scales.dtype != torch.float16
    ):
        raise ValueError("expected FP16 activations/scales and INT8 codes")
    if codes.device != x.device or scales.device != x.device:
        raise ValueError("all inputs must share a device")
    if op is None:
        op = torch.ops._C_ascend.npu_qwen_w4_group_matmul_310
    # Avoid an unsupported final K=128 by shortening the preceding chunk.
    ranges = []
    start = 0
    while start < k:
        width = min(k_chunk, k - start)
        if k - start - width == 128:
            width -= 128
        if width < 256:
            raise ValueError("K cannot be split into supported chunks")
        ranges.append((start, start + width))
        start += width
    result = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for row in range(0, m, token_chunk):
        accumulator = torch.zeros(
            (min(token_chunk, m - row), n), dtype=torch.float32, device=x.device
        )
        for start, end in ranges:
            q = codes[:, start // 2 : end // 2].contiguous()
            scale = scales[:, start // 128 : end // 128].contiguous()
            partial = op(
                x[row : row + token_chunk, start:end].contiguous(),
                q,
                scale,
                torch.zeros_like(scale, dtype=torch.int8),
                False,
            )
            accumulator.add_(partial.float())
        result[row : row + token_chunk] = accumulator.to(x.dtype)
    return result
