"""Experimental Qwen native W4A8 adapter; never selected by the serving runtime.

Weights retain their W4 values. Activations are quantized per row and G128 by
the Qwen operators, so this is not an exact-output replacement for W4A16.
The physical INT4 layout follows vllm-ascend's Apache-2.0 w4a8_int4.py.
"""

import torch
import torch.nn.functional as F


def canonical_codes(packed):
    """Reverse Muse's biased NZ packing without changing signed W4 values."""
    if (
        packed.device.type != "cpu"
        or packed.dtype != torch.int8
        or packed.ndim != 2
        or packed.shape[0] % 32
        or packed.shape[1] % 64
    ):
        raise ValueError("expected CPU INT8 NZ codes with N%32=K%128=0")
    n, k = packed.shape[0], packed.shape[1] * 2
    tiles = packed.reshape(n // 32, k // 16, 16, 16)
    fields = torch.stack((tiles & 15, (tiles >> 4) & 15), dim=3)
    nibbles = fields.permute(0, 3, 4, 1, 2).reshape(n, k) ^ 8
    return (nibbles[:, ::2] | (nibbles[:, 1::2] << 4)).to(torch.int8)


def native_metadata(value):
    n = value.shape[0]
    return value.reshape(n // 16, 16, -1).transpose(1, 2).contiguous().view_as(value)


def pack_part(codes, scales, offsets, *, alignment=128):
    """Encode a CPU bank; alignment below 128 requires a specialized kernel."""
    if alignment not in (32, 64, 128):
        raise ValueError("native INT4 alignment must be 32, 64 or 128")
    if (
        codes.ndim != 3
        or codes.dtype != torch.int8
        or codes.device.type != "cpu"
        or codes.shape[0] < 1
        or codes.shape[1] < 32
        or codes.shape[1] % 32
        or not 128 <= codes.shape[2] <= 1024
        or codes.shape[2] % 64
        or scales.dtype != torch.float16
        or offsets.dtype != torch.int8
        or scales.device.type != "cpu"
        or offsets.device.type != "cpu"
        or scales.shape != (*codes.shape[:2], codes.shape[2] // 64)
        or offsets.shape != scales.shape
    ):
        raise ValueError("invalid CPU Muse W4 bank")
    if not bool((offsets == 8).all()) or not bool(torch.isfinite(scales).all()):
        raise ValueError("native INT4 conversion requires symmetric finite weights")
    experts, n, packed_k = codes.shape
    padded = (n + alignment - 1) // alignment * alignment
    if padded > 5120:
        raise ValueError("native INT4 binding requires output shards <=5120")
    q, s, sums = [], [], []
    for index in range(experts):
        canonical = F.pad(canonical_codes(codes[index]), (0, 0, 0, padded - n))
        scale = scales[index].reshape(n // 32, packed_k // 64, 32)
        scale = scale.transpose(1, 2).contiguous().reshape_as(scales[index])
        scale = F.pad(scale, (0, 0, 0, padded - n))
        fields = torch.stack((canonical & 15, (canonical >> 4) & 15), -1)
        signed = torch.where(fields >= 8, fields - 16, fields)
        weight_sum = signed.reshape(padded, -1, 128).sum(-1).half()
        tiles = canonical.reshape(padded // 16, 16, packed_k // 64, 2, 32)
        q.append(tiles.permute(0, 2, 3, 1, 4).contiguous().view_as(canonical))
        s.append(native_metadata(scale))
        sums.append(native_metadata(weight_sum))
    q, s, sums = torch.stack(q), torch.stack(s), torch.stack(sums)
    return q, s, torch.zeros_like(s), sums


class NativeA8Bank:
    """Explicit experimental adapter; repacks before device transfer/capture."""

    def __init__(self, bank, device):
        if bank.inputs != sum(part[0].shape[-1] * 2 for part in bank.parts):
            raise ValueError("split-K bank width mismatch")
        if not bank.parts or any(
            part[0].shape[:2] != (bank.parts[0][0].shape[0], bank.outputs)
            for part in bank.parts
        ):
            raise ValueError("split-K bank output geometry mismatch")
        self.inputs, self.outputs = bank.inputs, bank.outputs
        self.parts = [
            tuple(value.to(device) for value in pack_part(*part)) for part in bank.parts
        ]

    def linear(self, x, *, ids=None, ends=None):
        if (ids is None) == (ends is None):
            raise ValueError("provide routed IDs or grouped ends")
        routes = ids if ids is not None else ends
        experts = self.parts[0][0].shape[0]
        if (
            x.ndim != 2
            or x.shape[1] != self.inputs
            or x.dtype != torch.float16
            or not 1 <= len(x) <= 4096
            or x.device != self.parts[0][0].device
            or routes.device != x.device
            or routes.ndim != 1
            or routes.dtype != (torch.int32 if ids is not None else torch.int64)
            or len(routes) != (len(x) if ids is not None else experts)
            or (ids is not None and len(x) > 128)
        ):
            raise ValueError("invalid native INT4 input/route geometry")
        routes = routes.contiguous()
        accumulator, first = None, 0
        for weights in self.parts:
            width = weights[0].shape[-1] * 2
            current = x[:, first : first + width].contiguous()
            prepared = torch.ops._C_ascend.npu_qwen_w4_a8_pack_310(current)
            value = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(
                *prepared, *weights, routes
            )[:, : self.outputs].float()
            accumulator = value if accumulator is None else accumulator + value
            first += width
        return accumulator.to(x.dtype)
