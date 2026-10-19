"""Experimental Qwen native W4A8 adapter; serving defaults remain W4A16.

Weights retain their W4 values. Activations are quantized per row and G128 by
the Qwen operators, so this is not an exact-output replacement for W4A16.
The physical INT4 layout follows vllm-ascend's Apache-2.0 w4a8_int4.py.
"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
from threading import RLock

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


def check_direct_manifest(value):
    if (
        set(value) != {"name", "namespace", "abi", "library", "kernel", "quantizer"}
        or value["abi"] != "muse-w4a8-zero64-v1"
        or not re.fullmatch(r"muse_a8_[a-zA-Z0-9_]+", value["namespace"])
        or not re.fullmatch(r"[a-zA-Z0-9_]+", value["name"])
    ):
        raise ValueError("invalid direct A8 manifest")
    for name in ("library", "kernel", "quantizer"):
        entry = value[name]
        if set(entry) != {"path", "sha256"} or not Path(entry["path"]).is_absolute():
            raise ValueError("invalid direct A8 artifact")
        with Path(entry["path"]).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"direct A8 {name} hash mismatch")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class DirectA8Bank:
    """64-aligned native weights without resident zero-offset tensors."""

    def __init__(self, bank, device, backend):
        if not bank.parts or bank.inputs != sum(p[0].shape[-1] * 2 for p in bank.parts):
            raise ValueError("invalid direct A8 split geometry")
        experts = bank.parts[0][0].shape[0]
        if not 1 <= experts <= 128 or any(
            p[0].shape[:2] != (experts, bank.outputs) for p in bank.parts
        ):
            raise ValueError("invalid direct A8 expert geometry")
        self.inputs, self.outputs, self.backend = bank.inputs, bank.outputs, backend
        self.parts = []
        for part in bank.parts:
            q, scale, _, weight_sum = pack_part(*part, alignment=64)
            self.parts.append(tuple(v.to(device) for v in (q, scale, weight_sum)))

    def linear(self, x, *, ids=None, ends=None):
        return self.backend.linear(self, x, ids=ids, ends=ends)


class DirectA8:
    """Versioned direct launch; graph owners must retain warmup geometry tensors."""

    def __init__(self, manifest, devices):
        check_direct_manifest(manifest)
        self.manifest = manifest
        self.geometry, self.kernels, self.empty = OrderedDict(), {}, {}
        self.geometry_lock = RLock()
        torch.ops.load_library(manifest["library"]["path"])
        namespace = manifest["namespace"]
        self.launch = getattr(getattr(torch.ops, namespace), "launch")
        cls = getattr(getattr(torch.classes, namespace), "Kernel")
        for device in devices:
            device = torch.device(device)
            with torch.npu.device(device):
                self.kernels[device] = (
                    cls(manifest["kernel"]["path"], "muse_a8_native"),
                    cls(manifest["quantizer"]["path"], "qwen_w4_a8_pack_v310"),
                )
                self.empty[device] = torch.empty(0, dtype=torch.uint8, device=device)

    def metadata(self, key, values):
        # Six rank threads share this backend; tensor creation releases the GIL.
        with self.geometry_lock:
            if key not in self.geometry:
                self.geometry[key] = torch.tensor(
                    values, dtype=torch.int64, device=key[0]
                )
                if len(self.geometry) > 256:
                    self.geometry.popitem(last=False)
            self.geometry.move_to_end(key)
            return self.geometry[key]

    def linear(self, bank, x, *, ids=None, ends=None):
        if (ids is None) == (ends is None):
            raise ValueError("provide routed IDs or grouped ends")
        if not bank.parts or bank.inputs != sum(p[0].shape[-1] * 2 for p in bank.parts):
            raise ValueError("invalid direct A8 split geometry")
        routes = ids if ids is not None else ends
        experts = bank.parts[0][0].shape[0]
        if (
            x.ndim != 2
            or x.shape[1] != bank.inputs
            or x.dtype != torch.float16
            or x.device not in self.kernels
            or not 1 <= len(x) <= 4096
            or routes.ndim != 1
            or routes.device != x.device
            or routes.dtype != (torch.int32 if ids is not None else torch.int64)
            or len(routes) != (len(x) if ids is not None else experts)
            or (ids is not None and len(x) > 128)
        ):
            raise ValueError("invalid direct A8 input/route geometry")
        routes = routes.contiguous()
        kernel, quantizer = self.kernels[x.device]
        accumulator, first = None, 0
        for q, scale, weight_sum in bank.parts:
            e, n, packed_k = q.shape
            k = packed_k * 2
            if (
                e != experts
                or not 1 <= e <= 128
                or n != (bank.outputs + 63) // 64 * 64
                or not 64 <= n <= 5120
                or not 256 <= k <= 2048
                or k % 128
                or q.dtype != torch.int8
                or scale.dtype != torch.float16
                or weight_sum.dtype != torch.float16
                or scale.shape != (e, n, k // 128)
                or weight_sum.shape != scale.shape
                or any(
                    v.device != x.device or not v.is_contiguous()
                    for v in (q, scale, weight_sum)
                )
            ):
                raise ValueError("invalid direct A8 weight ABI")
            current = x[:, first : first + k].contiguous()
            low = torch.empty((len(x), packed_k), dtype=torch.int8, device=x.device)
            high = torch.empty_like(low)
            activation_scale = torch.empty(
                (len(x), k // 128, 8), dtype=torch.float32, device=x.device
            )
            activation_sum = torch.empty_like(activation_scale)
            pack_meta = self.metadata(
                (x.device, len(x), 0, k, 0, -1), [len(x) * k // 128]
            )
            self.launch(
                quantizer,
                [
                    current,
                    low,
                    high,
                    activation_scale,
                    activation_sum,
                    self.empty[x.device],
                    pack_meta,
                ],
                8,
            )
            result = torch.empty((len(x), n), dtype=x.dtype, device=x.device)
            meta = self.metadata(
                (x.device, len(x), bank.outputs, k, e, int(ends is not None)),
                [len(x), e, n, k, 8, int(ids is not None), 1],
            )
            self.launch(
                kernel,
                [
                    low,
                    high,
                    activation_scale,
                    activation_sum,
                    q,
                    scale,
                    weight_sum,
                    routes,
                    result,
                    meta,
                ],
                8,
            )
            value = result[:, : bank.outputs].float()
            accumulator = value if accumulator is None else accumulator + value
            first += k
        return accumulator.to(x.dtype)
