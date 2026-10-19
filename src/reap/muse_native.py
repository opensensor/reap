"""Versioned direct-launch W4 backend; shares the resident TiledBank allocations."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re

import torch

from reap.muse_packed import digest


def check_manifest(value):
    if set(value) != {"name", "namespace", "library", "kernel", "tile", "abi"}:
        raise ValueError("invalid native manifest fields")
    if (
        not re.fullmatch(r"muse_w4_[a-zA-Z0-9_]+", value["namespace"])
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", value["name"])
        or value["tile"] not in (512, 1024)
        or value["abi"] not in ("muse-w4-v1", "muse-w4-offset8-v1")
    ):
        raise ValueError("unsupported native manifest")
    for name in ("library", "kernel"):
        item = value[name]
        path = Path(item["path"])
        if not path.is_absolute() or digest(path) != item["sha256"]:
            raise ValueError(f"native {name} hash mismatch")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class NativeW4:
    def __init__(self, manifest, devices):
        self.manifest = manifest
        self.kernels, self.scratch, self.geometry = {}, {}, OrderedDict()
        self.validated_offsets = {}
        namespace = manifest["namespace"]
        torch.ops.load_library(manifest["library"]["path"])
        self.launch = getattr(getattr(torch.ops, namespace), "launch")
        cls = getattr(getattr(torch.classes, namespace), "Kernel")
        for device in devices:
            device = torch.device(device)
            with torch.npu.device(device):
                self.kernels[device] = cls(manifest["kernel"]["path"], "muse_w4_native")
                self.scratch[device] = torch.empty(
                    8 * 32 * 2048 * 2, dtype=torch.uint8, device=device
                )

    def linear(self, bank, x, *, ids=None, ends=None):
        if (ids is None) == (ends is None):
            raise ValueError("provide routed IDs or grouped ends")
        if (
            x.ndim != 2
            or x.shape[1] != bank.inputs
            or x.dtype != torch.float16
            or not 1 <= len(x) <= 4096
            or x.device not in self.kernels
        ):
            raise ValueError("invalid native input")
        routes = ids if ids is not None else ends
        experts = bank.parts[0][0].shape[0]
        if (
            routes.ndim != 1
            or routes.device != x.device
            or routes.dtype != (torch.int32 if ids is not None else torch.int64)
            or len(routes) != (len(x) if ids is not None else experts)
        ):
            raise ValueError("invalid native route metadata")
        routes = routes.contiguous()
        accumulator = None
        for part, (q, s, o) in enumerate(bank.parts):
            e, n, packed = q.shape
            k = packed * 2
            if (
                e != experts
                or n != bank.outputs
                or n % 32
                or not 256 <= k <= 2048
                or k % 128
                or q.dtype != torch.int8
                or s.dtype != torch.float16
                or o.dtype != torch.int8
                or s.shape != (e, n, k // 128)
                or o.shape != s.shape
                or any(t.device != x.device or not t.is_contiguous() for t in (q, s, o))
            ):
                raise ValueError("invalid native bank ABI")
            if self.manifest["abi"] == "muse-w4-offset8-v1":
                self.require_offset8(o)
            current = x[:, part * 2048 : part * 2048 + k].contiguous()
            key = (x.device, len(x), n, k, e, int(ends is not None))
            if key not in self.geometry:
                self.geometry[key] = torch.tensor(
                    key[1:], dtype=torch.int64, device=x.device
                )
                if len(self.geometry) > 256:
                    self.geometry.popitem(last=False)
            meta = self.geometry[key]
            self.geometry.move_to_end(key)
            result = torch.empty((len(x), n), dtype=x.dtype, device=x.device)
            self.launch(
                self.kernels[x.device],
                [current, q, s, o, routes, result, self.scratch[x.device], meta],
                8,
            )
            result = result.float()
            accumulator = result if accumulator is None else accumulator + result
        return accumulator.to(x.dtype)

    def require_offset8(self, offsets):
        # Weights are immutable during serving. Validate before warmup/capture,
        # retain the tensor so recycled addresses cannot inherit validation,
        # and revalidate versioned tensors after an in-place modification.
        # Inference tensors have no version counter and share the runtime's
        # existing immutable-weight contract.
        try:
            version = offsets._version
        except RuntimeError:
            version = None
        previous = self.validated_offsets.get(id(offsets))
        if previous is not None and previous[0] is offsets and previous[1] == version:
            return
        if not bool(torch.all(offsets == 8).item()):
            raise ValueError(
                "offset8 native ABI requires every weight offset to be eight"
            )
        self.validated_offsets[id(offsets)] = offsets, version
