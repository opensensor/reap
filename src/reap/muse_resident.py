"""Trusted local experiments with resident weights and exact-logit switch gates.

Native registrations cannot be unloaded. A failure after native loading blocks
inference until restart. Python candidates must only define replacement functions;
they must never mutate tensors, construct weights, or change device state.
"""

import hashlib
import importlib
import json
import os
import time

import torch

from reap.muse_native import NativeW4, check_manifest

TARGETS = {
    "reap.muse_fast:FastMuseLayer.__call__",
    "reap.muse_fast:FastMuseLayer.moe",
    "reap.muse_fast:FastMuseLayer.fast_rms",
    "reap.muse_fast:FastMuseLayer.attention_inputs",
    "reap.muse_fast:FastMuseLayer.attention_output",
    "reap.muse_fast:TiledBank.linear",
}


def target(name):
    if name not in TARGETS:
        raise ValueError(f"unsupported replacement: {name}")
    module, path = name.split(":")
    owner = importlib.import_module(module)
    *parents, attribute = path.split(".")
    for parent in parents:
        owner = getattr(owner, parent)
    return owner, attribute


class Resident:
    def __init__(self, runtime):
        if not runtime.optimized:
            raise ValueError("resident controls require optimized runtime")
        self.runtime = runtime
        self.layers = getattr(runtime, "resident_layers", runtime.layers)
        self.loaded, self.namespaces = {}, {}
        self.active, self.python_digest, self.failed = None, None, None
        self.generation = 0
        self.originals = {name: getattr(*target(name)) for name in TARGETS}
        self.devices = sorted({str(layer.device) for layer in self.layers})
        self.identity = self.storage_identity()
        self.a8_manifest = getattr(runtime, "experimental_a8_manifest", None)
        if self.a8_manifest is not None:
            from reap.muse_int4 import check_direct_manifest

            key = check_direct_manifest(self.a8_manifest)
            backend = self.layers[0].gate_up.backend
            if backend is None or any(
                bank.backend is not backend
                for layer in self.layers
                for bank in (layer.gate_up, layer.down)
            ):
                raise ValueError("A8 resident banks must share their loaded backend")
            name = self.a8_manifest["name"]
            self.loaded[name] = key, backend
            self.namespaces[self.a8_manifest["namespace"]] = self.a8_manifest["library"]
            self.active = name

    def storage_identity(self):
        tensors = list(self.runtime.globals.items())
        if hasattr(self.runtime, "output_head"):
            tensors.extend(
                (f"output_head.{i}", rank.weight)
                for i, rank in enumerate(self.runtime.output_head.ranks)
            )
        for i, layer in enumerate(self.layers):
            tensors.extend((f"{i}.{name}", value) for name, value in layer.w.items())
            if hasattr(layer, "attention_qg_weight"):
                tensors.append((f"{i}.attention_qg", layer.attention_qg_weight))
            for name in ("gate_up", "down"):
                for p, part in enumerate(getattr(layer, name).parts):
                    tensors.extend(
                        (f"{i}.{name}.{p}.{j}", t) for j, t in enumerate(part)
                    )
        values = [
            (
                name,
                str(t.device),
                str(t.dtype),
                list(t.shape),
                t.untyped_storage().data_ptr(),
            )
            for name, t in tensors
        ]
        return hashlib.sha256(json.dumps(values).encode()).hexdigest()

    def synchronize(self):
        for device in self.devices:
            torch.npu.synchronize(device)

    def status(self):
        return {
            "pid": os.getpid(),
            "generation": self.generation,
            "active_native": self.active,
            "python_sha256": self.python_digest,
            "loaded_native": list(self.loaded),
            "failed": self.failed,
            "weight_storage_sha256": self.identity,
            "expert_format": "W4A8" if self.a8_manifest is not None else "W4A16",
        }

    def require_ready(self):
        if self.failed:
            raise RuntimeError(
                f"resident transaction failed; restart required: {self.failed}"
            )

    def load_native(self, manifest):
        self.require_ready()
        checker, factory = check_manifest, NativeW4
        if self.a8_manifest is not None:
            from reap.muse_int4 import DirectA8, check_direct_manifest

            checker, factory = check_direct_manifest, DirectA8
        key = checker(manifest)  # Reject cross-format swaps before native mutation.
        name, namespace = manifest["name"], manifest["namespace"]
        if name in self.loaded:
            if self.loaded[name][0] != key:
                raise ValueError("native name already used with different content")
            return self.status()
        library = manifest["library"]
        if namespace in self.namespaces and self.namespaces[namespace] != library:
            raise ValueError("namespace already belongs to another library")
        if namespace not in self.namespaces and hasattr(
            getattr(torch.ops, namespace), "launch"
        ):
            raise ValueError("native namespace already registered outside this session")
        started = time.monotonic()
        try:
            backend = factory(manifest, self.devices)
            # Real resident gate/up banks on every stage: decode and grouped rows
            # crossing the old 128-row limit, with empty expert groups.
            with torch.inference_mode():
                for device in self.devices:
                    layer = next(l for l in self.layers if str(l.device) == device)
                    banks = (
                        (layer.gate_up, layer.down)
                        if self.a8_manifest is not None
                        else (layer.gate_up,)
                    )
                    for bank in banks:
                        for grouped in (False, True):
                            rows = 137 if grouped else 8
                            x = (
                                (
                                    torch.arange(
                                        rows * bank.inputs, dtype=torch.float32
                                    )
                                    .reshape(rows, bank.inputs)
                                    .remainder(97)
                                    / 970
                                )
                                .half()
                                .to(device)
                            )
                            if grouped:
                                ends = torch.full(
                                    (bank.parts[0][0].shape[0],),
                                    rows,
                                    dtype=torch.int64,
                                    device=device,
                                )
                                expected = (
                                    bank.linear(x, ends=ends)
                                    if self.a8_manifest is not None
                                    else bank.linear(x, expert=0)
                                )
                                actual = backend.linear(bank, x, ends=ends)
                            else:
                                ids = torch.arange(
                                    rows, dtype=torch.int32, device=device
                                )
                                old = bank.backend
                                if self.a8_manifest is None:
                                    bank.backend = None
                                try:
                                    expected = bank.linear(x, ids=ids)
                                finally:
                                    bank.backend = old
                                actual = backend.linear(bank, x, ids=ids)
                            torch.testing.assert_close(
                                actual.cpu(), expected.cpu(), rtol=0, atol=0
                            )
            self.synchronize()
            if self.storage_identity() != self.identity:
                raise RuntimeError("resident weight storage changed")
            self.loaded[name] = key, backend
            self.namespaces[namespace] = library
        except Exception as exc:
            self.failed = str(exc)
            raise
        return {
            **self.status(),
            "load_seconds": time.monotonic() - started,
            "probe_passed": True,
        }

    def probe(self):
        self.runtime.reset()
        try:
            first = self.runtime.forward([1, 4, 5, 7], all_logits=True)
            second = self.runtime.forward([9], all_logits=True)
            return first, second
        finally:
            self.runtime.reset()

    def invalidate_graphs(self):
        invalidate = getattr(self.runtime, "invalidate_graphs", None)
        if invalidate is not None:
            invalidate()

    def switch(self, *, native=None, source=None):
        self.require_ready()
        # A8 banks cannot fall back to the W4A16 vendor path. Python-only
        # changes retain the active A8 kernel and its exact-output reference.
        if native is None and self.a8_manifest is not None:
            native = self.active
        if native is not None and native not in self.loaded:
            raise ValueError("native backend must be loaded and qualified first")
        code = compile(source, "<resident-candidate>", "exec") if source else None
        old = {name: getattr(*target(name)) for name in TARGETS}
        old_backend = self.layers[0].gate_up.backend
        started = time.monotonic()
        reference = self.probe()
        try:
            replacements = {}
            if code:
                scope = {}
                exec(code, scope)
                replacements = scope["replacements"]()
                if not isinstance(replacements, dict):
                    raise ValueError("replacements() must return a mapping")
                for name, function in replacements.items():
                    target(name)
                    if not callable(function):
                        raise ValueError("replacement must be callable")
            for name, original in self.originals.items():
                owner, attribute = target(name)
                setattr(owner, attribute, replacements.get(name, original))
            backend = self.loaded[native][1] if native else None
            for layer in self.layers:
                layer.gate_up.backend = layer.down.backend = backend
            self.invalidate_graphs()
            actual = self.probe()
            self.synchronize()
            if self.storage_identity() != self.identity:
                raise RuntimeError("resident weight storage changed")
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        except Exception:
            for name, function in old.items():
                owner, attribute = target(name)
                setattr(owner, attribute, function)
            for layer in self.layers:
                layer.gate_up.backend = layer.down.backend = old_backend
            try:
                self.invalidate_graphs()
                restored = self.probe()
                self.synchronize()
                if self.storage_identity() != self.identity:
                    raise RuntimeError("weight storage changed")
                for a, b in zip(restored, reference):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            except Exception as exc:
                self.failed = f"rollback failed: {exc}"
            raise
        self.active = native
        self.python_digest = (
            hashlib.sha256(source.encode()).hexdigest() if source else None
        )
        self.generation += 1
        return {
            **self.status(),
            "switch_seconds": time.monotonic() - started,
            "probe_passed": True,
        }
