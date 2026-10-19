"""Atomic native INT4 host cache, bound to export receipts and shard topology.

This is an opt-in companion to the experimental A8 runtime. It stores the same
W4 codes/scales as the direct conversion, plus weight sums; no activations are
cached. Cache corruption is an error, never a reason to skip verification.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

from safetensors.torch import load_file, save_file

from reap.muse_int4 import DirectA8Bank
from reap.muse_packed import digest, identity as source_identity
from reap.muse_tensor import partitions, slice_bank

ABI = "muse-a8-native-zero64-g128-split2048-v1"
KINDS = ("codes", "scales", "weight_sums")


def identity(root, index, journal, ranks):
    value = dict(abi=ABI, source=source_identity(root, index, journal), ranks=ranks)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def widths(config, index, ranks):
    up = partitions(config.moe_intermediate_size, ranks, index, alignment=32)
    down = partitions(config.moe_hidden_size, ranks, index, alignment=32)
    return up, down


def encode(cpu, ranks):
    if (
        cpu.gate_up.inputs != cpu.c.moe_hidden_size
        or cpu.gate_up.outputs != 2 * cpu.c.moe_intermediate_size
        or cpu.down.inputs != cpu.c.moe_intermediate_size
        or cpu.down.outputs != cpu.c.moe_hidden_size
    ):
        raise ValueError("native A8 cache layer geometry differs from configuration")
    tensors = {"dense." + k: v.contiguous() for k, v in cpu.w.items()}
    up, down = widths(cpu.c, cpu.index, ranks)
    for rank, ((lo, hi), (down_lo, down_hi)) in enumerate(zip(up, down)):
        middle = cpu.down.inputs
        for phase, ranges in (
            ("gate_up", [(lo, hi), (middle + lo, middle + hi)]),
            ("down", [(down_lo, down_hi)]),
        ):
            host = slice_bank(getattr(cpu, phase), ranges, "cpu")
            bank = DirectA8Bank(host, "cpu", None)
            for part, values in enumerate(bank.parts):
                for kind, value in zip(KINDS, values):
                    tensors[f"rank.{rank}.{phase}.{part}.{kind}"] = value
    return tensors


def decode(tensors, config, index, ranks):
    up, down = widths(config, index, ranks)
    dense = {
        k.removeprefix("dense."): v
        for k, v in tensors.items()
        if k.startswith("dense.")
    }
    consumed = {"dense." + k for k in dense}
    banks = []
    for rank, ((lo, hi), (down_lo, down_hi)) in enumerate(zip(up, down)):
        pair = []
        for phase, inputs, outputs in (
            ("gate_up", config.moe_hidden_size, 2 * (hi - lo)),
            ("down", config.moe_intermediate_size, down_hi - down_lo),
        ):
            parts = []
            for part in range((inputs + 2047) // 2048):
                names = [f"rank.{rank}.{phase}.{part}.{kind}" for kind in KINDS]
                consumed.update(names)
                parts.append(tuple(tensors[name] for name in names))
            if any(p[0].shape[0] != config.num_local_experts for p in parts):
                raise ValueError("cached A8 expert count differs from configuration")
            # Keep mmap views on the host; shard_layer materializes independent
            # staging immediately before each device transfer.
            pair.append(SimpleNamespace(parts=parts, inputs=inputs, outputs=outputs))
        banks.append(pair)
    if consumed != set(tensors) or not dense:
        raise ValueError("invalid native A8 cache tensor inventory")
    return SimpleNamespace(
        c=config,
        index=index,
        w=dense,
        native_a8_ranks=banks,
        gate_up=SimpleNamespace(
            inputs=config.moe_hidden_size, outputs=2 * config.moe_intermediate_size
        ),
        down=SimpleNamespace(
            inputs=config.moe_intermediate_size, outputs=config.moe_hidden_size
        ),
    )


def load_native_layer(
    root, index, config, ranks, cache, *, journal, packed_cache=None, verify=True
):
    if not verify:
        raise ValueError("native A8 cache requires verification")
    widths(config, index, ranks)  # Reject invalid topology before creating files.
    started = time.monotonic()
    key = identity(root, index, journal, ranks)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"layer-{index:03d}-{key}"
    with (cache / f"layer-{index:03d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = destination.exists()
        if not hit:
            from reap.muse_fast import load_fast_layer

            cpu = load_fast_layer(
                root,
                index,
                config,
                "cpu",
                journal=journal,
                verify=True,
                packed_cache=packed_cache,
            )
            tensors = encode(cpu, ranks)
            temporary = Path(tempfile.mkdtemp(prefix=".packing-", dir=cache))
            try:
                path = temporary / "weights.safetensors"
                save_file(tensors, str(path))
                receipt = dict(
                    identity=key,
                    abi=ABI,
                    ranks=ranks,
                    sha256=digest(path),
                    bytes=path.stat().st_size,
                )
                (temporary / "receipt.json").write_text(json.dumps(receipt) + "\n")
                for path in temporary.iterdir():
                    with path.open("rb") as f:
                        os.fsync(f.fileno())
                fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.rename(temporary, destination)
                fd = os.open(cache, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            del tensors, cpu
        path = destination / "weights.safetensors"
        receipt = json.loads((destination / "receipt.json").read_text())
        if (
            receipt["identity"] != key
            or receipt["abi"] != ABI
            or receipt["ranks"] != ranks
            or path.stat().st_size != receipt["bytes"]
            or digest(path) != receipt["sha256"]
        ):
            raise ValueError(f"native A8 cache integrity failure: {destination}")
        layer = decode(load_file(str(path)), config, index, ranks)
    print(
        json.dumps(
            dict(
                event="native_a8_cache",
                layer=index,
                hit=hit,
                seconds=time.monotonic() - started,
            )
        ),
        flush=True,
    )
    return layer
