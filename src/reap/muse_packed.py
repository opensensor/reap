"""Atomic, checksummed CPU safetensors cache of the lossless serving layout.

A hit is bound to the source export receipts, config and packing ABI; it verifies
its own bytes instead of rereading and repacking the canonical expert shards.
NPU-specific NZ formatting is still performed on transfer. No pickle is loaded.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file

ABI = "muse-qwen-nz-w4-split2048-dense-fp16-v1"


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(root, index, journal):
    filenames = [
        f"dense-{index:03d}.safetensors",
        f"experts-{index:03d}-ep0.safetensors",
        f"experts-{index:03d}-ep1.safetensors",
    ]
    value = {
        "abi": ABI,
        "config": digest(Path(root) / "config.json"),
        "shards": {name: journal["shards"][name] for name in filenames},
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def encode(layer):
    tensors = {"dense." + name: value.contiguous() for name, value in layer.w.items()}
    geometry = {}
    for name in ("gate_up", "down"):
        bank = getattr(layer, name)
        geometry[name] = [bank.inputs, bank.outputs, len(bank.parts)]
        for i, (codes, scales, _) in enumerate(bank.parts):
            tensors[f"{name}.{i}.codes"] = codes
            tensors[f"{name}.{i}.scales"] = scales
    return tensors, geometry


def decode(tensors, geometry, config, index, device):
    from reap.muse_fast import FastMuseLayer, TiledBank, native_weight

    def staging(value):
        # Cached tensors are views into a multi-GB mmap. Use an independently
        # allocated, aligned host buffer for the driver's DMA registration.
        return value.clone() if str(device).startswith("npu:") else value

    dense = {
        name.removeprefix("dense."): native_weight(staging(value), device)
        for name, value in tensors.items()
        if name.startswith("dense.")
    }
    banks = []
    for name in ("gate_up", "down"):
        inputs, outputs, count = geometry[name]
        parts = []
        for i in range(count):
            q = staging(tensors[f"{name}.{i}.codes"]).to(device)
            s = staging(tensors[f"{name}.{i}.scales"]).to(device)
            parts.append((q, s, torch.full_like(s, 8, dtype=torch.int8)))
        banks.append(TiledBank(parts, inputs, outputs))
    return FastMuseLayer(config, index, dense, *banks)


def load_cached_layer(root, index, config, device, cache, *, journal, verify=True):
    from reap.muse_fast import load_fast_layer

    started = time.monotonic()
    if not verify:
        raise ValueError("packed cache requires verification")
    journal = journal or json.loads((Path(root) / "export-progress.json").read_text())
    key = identity(root, index, journal)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"layer-{index:03d}-{key}"
    with (cache / f"layer-{index:03d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = destination.exists()
        if not hit:
            layer = load_fast_layer(root, index, config, "cpu", journal=journal)
            tensors, geometry = encode(layer)
            temporary = Path(tempfile.mkdtemp(prefix=".packing-", dir=cache))
            try:
                filename = temporary / "weights.safetensors"
                save_file(tensors, str(filename))
                receipt = {
                    "identity": key,
                    "abi": ABI,
                    "geometry": geometry,
                    "sha256": digest(filename),
                    "bytes": filename.stat().st_size,
                }
                (temporary / "receipt.json").write_text(json.dumps(receipt) + "\n")
                for path in temporary.iterdir():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
                os.rename(temporary, destination)
                fd = os.open(cache, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            del tensors, layer
        receipt = json.loads((destination / "receipt.json").read_text())
        filename = destination / "weights.safetensors"
        if (
            receipt["identity"] != key
            or receipt["abi"] != ABI
            or filename.stat().st_size != receipt["bytes"]
            or digest(filename) != receipt["sha256"]
        ):
            raise ValueError(f"packed cache integrity failure: {destination}")
        tensors = load_file(str(filename))
    # Device transfer must not hold the writer lock or stall other cache readers.
    result = decode(tensors, receipt["geometry"], config, index, device)
    print(
        json.dumps(
            {
                "event": "packed_cache",
                "layer": index,
                "hit": hit,
                "seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )
    return result


def main():
    import argparse
    from reap.muse_runtime import configuration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(8)
    config = configuration(json.loads((args.model / "config.json").read_text()))
    journal = json.loads((args.model / "export-progress.json").read_text())
    for index in args.layers or range(config.num_hidden_layers):
        if not 0 <= index < config.num_hidden_layers:
            parser.error("layer index outside model")
        layer = load_cached_layer(
            args.model, index, config, "cpu", args.cache, journal=journal
        )
        del layer


if __name__ == "__main__":
    main()
