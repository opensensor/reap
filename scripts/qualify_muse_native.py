#!/usr/bin/env python3
"""Qualify a native W4 module against a real Muse layer with resident weights.

Source CANN's environment and select the matching custom OPP before invoking.
All timings synchronize the NPU. This is an isolated MoE test, not model tok/s.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu

from reap.muse_fast import load_fast_layer
from reap.muse_native import NativeW4, check_manifest
from reap.muse_runtime import configuration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--device", default="npu:1")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 7, 128, 512])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repeats < 1 or any(not 1 <= row <= 512 for row in args.rows):
        p.error("positive repeats and row counts in 1..512 required")
    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.use_deterministic_algorithms(True)
    torch.ops.load_library(str(args.library))
    manifest = json.loads(args.manifest.read_text())
    check_manifest(manifest)
    config = configuration(json.loads((args.model / "config.json").read_text()))
    layer = load_fast_layer(
        args.model, args.layer, config, args.device, packed_cache=args.cache
    )
    backend = NativeW4(manifest, [args.device])
    torch.set_num_threads(2)
    records = []
    with torch.inference_mode():
        for rows in args.rows:
            torch.manual_seed(415)
            x = (torch.randn(1, rows, config.hidden_size) * 0.1).half().to(args.device)
            reference = None
            for name, value in [("baseline", None), (manifest["name"], backend)]:
                layer.gate_up.backend = layer.down.backend = value
                warm = layer.moe(x).cpu()
                if reference is None:
                    reference = warm
                torch.testing.assert_close(warm, reference, rtol=0, atol=0)
                timings = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    actual = layer.moe(x)
                    torch.npu.synchronize(args.device)
                    timings.append(time.perf_counter() - started)
                    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
                record = dict(
                    rows=rows,
                    backend=name,
                    seconds=statistics.median(timings),
                    repeats=args.repeats,
                    max_error=0.0,
                )
                records.append(record)
                print(json.dumps(record), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                layer=args.layer,
                device=args.device,
                deterministic=True,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
