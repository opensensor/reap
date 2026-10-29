#!/usr/bin/env python3
"""Compare propagated W4A8 errors with captured, complete Muse MLP outputs.

Run with CANN and PYTHONPATH=src. The vendor candidate needs the Qwen A8
OPP; --direct-a8-manifest needs only the serving W4 control OPP. Captures must
contain block-{layer}-{1,4}.pt with x, output, and CPU mlp weights. Only trusted
locally generated captures are accepted by convention; loading uses weights_only.
All six output shards execute serially on one device to bound device memory.
This reports numerical differences, not full-model accuracy or throughput.
"""

import gc
import hashlib
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from safetensors.torch import load_file
from reap.muse_fast import FastMuseLayer, TiledBank, native_weight
from reap.muse_runtime import configuration
from reap.muse_tensor import partitions
from reap.muse_int4 import DirectA8, DirectA8Bank, NativeA8Bank, check_direct_manifest
from reap.muse_native import NativeW4, check_manifest
from reap.muse_packed import identity

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--captures", type=Path, required=True)
parser.add_argument("--packed-cache", type=Path, required=True)
parser.add_argument("--native-manifest", type=Path, required=True)
parser.add_argument(
    "--direct-a8-manifest",
    type=Path,
    help="Use the 64-aligned direct backend instead of the vendor A8 OPP",
)
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--layers", type=int, nargs="+", default=[0, 31, 61])
parser.add_argument("--device", default="npu:0")
args = parser.parse_args()
out = args.captures
config = configuration(json.loads((args.model / "config.json").read_text()))
journal = json.loads((args.model / "export-progress.json").read_text())
torch.set_num_threads(1)
torch.npu.set_device(args.device)
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
torch.npu.matmul.allow_hf32 = False
torch.use_deterministic_algorithms(True)
torch.ops.load_library(str(args.library))
manifest = json.loads(args.native_manifest.read_text())
check_manifest(manifest)
backend = NativeW4(manifest, [args.device])
direct_manifest = (
    json.loads(args.direct_a8_manifest.read_text()) if args.direct_a8_manifest else None
)
direct = DirectA8(direct_manifest, [args.device]) if direct_manifest else None
records = []


def error(a, b):
    diff = a.float() - b.float()
    return dict(
        exact=torch.equal(a, b),
        max_abs=float(diff.abs().max()),
        relative_rms=float(
            diff.square().mean().sqrt() / b.float().square().mean().sqrt()
        ),
        cosine=float(
            torch.nn.functional.cosine_similarity(
                a.float().flatten(), b.float().flatten(), dim=0
            )
        ),
    )


def run(index):
    saved = {
        count: torch.load(out / f"block-{index}-{count}.pt", weights_only=True)
        for count in (1, 4)
    }
    cache = (
        args.packed_cache / f"layer-{index:03d}-{identity(args.model, index, journal)}"
    )
    receipt = json.loads((cache / "receipt.json").read_text())
    with (cache / "weights.safetensors").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != receipt["sha256"]:
            raise ValueError("packed cache checksum mismatch")
    tensors = load_file(str(cache / "weights.safetensors"))
    weights = {
        name: native_weight(v.clone(), args.device)
        for name, v in saved[1]["weights"].items()
    }
    weights["input_layernorm.weight"] = (
        tensors["dense.input_layernorm.weight"].clone().to(args.device)
    )
    layer = FastMuseLayer(config, index, weights, SimpleNamespace(backend=True), None)
    layer.tensor_experts = SimpleNamespace()
    inputs = {count: record["x"].to(args.device) for count, record in saved.items()}
    active = []
    for x in inputs.values():
        flat = x.reshape(-1, x.shape[-1])
        scores = torch.sigmoid(
            torch.nn.functional.linear(flat.float(), weights["mlp.gate.weight"].float())
            * config.output_multiplier
        )
        active.append(
            torch.topk(
                scores + weights["mlp.gate.e_score_correction_bias"],
                config.num_experts_per_tok,
                dim=-1,
                sorted=False,
            ).indices.cpu()
        )
    selected = torch.cat([v.flatten() for v in active]).unique(sorted=True)
    lookup = torch.full((config.num_local_experts,), -1, dtype=torch.int32)
    lookup[selected] = torch.arange(len(selected), dtype=torch.int32)
    lookup = lookup.to(args.device)
    banks = {}
    for phase in ("gate_up", "down"):
        input_width, output_width, parts = receipt["geometry"][phase]
        width = output_width // 2 if phase == "gate_up" else output_width
        ranks = []
        for lo, hi in partitions(width, 6, index, alignment=32):
            ranges = (
                [(lo, hi), (width + lo, width + hi)]
                if phase == "gate_up"
                else [(lo, hi)]
            )
            hostparts = []
            for p in range(parts):
                cs = []
                for kind in ("codes", "scales"):
                    value = tensors[f"{phase}.{p}.{kind}"][selected]
                    cs.append(
                        torch.cat(
                            [value[:, a:b] for a, b in ranges], dim=1
                        ).contiguous()
                    )
                hostparts.append((*cs, torch.full_like(cs[1], 8, dtype=torch.int8)))
            host = TiledBank(hostparts, input_width, sum(b - a for a, b in ranges))
            native = (
                DirectA8Bank(host, args.device, direct)
                if direct is not None
                else NativeA8Bank(host, args.device)
            )
            old = TiledBank(
                [tuple(v.to(args.device) for v in part) for part in hostparts],
                host.inputs,
                host.outputs,
            )
            ranks.append((old, native))
        banks[phase] = ranks
    del tensors, hostparts, host, value, cs
    mode = "control"

    def operation(x, *, ids=None, ends=None):
        if ids is not None:
            mapped = lookup[ids.long()]
            if not bool((mapped >= 0).all()):
                raise ValueError("capture selected an unavailable expert")
            metadata = {"ids": mapped}
        else:
            counts = torch.diff(ends.cpu(), prepend=torch.zeros(1, dtype=torch.int64))
            if counts[selected].sum() != len(x):
                raise ValueError("grouped routes exceed selected expert coverage")
            metadata = {"ends": counts[selected].cumsum(0).to(args.device)}
        for phase in ("gate_up", "down"):
            results = []
            for old, new in banks[phase]:
                v = (
                    backend.linear(old, x, **metadata)
                    if mode == "control"
                    else new.linear(x, **metadata)
                )
                if phase == "gate_up":
                    gate, up = v.chunk(2, dim=-1)
                    v = torch.nn.functional.silu(gate) * up
                results.append(v)
            x = torch.cat(results, dim=-1)
        return x

    layer.expert_projection = operation
    for count, x in inputs.items():
        mode = "control"
        expected = layer.moe(x).cpu()
        baseline_error = error(expected, saved[count]["output"])
        print(
            json.dumps(dict(layer=index, tokens=count, control_error=baseline_error)),
            flush=True,
        )
        torch.testing.assert_close(expected, saved[count]["output"], rtol=0, atol=0)
        mode = "candidate"
        actual = layer.moe(x).cpu()
        if not bool(torch.isfinite(actual).all()):
            raise ValueError("nonfinite candidate MLP output")
        record = dict(
            layer=index,
            tokens=count,
            selected_experts=len(selected),
            control_exact=True,
            error=error(actual, expected),
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        args.output.write_text(
            json.dumps(
                dict(
                    completed=False,
                    scope="complete real MLP, propagated candidate up/down; sequential six shards on one NPU; no end-to-end throughput claim",
                    records=records,
                ),
                indent=2,
            )
            + "\n"
        )
    backend.geometry.clear()
    backend.validated_offsets.clear()


with torch.inference_mode():
    for index in args.layers:
        run(index)
        gc.collect()
        torch.npu.empty_cache()

report = json.loads(args.output.read_text())
report["completed"] = True
report["layers"] = args.layers
report["native_manifest_sha256"] = check_manifest(manifest)
report["candidate"] = "direct-zero64" if direct is not None else "vendor-padded128"
if direct_manifest is not None:
    report["direct_a8_manifest_sha256"] = check_direct_manifest(direct_manifest)
args.output.write_text(json.dumps(report, indent=2) + "\n")
