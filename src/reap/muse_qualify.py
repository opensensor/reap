"""Prepare real Muse W4 projections and compare the existing 310P operator.

This tests the quantized projection contract, not full-model quality or serving.
The run command is intended for either msprof's 310P simulator application or a
small isolated physical-device check. It never installs kernels or stops servers.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from reap.muse_streaming import _file_digest, _save_json, _save_tensors
from reap.muse_ascend import w4_linear


@torch.inference_mode()
def prepare(export, output):
    from reap.muse_export import dequantize_weight

    export, output = Path(export), Path(output)
    journal = json.loads((export / "export-progress.json").read_text())
    filename = "experts-000-ep0.safetensors"
    if _file_digest(export / filename) != journal["shards"][filename]["sha256"]:
        raise ValueError("source exported shard checksum differs")
    if output.exists():
        raise ValueError("qualification directory already exists")
    values = load_file(export / filename)
    plan = journal["identity"]["plan"]
    generator = torch.Generator().manual_seed(171028)
    results = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"model.language_model.layers.0.mlp.experts.0.{projection}."
        codes, scales = values[prefix + "weight"], values[prefix + "weight_scale"]
        weight = dequantize_weight(codes, scales).half()
        inputs = (torch.randn(129, weight.shape[1], generator=generator) * 0.1).half()
        tensors = {
            "codes": codes,
            "scale": scales,
            "offset": torch.zeros_like(scales, dtype=torch.int8),
            "x": inputs,
            "expected": F.linear(inputs, weight),
        }
        path = output / f"{projection}.safetensors"
        _save_tensors(path, tensors)
        results[projection] = {
            "shape": list(weight.shape),
            "sha256": _file_digest(path),
        }
    _save_json(
        output / "manifest.json",
        {
            "schema": 1,
            "format": journal["identity"]["format"],
            "source_shard_sha256": journal["shards"][filename]["sha256"],
            "retained_id": 0,
            "original_id": plan["retained_experts"]["0"][0],
            "reference": "CPU FP16 F.linear using dequantized exported W4/FP16 scales",
            "projections": results,
        },
    )


@torch.inference_mode()
def run(
    data,
    library,
    output,
    *,
    tokens=(1, 17, 128, 129),
    rows=None,
    projection=None,
    device=0,
):
    import torch_npu

    data, library, output = Path(data), Path(library), Path(output)
    if output.exists():
        raise ValueError("result already exists")
    if (
        not tokens
        or min(tokens) < 1
        or max(tokens) > 129
        or (rows is not None and rows <= 0)
    ):
        raise ValueError("invalid test geometry")
    torch.set_num_threads(2)
    torch.ops.load_library(str(library))
    op = torch.ops._C_ascend.npu_qwen_w4_group_matmul_310
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    manifest = json.loads((data / "manifest.json").read_text())
    reports = []
    for name, record in manifest["projections"].items():
        if projection is not None and name != projection:
            continue
        path = data / f"{name}.safetensors"
        if _file_digest(path) != record["sha256"]:
            raise ValueError("qualification data checksum differs")
        values = load_file(path)
        width = record["shape"][0] if rows is None else rows
        if width > record["shape"][0] or width % 128:
            raise ValueError("test output rows must be aligned and within source")
        packed = [
            values[k][:width].contiguous().npu() for k in ("codes", "scale", "offset")
        ]
        for count in tokens:
            inputs = values["x"][:count].contiguous().npu()
            actual = w4_linear(inputs, packed[0], packed[1], op=op).cpu()
            expected = values["expected"][:count, :width]
            torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.003)
            difference = actual.float() - expected.float()
            reports.append(
                {
                    "projection": name,
                    "tokens": count,
                    "N": width,
                    "K": values["x"].shape[1],
                    "max_absolute_error": float(difference.abs().max()),
                    "rmse": float(difference.square().mean().sqrt()),
                    "passed": True,
                }
            )
        del packed
    if not reports:
        raise ValueError("no qualification cases selected")
    maps = Path("/proc/self/maps").read_text().splitlines()
    simulator_libraries = sorted(
        {line.split()[-1] for line in maps if "/simulator/" in line}
    )
    result = {
        "cases": reports,
        "library_sha256": _file_digest(library),
        "manifest_sha256": _file_digest(data / "manifest.json"),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "simulator_libraries": simulator_libraries,
        "scope": "same-quantized projection parity; not full-model inference",
        "adapter": "split K at 2048 and M at 128; FP32 partial accumulation",
    }
    _save_json(output, result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--export", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    test = commands.add_parser("run")
    test.add_argument("--data", type=Path, required=True)
    test.add_argument("--library", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--tokens", type=int, nargs="+", default=[1, 17, 128, 129])
    test.add_argument("--rows", type=int)
    test.add_argument("--projection", choices=["gate_proj", "up_proj", "down_proj"])
    test.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        torch.set_num_threads(8)
        prepare(args.export, args.output)
    else:
        run(
            args.data,
            args.library,
            args.output,
            tokens=args.tokens,
            rows=args.rows,
            projection=args.projection,
            device=args.device,
        )


if __name__ == "__main__":
    main()
