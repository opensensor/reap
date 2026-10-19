#!/usr/bin/env python3
"""Build experimental Muse W4A8 kernels from pinned Qwen sources, without OPP.

Run with the target torch-npu Python and CANN environment. External sources
retain their original licenses. A new output directory freezes every input.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig


def zero_offsets(text):
    old = "        DataCopy(metadata[N], zw_[index], metadataCopy);"
    if text.count(old) != 1:
        raise ValueError("unexpected Qwen native INT4 offset load")
    return text.replace(
        old,
        "        Duplicate(metadata[N], static_cast<half>(0), N);\n"
        "        PipeBarrier<PIPE_V>();",
    )


def direct_quantizer(text):
    old = "  auto td = reinterpret_cast<__gm__ PackTiling*>(tiling);"
    if text.count(old) != 1 or text.count("KERNEL_TYPE_AIC_ONLY") != 1:
        raise ValueError("unexpected Qwen activation pack entry")
    return text.replace("KERNEL_TYPE_AIC_ONLY", "KERNEL_TYPE_AICORE").replace(
        old, "  AscendC::InitSocState();\n" + old
    )


def balanced_groups(text):
    """Rotate grouped output tiles across cores, retaining each tile's math."""
    start = "  __aicore__ inline void Process() {"
    end = "\n private:"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("unexpected Qwen grouped schedule scope")
    before, body = text.split(start)
    body, after = body.split(end)
    anchors = (
        (
            "    int64_t previousEnd = 0;",
            "    int64_t previousEnd = 0;\n    int64_t tileOffset = 0;",
        ),
        (
            "      for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {",
            "      const int64_t firstTile = (GetBlockIdx() + GetBlockNum() - tileOffset % GetBlockNum()) % GetBlockNum();\n"
            "      for (int64_t tile = firstTile; tile < n_ / N; tile += GetBlockNum()) {",
        ),
        (
            "      }\n    }\n  }\n",
            "      }\n      tileOffset += (n_ / N) * ((end - begin + M - 1) / M);\n    }\n  }\n",
        ),
    )
    for old, new in anchors:
        if body.count(old) != 1:
            raise ValueError("unexpected Qwen grouped schedule anchor")
        body = body.replace(old, new)
    return before + start + body + end + after


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen-source", type=Path, required=True)
    p.add_argument("--cann", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--namespace", default="muse_a8_v1")
    p.add_argument(
        "--schedule", choices=("baseline", "balanced", "wide"), default="baseline"
    )
    args = p.parse_args()
    if not re.fullmatch(r"muse_a8_[a-zA-Z0-9_]+", args.namespace):
        p.error("namespace must start muse_a8_ and contain identifier characters")
    import torch
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths

    root, cann, out = (
        args.qwen_source.resolve(),
        args.cann.resolve(),
        args.output.resolve(),
    )
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    for name in ("compile_reconstruction.cpp", "reconstruction_bridge.cpp"):
        shutil.copy2(root / "tools/glm_perf" / name, inputs / name)
    entry = "native_a8_wide.cpp" if args.schedule == "wide" else "native_a8.cpp"
    shutil.copy2(Path(__file__).with_name(entry), inputs / "native_a8.cpp")
    kernel_root = root / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel"
    for name in (
        "native_int4_schedule.h",
        "native_int4_route_groups.h",
        "qwen_w4_a8_int4_matmul_v310_tiling_data.h",
    ):
        text = (kernel_root / name).read_text()
        if name == "native_int4_schedule.h":
            if args.schedule != "baseline":
                text = balanced_groups(text)
            text = zero_offsets(text)
        (inputs / name).write_text(text)
    quantizer = (
        root / "csrc/gmm/qwen_w4_a8_pack_v310/op_kernel/qwen_w4_a8_pack_v310.cpp"
    )
    (inputs / "quantizer.cpp").write_text(direct_quantizer(quantizer.read_text()))
    commands = []

    def run(command):
        commands.append([str(x) for x in command])
        subprocess.run(commands[-1], check=True)

    compiler = out / "compile-native"
    run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            inputs / "compile_reconstruction.cpp",
            f"-I{cann}/include",
            f"-L{cann}/lib64",
            "-lacl_rtc",
            "-lascendcl",
            f"-Wl,-rpath,{cann}/lib64",
            "-o",
            compiler,
        ]
    )
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    for name in ("native_a8", "quantizer"):
        run(
            [
                compiler,
                inputs / f"{name}.cpp",
                out / f"{name}.bin",
                "--npu-arch=dav-2002",
                f"--sysroot={cann}/tools/hcc/sysroot",
                f"-isystem{target}",
                f"-isystem{target}/aarch64-target-linux-gnu",
                f"-I{inputs}",
            ]
        )
    npu = Path(torch_npu.__file__).parent
    includes = [
        *include_paths(),
        str(npu / "include"),
        str(cann / "include"),
        sysconfig.get_path("include"),
    ]
    libraries = [*library_paths(), str(npu / "lib"), str(cann / "lib64")]
    library = out / (args.namespace + ".so")
    run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-DGLM_RECONSTRUCTION_NAMESPACE={args.namespace}",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *[f"-I{x}" for x in includes],
            inputs / "reconstruction_bridge.cpp",
            *[f"-L{x}" for x in libraries],
            *[f"-Wl,-rpath,{x}" for x in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-ltorch_python",
            "-o",
            library,
        ]
    )

    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    manifest = dict(
        name=args.namespace, namespace=args.namespace, abi="muse-w4a8-zero64-v1"
    )
    for name, path in (
        ("library", library),
        ("kernel", out / "native_a8.bin"),
        ("quantizer", out / "quantizer.bin"),
    ):
        manifest[name] = dict(path=str(path), sha256=digest(path))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "build-receipt.json").write_text(
        json.dumps(
            dict(
                commands=commands,
                schedule=args.schedule,
                manifest=manifest,
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                sources={str(x.relative_to(out)): digest(x) for x in inputs.iterdir()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
