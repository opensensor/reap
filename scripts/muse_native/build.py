#!/usr/bin/env python3
"""Build a versioned 310P direct-launch W4 module from a pinned Qwen source tree.

Run with the target torch_npu Python after sourcing CANN's set_env.sh. External
CANN/Catlass/Qwen sources retain their upstream licenses; they are not vendored
in REAP. Copies and SHA256 receipts freeze the actual build inputs, including
local modifications. Output directories must be new.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig

import torch
import torch_npu
from torch.utils.cpp_extension import include_paths, library_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen-source", type=Path, required=True)
    p.add_argument("--catlass-include", type=Path, required=True)
    p.add_argument("--cann", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--namespace", default="muse_w4_v1")
    p.add_argument("--tile", type=int, choices=(512, 1024), default=1024)
    args = p.parse_args()
    if not re.fullmatch(r"muse_w4_[a-zA-Z0-9_]+", args.namespace):
        p.error("namespace must start muse_w4_ and contain only identifier characters")
    out, root, cann = (
        args.output.resolve(),
        args.qwen_source.resolve(),
        args.cann.resolve(),
    )
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    source = root / "csrc/gmm/qwen_w4_group_matmul_v310/op_kernel"
    for name in ("compile_reconstruction.cpp", "reconstruction_bridge.cpp"):
        shutil.copy2(root / "tools/glm_perf" / name, inputs / name)
    shutil.copy2(Path(__file__).with_name("native_w4.cpp"), inputs / "native_w4.cpp")
    shutil.copytree(root / "csrc/moe/common", inputs / "common")
    shutil.copytree(args.catlass_include, inputs / "catlass")
    for name in (
        "compat_310p.h",
        "qwen_w4_group_matmul_v310_tiling_data.h",
        "qwen_w4_group_matmul_v310.h",
    ):
        text = (source / name).read_text()
        if name == "qwen_w4_group_matmul_v310.h":
            if text.count("QW4_TILED_K_BATCH = 512;") != 1:
                raise ValueError("unexpected upstream unpacking constant")
            text = text.replace(
                "QW4_TILED_K_BATCH = 512;", f"QW4_TILED_K_BATCH = {args.tile};"
            )
        (inputs / name).write_text(text)
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
    binary = out / "muse_w4.bin"
    run(
        [
            compiler,
            inputs / "native_w4.cpp",
            binary,
            "--npu-arch=dav-2002",
            f"--sysroot={cann}/tools/hcc/sysroot",
            f"-isystem{target}",
            f"-isystem{target}/aarch64-target-linux-gnu",
            f"-I{inputs}",
            f"-I{inputs}/catlass",
            f"-I{inputs}/common",
        ]
    )
    npu = Path(torch_npu.__file__).parent
    inc = [
        *include_paths(),
        str(npu / "include"),
        str(cann / "include"),
        sysconfig.get_path("include"),
    ]
    libs = [*library_paths(), str(npu / "lib"), str(cann / "lib64")]
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
            *[f"-I{x}" for x in inc],
            inputs / "reconstruction_bridge.cpp",
            *[f"-L{x}" for x in libs],
            *[f"-Wl,-rpath,{x}" for x in libs],
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
        name=f"{args.namespace}_k{args.tile}",
        namespace=args.namespace,
        tile=args.tile,
        abi="muse-w4-v1",
        library=dict(path=str(library), sha256=digest(library)),
        kernel=dict(path=str(binary), sha256=digest(binary)),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = dict(
        commands=commands,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        sources={
            str(x.relative_to(out)): digest(x) for x in inputs.rglob("*") if x.is_file()
        },
        manifest=manifest,
    )
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
