"""Check the optimized RMS dispatch against CPU math on an Ascend device.

Run with CANN sourced and PYTHONPATH=src. This does not load model weights or
custom libraries. Width eight exercises the reference fallback needed by the
installed 310P kernel; aligned widths exercise fused dispatch.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from reap.muse_fast import FastMuseLayer
from reap.muse_runtime import rms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(2)
    layer = FastMuseLayer.__new__(FastMuseLayer)
    layer.unit_weights = {}
    generator = torch.Generator().manual_seed(76)
    records = []
    with torch.inference_mode():
        for width in (8, 16, 32, 64, 256, 4096, 8192):
            for scale in (0, 0.0001, 1):
                x = (torch.randn(32, width, generator=generator) * scale).half()
                weight = (1 + torch.randn(width, generator=generator) * 0.1).half()
                device_x, device_weight = x.to(args.device), weight.to(args.device)
                for eps in (1e-5, 1e-8):
                    for weighted in (False, True):
                        expected = rms(x, eps, weight if weighted else None)
                        errors = []
                        for _ in range(3):
                            actual = layer.fast_rms(
                                device_x, eps, device_weight if weighted else None
                            ).cpu()
                            torch.testing.assert_close(
                                actual, expected, rtol=0.002, atol=0.002
                            )
                            errors.append(float((actual - expected).abs().max()))
                        records.append(
                            dict(
                                width=width,
                                scale=scale,
                                eps=eps,
                                weighted=weighted,
                                max_errors=errors,
                            )
                        )
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(dict(passed=len(records), output=str(args.output))))


if __name__ == "__main__":
    main()
