"""Compare one complete exported Muse layer with native CPU and Ascend forwards."""

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from reap.muse_runtime import PREFIX, LayerCache, configuration, load_layer, rms
from reap.muse_export import dequantize_weight
from reap.muse_streaming import _file_digest, _save_json


@torch.inference_mode()
def prepare(source, export, output):
    from transformers import AutoTokenizer
    from transformers.models.muse_spark.configuration_muse_spark import (
        MuseSparkTextConfig,
    )
    from transformers.models.muse_spark.modeling_muse_spark import (
        MuseSparkTextDecoderLayer,
        MuseSparkTextRotaryEmbedding,
    )
    from transformers.masking_utils import create_sliding_window_causal_mask

    source, export, output = map(Path, (source, export, output))
    if output.exists():
        raise ValueError("probe directory exists")
    journal = json.loads((export / "export-progress.json").read_text())
    raw = json.loads((source / "config.json").read_text())
    raw["text_config"]["num_local_experts"] = journal["identity"]["plan"]["keep"]
    c = configuration(raw)
    layer = load_layer(export, 0, c, "cpu", journal=journal)
    native_config = MuseSparkTextConfig.from_dict(raw["text_config"])
    native_config._attn_implementation = native_config._experts_implementation = "eager"
    with torch.device("meta"):
        native = MuseSparkTextDecoderLayer(native_config, 0)
    values = {}
    for name in native.state_dict():
        if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
            prefix, bank = name.rsplit(".", 1)
            projections = (
                ["gate_proj", "up_proj"] if bank == "gate_up_proj" else ["down_proj"]
            )
            geometry = (
                (
                    c.num_local_experts,
                    len(projections) * c.moe_intermediate_size,
                    c.moe_hidden_size,
                )
                if bank == "gate_up_proj"
                else (c.num_local_experts, c.moe_hidden_size, c.moe_intermediate_size)
            )
            result = torch.empty(geometry, dtype=torch.float16)
            for expert in range(c.num_local_experts):
                for part, projection in enumerate(projections):
                    key = f"{prefix}.{expert}.{projection}"
                    weight = dequantize_weight(
                        layer.w[key + ".weight"], layer.w[key + ".weight_scale"]
                    ).half()
                    result[
                        expert, part * weight.shape[0] : (part + 1) * weight.shape[0]
                    ] = weight
            values[name] = result
        elif name + "_scale" in layer.w:
            values[name] = dequantize_weight(
                layer.w[name], layer.w[name + "_scale"], bits=8
            ).half()
        else:
            values[name] = layer.w[name]
    native.load_state_dict(values, strict=True, assign=True)
    native.eval()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    ids = tokenizer.encode("def add(a, b):\n    return a + b", add_special_tokens=True)[
        :16
    ]
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = PREFIX + "embed_tokens.weight"
    with safe_open(source / weight_map[name], framework="pt") as reader:
        x = rms(
            F.embedding(torch.tensor([ids]), reader.get_tensor(name)).half(),
            c.rms_norm_eps,
        )
    positions = torch.arange(len(ids)).unsqueeze(0)
    mask = create_sliding_window_causal_mask(
        config=native_config,
        inputs_embeds=x,
        attention_mask=torch.ones(1, len(ids), dtype=torch.long),
        position_ids=positions,
        past_key_values=None,
    )
    rope = MuseSparkTextRotaryEmbedding(native_config)(x, positions)
    expected = native(
        x, attention_mask=mask, position_embeddings=rope, position_ids=positions
    )
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.005)
    output.mkdir(parents=True)
    save_file(
        {"x": x.contiguous(), "expected": expected.contiguous()},
        output / "reference.safetensors",
    )
    _save_json(output / "config.json", raw)
    subset = {key: value for key, value in journal.items() if key != "shards"}
    subset["shards"] = {
        k: journal["shards"][k]
        for k in (
            "dense-000.safetensors",
            "experts-000-ep0.safetensors",
            "experts-000-ep1.safetensors",
        )
    }
    subset["complete"] = False
    _save_json(output / "export-progress.json", subset)
    _save_json(
        output / "manifest.json",
        {
            "input_ids": ids,
            "reference_sha256": _file_digest(output / "reference.safetensors"),
            "native_reference": "Transformers d4804f9d Muse decoder with dequantized exported weights",
            "cpu_max_absolute_error": float(
                (actual.float() - expected.float()).abs().max()
            ),
        },
    )
    print(
        json.dumps(
            {
                "event": "prepared",
                "tokens": len(ids),
                "cpu_max_absolute_error": float(
                    (actual.float() - expected.float()).abs().max()
                ),
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def run(probe, library, device, output):
    import torch_npu

    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.ops.load_library(str(library))
    probe, output = Path(probe), Path(output)
    if output.exists():
        raise ValueError("result exists")
    manifest = json.loads((probe / "manifest.json").read_text())
    if _file_digest(probe / "reference.safetensors") != manifest["reference_sha256"]:
        raise ValueError("reference checksum differs")
    c = configuration(json.loads((probe / "config.json").read_text()))
    start = time.monotonic()
    layer = load_layer(
        probe / "model",
        0,
        c,
        f"npu:{device}",
        journal=json.loads((probe / "export-progress.json").read_text()),
    )
    print(
        json.dumps({"event": "loaded", "seconds": time.monotonic() - start}), flush=True
    )
    values = load_file(probe / "reference.safetensors")
    x, expected = values["x"].to(f"npu:{device}"), values["expected"]
    reports = []
    for cached in (False, True):
        start = time.monotonic()
        if cached:
            cache = LayerCache()
            actual = torch.cat(
                [
                    layer(x[:, i : i + 1], start=i, cache=cache).cpu()
                    for i in range(x.shape[1])
                ],
                dim=1,
            )
        else:
            actual = layer(x).cpu()
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.005)
        reports.append(
            {
                "cached": cached,
                "tokens": x.shape[1],
                "max_absolute_error": float(
                    (actual.float() - expected.float()).abs().max()
                ),
                "rmse": float(
                    (actual.float() - expected.float()).square().mean().sqrt()
                ),
                "seconds": time.monotonic() - start,
                "passed": True,
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    _save_json(
        output,
        {
            "cases": reports,
            "manifest_sha256": _file_digest(probe / "manifest.json"),
            "library_sha256": _file_digest(Path(library)),
            "scope": "one real complete decoder layer; native CPU reference; full and cached NPU forward",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8 if args.command == "prepare" else 2)
    if args.command == "prepare":
        if args.source is None or args.export is None:
            parser.error("prepare requires --source and --export")
        prepare(args.source, args.export, args.probe)
    else:
        if args.library is None or args.output is None:
            parser.error("run requires --library and --output")
        run(args.probe, args.library, args.device, args.output)


if __name__ == "__main__":
    main()
