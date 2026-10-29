"""Export a frozen Muse pruning plan with ModelSlim W4 experts/W8 attention.

CPU only, one expert projection at a time. This is a custom deployment format,
not a checkpoint that unmodified Transformers/vLLM can load. ModelSlim min/max
RTN is used without activation calibration. Original checkpoints are read-only.
"""

import argparse
import copy
import fcntl
import inspect
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from reap.muse_evaluate import PrunedMuseShardSource
from reap.muse_streaming import MuseShardSource, _file_digest, _save_json, _save_tensors


FORMAT = "muse_spark_w4a16_g128_w8attn_v1"


def slim_backend():
    from msmodelslim.format.common.pack import _pack_int4
    from msmodelslim.ir.api import calculate_qparam, quantize
    from msmodelslim.ir.qal import QDType, QScope, QStorage

    return _pack_int4, calculate_qparam, quantize, QDType, QScope, QStorage


@torch.inference_mode()
def quantize_weight(weight, *, bits=4, group_size=128):
    """Return signed codes and scales; zero offsets are implicit.

    W4 bytes pack consecutive K entries: even K in low nibble, odd K in high.
    Scale [N,K/128] is FP16. W8 is signed [N,K] with FP32 [N,1] scales.
    Codes use FP32 ModelSlim scales; dequantization uses the exported scales.
    """
    if weight.ndim != 2 or not weight.is_floating_point() or bits not in (4, 8):
        raise ValueError("expected floating [N,K] weight and bits 4 or 8")
    if bits == 4 and (
        group_size <= 0 or weight.shape[1] % group_size or weight.shape[1] % 2
    ):
        raise ValueError("W4 requires even K divisible by group_size")
    pack, calculate, quantize, dtype, scope, storage = slim_backend()
    weight = weight.float()
    if not torch.isfinite(weight).all():
        raise ValueError("nonfinite source weight")
    if bits == 4:
        groups = weight.reshape(weight.shape[0], -1, group_size)
        params = calculate(
            groups.amin(-1), groups.amax(-1), dtype.INT4, scope.PER_GROUP, True
        )
        params.ext["group_size"] = group_size
        codes = (
            quantize(storage(dtype.FLOAT, weight.t()), params).value.t().contiguous()
        )
        packed = pack(codes).contiguous()
        scales = params.ext["scale"].half().contiguous()
    else:
        params = calculate(
            weight.amin(-1, keepdim=True),
            weight.amax(-1, keepdim=True),
            dtype.INT8,
            scope.PER_CHANNEL,
            True,
        )
        packed = quantize(storage(dtype.FLOAT, weight), params).value.contiguous()
        scales = params.ext["scale"].float().contiguous()
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("exported scales must be finite and positive")
    if not (params.ext["offset"] == 0).all():
        raise ValueError("unexpected nonzero symmetric offset")
    return {"weight": packed, "weight_scale": scales}


def dequantize_weight(codes, scales, *, bits=4, group_size=128):
    if codes.dtype != torch.int8 or codes.ndim != 2:
        raise ValueError("codes must be a signed int8 matrix")
    if bits == 4:
        unsigned = codes.to(torch.uint8)
        nibble = (
            torch.stack((unsigned & 15, unsigned >> 4), dim=-1)
            .reshape(codes.shape[0], -1)
            .to(torch.int8)
        )
        signed = torch.where(nibble >= 8, nibble - 16, nibble)
        return (
            signed.reshape(codes.shape[0], -1, group_size).float()
            * scales.float().unsqueeze(-1)
        ).reshape(codes.shape[0], -1)
    if bits == 8:
        return codes.float() * scales.float()
    raise ValueError("bits must be 4 or 8")


def _backend_identity():
    import msmodelslim
    from msmodelslim.ir.api.impl import int_quantization
    from msmodelslim.format.common import pack

    root = Path(msmodelslim.__file__).resolve().parent
    # Capture implementation, API, schemas and import patches, not just Git HEAD.
    paths = sorted(
        {
            *root.joinpath("ir").rglob("*.py"),
            *root.joinpath("utils").glob("patch*.py"),
            Path(inspect.getfile(pack)),
            Path(inspect.getfile(int_quantization)),
            root / "__init__.py",
        }
    )
    return {str(p.relative_to(root)): _file_digest(p) for p in paths}


def export_checkpoint(
    source_root, plan, output, *, resume=False, stop_after_layer=None, progress=print
):
    source = MuseShardSource(source_root)
    PrunedMuseShardSource(
        source_root, plan
    )  # Validate the original plan, no materialization.
    c = source.config
    if plan["keep"] % 2 or c.moe_hidden_size % 128 or c.moe_intermediate_size % 128:
        raise ValueError(
            "EP2 needs even retained count; W4 group128 must divide expert dimensions"
        )
    if any(kind != "sparse" for kind in c.mlp_layer_types):
        raise ValueError("initial deployment export requires all-sparse Muse layers")
    output = Path(output).resolve()
    if (
        output == source.root
        or output.is_relative_to(source.root)
        or source.root.is_relative_to(output)
    ):
        raise ValueError("export destination must be separate from source")
    stop = c.num_hidden_layers if stop_after_layer is None else stop_after_layer
    if not 0 <= stop <= c.num_hidden_layers:
        raise ValueError("stop_after_layer outside model")
    identity = {
        "format": FORMAT,
        "source": source.identity,
        "plan": plan,
        "exporter_sha256": _file_digest(Path(__file__)),
        "modelslim": _backend_identity(),
    }
    # Header preflight before creating output. Never infer bank geometry from filenames.
    for i in range(c.num_hidden_layers):
        for bank, shape in (
            (
                "gate_up_proj",
                [c.num_local_experts, 2 * c.moe_intermediate_size, c.moe_hidden_size],
            ),
            (
                "down_proj",
                [c.num_local_experts, c.moe_hidden_size, c.moe_intermediate_size],
            ),
        ):
            name = source.prefix + f"layers.{i}.mlp.experts.{bank}"
            with safe_open(
                source.files[source.weight_map[name]], framework="pt"
            ) as reader:
                view = reader.get_slice(name)
                if view.get_shape() != shape or view.get_dtype() not in (
                    "BF16",
                    "F16",
                    "F32",
                ):
                    raise ValueError(f"invalid source expert bank: {name}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".export.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another exporter holds the destination") from exc
        journal_path = output / "export-progress.json"
        if journal_path.exists():
            if not resume:
                raise ValueError("existing export requires resume")
            journal = json.loads(journal_path.read_text())
            if journal["identity"] != identity:
                raise ValueError("export resume identity differs")
            for filename, receipt in journal["shards"].items():
                if (
                    Path(filename).name != filename
                    or _file_digest(output / filename) != receipt["sha256"]
                ):
                    raise ValueError("completed export shard checksum differs")
        else:
            if resume or any(p.name != ".export.lock" for p in output.iterdir()):
                raise ValueError("new export requires an empty destination")
            journal = {
                "identity": identity,
                "complete": False,
                "next_layer": 0,
                "shards": {},
            }
            _save_json(journal_path, journal)

        def save(filename, values):
            _save_tensors(output / filename, values)
            receipt = {
                "sha256": _file_digest(output / filename),
                "bytes": sum(v.numel() * v.element_size() for v in values.values()),
                "tensors": {
                    k: {
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                        "bytes": v.numel() * v.element_size(),
                    }
                    for k, v in values.items()
                },
            }
            journal["shards"][filename] = receipt
            _save_json(journal_path, journal)
            progress(
                json.dumps(
                    {
                        "event": "shard",
                        "file": filename,
                        "bytes": receipt["bytes"],
                        "sha256": receipt["sha256"],
                    }
                )
            )

        def read(name):
            with safe_open(
                source.files[source.weight_map[name]], framework="pt"
            ) as reader:
                return reader.get_tensor(name)

        def convert_float(name, tensor):
            if tensor.is_floating_point():
                tensor = tensor.to(
                    torch.float32 if tensor.dtype == torch.float32 else torch.float16
                )
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"float export overflows or is nonfinite: {name}")
            return tensor.contiguous()

        for i in range(stop):
            start = time.monotonic()
            prefix = source.prefix + f"layers.{i}."
            keep = plan["retained_experts"][str(i)]
            half = len(keep) // 2
            for ep in range(2):
                filename = f"experts-{i:03d}-ep{ep}.safetensors"
                if filename in journal["shards"]:
                    continue
                values = {}
                for new_id in range(ep * half, (ep + 1) * half):
                    old_id = keep[new_id]
                    for bank, projections in (
                        ("gate_up_proj", ("gate_proj", "up_proj")),
                        ("down_proj", ("down_proj",)),
                    ):
                        name = prefix + "mlp.experts." + bank
                        with safe_open(
                            source.files[source.weight_map[name]], framework="pt"
                        ) as reader:
                            view = reader.get_slice(name)
                            for part, projection in enumerate(projections):
                                weight = (
                                    view[
                                        old_id,
                                        part * c.moe_intermediate_size : (part + 1)
                                        * c.moe_intermediate_size,
                                        :,
                                    ]
                                    if bank == "gate_up_proj"
                                    else view[old_id, :, :]
                                )
                                for kind, value in quantize_weight(weight).items():
                                    values[
                                        f"{prefix}mlp.experts.{new_id}.{projection}.{kind}"
                                    ] = value
                                del weight
                    if (new_id + 1) % 8 == 0:
                        progress(
                            json.dumps(
                                {
                                    "event": "expert",
                                    "layer": i,
                                    "ep": ep,
                                    "completed_id": new_id,
                                    "seconds": time.monotonic() - start,
                                }
                            )
                        )
                save(filename, values)
                del values
            filename = f"dense-{i:03d}.safetensors"
            if filename not in journal["shards"]:
                values = {}
                for name in sorted(
                    k for k in source.weight_map if k.startswith(prefix)
                ):
                    if name in (
                        prefix + "mlp.experts.gate_up_proj",
                        prefix + "mlp.experts.down_proj",
                    ):
                        continue
                    tensor = read(name)
                    if name in (
                        prefix + "mlp.gate.weight",
                        prefix + "mlp.gate.e_score_correction_bias",
                    ):
                        tensor = tensor[keep]
                    if name.startswith(prefix + "self_attn.") and name.endswith(
                        "_proj.weight"
                    ):
                        for kind, value in quantize_weight(tensor, bits=8).items():
                            values[name.removesuffix("weight") + kind] = value
                    else:
                        values[name] = convert_float(name, tensor)
                    del tensor
                save(filename, values)
                del values
            journal["next_layer"] = max(journal["next_layer"], i + 1)
            _save_json(journal_path, journal)
        if stop < c.num_hidden_layers:
            return journal
        for name in sorted(source.weight_map):
            if not (
                name.startswith(source.prefix) or name == "lm_head.weight"
            ) or name.startswith(source.prefix + "layers."):
                continue
            filename = "global-" + name.replace(".", "_") + ".safetensors"
            if filename not in journal["shards"]:
                save(filename, {name: convert_float(name, read(name))})
        index = {}
        total_bytes = 0
        for filename, receipt in journal["shards"].items():
            for name in receipt["tensors"]:
                if name in index:
                    raise ValueError("duplicate output tensor")
                index[name] = filename
            total_bytes += receipt["bytes"]
        config = copy.deepcopy(json.loads((source.root / "config.json").read_text()))
        text = config.get("text_config", config)
        text["num_local_experts"] = plan["keep"]
        text["dtype"] = "float16"
        config["quantization_config"] = {
            "quant_method": FORMAT,
            "group_size": 128,
            "symmetric": True,
            "expert_scale_dtype": "float16",
            "attention_bits": 8,
            "attention_scale_dtype": "float32",
        }
        config["ascend_text_only"] = True
        _save_json(output / "config.json", config)
        _save_json(output / "retained-experts.json", plan)
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "generation_config.json",
            "LICENSE.md",
            "USE_POLICY.md",
        ):
            path = source.root / filename
            if path.is_file():
                from reap.muse_streaming import _atomic

                _atomic(
                    output / filename,
                    lambda dest, path=path: dest.write_bytes(path.read_bytes()),
                )
        _save_json(
            output / "deployment.json",
            {
                "format": FORMAT,
                "text_only": True,
                "serving_ready": False,
                "retained_experts": plan["keep"],
                "top_k": c.num_experts_per_tok,
                "weight_bytes": total_bytes,
                "quantizer": "msModelSlim IR symmetric minmax RTN; no activation calibration",
                "expert_layout": "signed int4 [N,K/2], even K low nibble; FP16 scale [N,K/128]; implicit zero offset",
                "attention_layout": "signed int8 [N,K], FP32 per-output scale [N,1]",
                "runtime_required": "Muse text adapter, TP2/PP3/EP2, packed W4A16 and W8A16 operators",
                "source_plan_metric": plan["metric"],
                "quantized_quality_measured": False,
            },
        )
        # Publish the index last. Partial output must never look loadable.
        _save_json(
            output / "model.safetensors.index.json",
            {
                "metadata": {"total_size": total_bytes, "format": FORMAT},
                "weight_map": index,
            },
        )
        journal["complete"] = True
        _save_json(journal_path, journal)
        return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    result = export_checkpoint(
        args.source,
        json.loads(args.plan.read_text()),
        args.output,
        resume=args.resume,
        stop_after_layer=args.stop_after_layer,
        progress=lambda value: print(value, flush=True),
    )
    print(
        json.dumps(
            {"complete": result["complete"], "next_layer": result["next_layer"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
