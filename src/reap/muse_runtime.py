"""Single-request Muse text runtime for the custom ModelSlim export.

Sequential layer placement across devices is a correctness bring-up path, not
vLLM integration or TP/EP execution. Equations follow the pinned Transformers
Muse implementation (d4804f9d), with native parity tests. Only contiguous,
unpadded text, default RoPE, sparse SiLU experts, and untied embeddings are
supported. CPU execution is a reference path; NPU experts use packed W4 kernels.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from reap.muse_ascend import w4_linear
from reap.muse_export import FORMAT, dequantize_weight

PREFIX = "model.language_model."


def configuration(raw):
    c = SimpleNamespace(**raw.get("text_config", raw))
    if (
        c.rope_parameters["rope_type"] != "default"
        or c.hidden_act != "silu"
        or c.tie_word_embeddings
        or getattr(c, "num_shared_experts", 0) != 0
        or any(t != "sparse" for t in c.mlp_layer_types)
    ):
        raise ValueError("unsupported Muse runtime configuration")
    if (
        len(c.layer_types) != c.num_hidden_layers
        or len(c.layer_rope_theta) != c.num_hidden_layers
        or len(c.mlp_layer_types) != c.num_hidden_layers
        or set(c.layer_types) - {"full_attention", "sliding_attention"}
        or any(
            t not in (0, c.rope_parameters["rope_theta"]) for t in c.layer_rope_theta
        )
        or c.num_attention_heads % c.num_key_value_heads
        or c.head_dim % 2
        or c.sliding_window < 1
    ):
        raise ValueError("invalid per-layer attention configuration")
    return c


def rms(x, eps, weight=None):
    value = x.float()
    value = value * torch.pow(value.square().mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        value = value * weight.float()
    return value.to(x.dtype)


def centered_rms(x, weight, eps, *, input_scaled=False):
    value = x.float()
    if input_scaled:
        value = (value * (1.0 + weight).float()).to(x.dtype).float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    if not input_scaled:
        value = value * (1.0 + weight).float()
    return value.to(x.dtype)


def residual_gate(residual, hidden, gate, temperature):
    w = gate / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return (alpha * residual + beta * hidden).to(residual.dtype)


def w8_linear(x, codes, scales, bias=None, row_chunk=512):
    """Bounded FP16 dequantization fallback; keeps resident attention weights W8."""
    if (
        codes.dtype != torch.int8
        or scales.shape != (codes.shape[0], 1)
        or row_chunk < 1
    ):
        raise ValueError("invalid W8 projection")
    flat = x.reshape(-1, x.shape[-1])
    out = torch.empty((flat.shape[0], codes.shape[0]), dtype=x.dtype, device=x.device)
    for first in range(0, codes.shape[0], row_chunk):
        last = first + row_chunk
        weight = (codes[first:last].float() * scales[first:last].float()).to(x.dtype)
        out[:, first:last] = F.linear(
            flat, weight, None if bias is None else bias[first:last]
        )
    return out.view(*x.shape[:-1], codes.shape[0])


@dataclass
class LayerCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    seen: int = 0


def checked_shard(root, filename, journal, *, verify=True):
    if Path(filename).name != filename:
        raise ValueError("invalid shard path")
    path = Path(root) / filename
    if verify:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != journal["shards"][filename]["sha256"]:
            raise ValueError(f"shard checksum differs: {filename}")
    values = load_file(path)
    expected = journal["shards"][filename]["tensors"]
    if values.keys() != expected.keys():
        raise ValueError("shard tensor names differ")
    for name, value in values.items():
        if (
            list(value.shape) != expected[name]["shape"]
            or str(value.dtype) != expected[name]["dtype"]
        ):
            raise ValueError(f"shard tensor schema differs: {name}")
    return values


class MuseLayer:
    def __init__(self, config, index, weights):
        self.c, self.index, self.w = config, index, weights
        self.device = weights["input_layernorm.weight"].device

    def linear(self, name, x, *, expert=False):
        weight = self.w[name + ".weight"]
        scale = self.w.get(name + ".weight_scale")
        bias = self.w.get(name + ".bias")
        if scale is None:
            return F.linear(x, weight, bias)
        if expert:
            if x.device.type == "cpu":
                return F.linear(x, dequantize_weight(weight, scale).to(x.dtype))
            shape = x.shape
            return w4_linear(x.reshape(-1, shape[-1]), weight, scale).view(
                *shape[:-1], weight.shape[0]
            )
        return w8_linear(x, weight, scale, bias)

    def norm(self, name, x, *, input_scaled=False):
        return centered_rms(
            x,
            self.w[name + ".weight"],
            self.c.post_norm_eps if input_scaled else self.c.rms_norm_eps,
            input_scaled=input_scaled,
        )

    def attention(self, x, start, cache=None):
        c = self.c
        count = x.shape[1]
        q, k, v = [
            self.linear("self_attn." + name + "_proj", x)
            .view(1, count, -1, c.head_dim)
            .transpose(1, 2)
            for name in ("q", "k", "v")
        ]
        q = rms(q, c.rms_norm_eps) * c.qk_scale_factor
        k = rms(k, c.rms_norm_eps)
        positions = torch.arange(start, start + count, device="cpu").to(x.device)
        if c.layer_rope_theta[self.index]:
            inv = 1.0 / (
                c.rope_parameters["rope_theta"]
                ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32) / c.head_dim)
            )
            freqs = positions[:, None].float() * inv.to(x.device)
            phase = torch.cat((freqs, freqs), dim=-1)
            cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)

            def rotate(t):
                half = t.shape[-1] // 2
                return (
                    t * cos + torch.cat((-t[..., half:], t[..., :half]), dim=-1) * sin
                )

            q, k = rotate(q), rotate(k)
        previous = 0
        if cache is not None:
            if cache.seen != start:
                raise ValueError("cache positions are not contiguous")
            if cache.key is not None:
                previous = cache.key.shape[-2]
                k = torch.cat((cache.key, k), dim=-2)
                v = torch.cat((cache.value, v), dim=-2)
        sliding = c.layer_types[self.index] == "sliding_attention"
        if cache is not None:
            keep = min(k.shape[-2], c.sliding_window - 1) if sliding else k.shape[-2]
            # clone avoids retaining the full concatenation via a small view.
            cache.key = k[..., -keep:, :].clone() if keep else None
            cache.value = v[..., -keep:, :].clone() if keep else None
            cache.seen = start + count
        key_positions = torch.arange(start - previous, start + count, device="cpu").to(
            x.device
        )
        allowed = key_positions[None, :] <= positions[:, None]
        if sliding:
            allowed = allowed & (
                key_positions[None, :] > positions[:, None] - c.sliding_window
            )
        mask = torch.zeros(allowed.shape, dtype=x.dtype, device=x.device).masked_fill(
            ~allowed, torch.finfo(x.dtype).min
        )
        repeats = c.num_attention_heads // c.num_key_value_heads
        k, v = [t.repeat_interleave(repeats, dim=1) for t in (k, v)]
        scores = torch.matmul(q, k.transpose(-1, -2)) * c.head_dim**-0.5
        probabilities = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attended = torch.matmul(probabilities, v).transpose(1, 2).contiguous()
        attended = rms(attended, c.rms_norm_eps).reshape(1, count, -1)
        attended = attended * torch.sigmoid(self.linear("self_attn.gate_proj", x))
        return self.linear("self_attn.o_proj", attended)

    def moe(self, x):
        c = self.c
        flat = x.reshape(-1, x.shape[-1])
        scores = torch.sigmoid(
            F.linear(flat.float(), self.w["mlp.gate.weight"].float())
            * c.output_multiplier
        )
        ids = torch.topk(
            scores + self.w["mlp.gate.e_score_correction_bias"],
            c.num_experts_per_tok,
            dim=-1,
            sorted=False,
        ).indices
        coefficients = scores.gather(1, ids)
        coefficients = (
            coefficients
            / (coefficients.sum(-1, keepdim=True) + 1e-15)
            * c.routed_scaling_factor
        )
        latent = self.norm(
            "mlp.pre_expert_norm", self.linear("mlp.pre_expert_proj", flat)
        )
        mixed = torch.zeros_like(latent)
        # CPU routing indices avoid an NPU host synchronization for every unused expert.
        host_ids = ids.cpu()
        for expert in sorted(set(host_ids.flatten().tolist())):
            tokens, slots = torch.where(host_ids == expert)
            tokens, slots = tokens.to(x.device), slots.to(x.device)
            current = latent[tokens]
            prefix = f"mlp.experts.{expert}."
            gate = self.linear(prefix + "gate_proj", current, expert=True)
            up = self.linear(prefix + "up_proj", current, expert=True)
            down = self.linear(prefix + "down_proj", F.silu(gate) * up, expert=True)
            down = self.norm("mlp.experts.post_expert_norm", down, input_scaled=True)
            down = (down * coefficients[tokens, slots, None]).to(mixed.dtype)
            mixed.index_add_(0, tokens, down)
        return self.linear("mlp.post_expert_proj", mixed).view_as(x)

    @torch.inference_mode()
    def __call__(self, x, *, start=0, cache=None):
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or start < 0:
            raise ValueError("runtime supports one contiguous unpadded sequence")
        c = self.c
        h = self.attention(self.norm("input_layernorm", x), start, cache)
        x = residual_gate(
            x,
            rms(h, c.post_norm_eps),
            self.w["post_attention_residual_gate.gate"],
            c.residual_gate_temperature,
        )
        h = self.moe(self.norm("pre_feedforward_layernorm", x))
        return residual_gate(
            x,
            self.norm("post_feedforward_layernorm", h, input_scaled=True),
            self.w["post_feedforward_residual_gate.gate"],
            c.residual_gate_temperature,
        )


def load_layer(root, index, config, device, *, journal=None, verify=True):
    root = Path(root)
    journal = journal or json.loads((root / "export-progress.json").read_text())
    prefix = PREFIX + f"layers.{index}."
    values = {}
    for filename in (
        f"dense-{index:03d}.safetensors",
        f"experts-{index:03d}-ep0.safetensors",
        f"experts-{index:03d}-ep1.safetensors",
    ):
        shard = checked_shard(root, filename, journal, verify=verify)
        for name, tensor in shard.items():
            if not name.startswith(prefix) or name[len(prefix) :] in values:
                raise ValueError("unexpected layer tensor")
            values[name[len(prefix) :]] = tensor.to(device)
        del shard
    return MuseLayer(config, index, values)


def placement_plan(root, devices, max_context=8192, reserve_gib=3, *, optimized=False):
    """Minimax contiguous layer placement, including globals and bounded KV."""
    root = Path(root)
    journal = json.loads((root / "export-progress.json").read_text())
    if not journal["complete"]:
        raise ValueError("full export must complete before planning resident execution")
    c = configuration(json.loads((root / "config.json").read_text()))

    # Planning is also supported on CPU-only hosts without torch_npu registered.
    def canonical(device):
        value = str(device)
        if value.startswith("npu:") and value[4:].isdigit():
            return "npu:" + str(int(value[4:]))
        return str(torch.device(device))

    accelerated = [canonical(d) for d in devices if str(d) != "cpu"]
    if len(set(accelerated)) != len(accelerated):
        raise ValueError("each accelerator must appear only once in the stage plan")
    if (
        not devices
        or len(devices) > c.num_hidden_layers
        or max_context < 1
        or reserve_gib < 0
    ):
        raise ValueError("invalid placement request")
    costs = []
    for i in range(c.num_hidden_layers):
        weight = sum(
            journal["shards"][f]["bytes"]
            for f in (
                f"dense-{i:03d}.safetensors",
                f"experts-{i:03d}-ep0.safetensors",
                f"experts-{i:03d}-ep1.safetensors",
            )
        )
        if optimized:
            for filename in (
                f"dense-{i:03d}.safetensors",
                f"experts-{i:03d}-ep0.safetensors",
                f"experts-{i:03d}-ep1.safetensors",
            ):
                tensors = journal["shards"][filename]["tensors"]
                for name, receipt in tensors.items():
                    if not name.endswith(".weight_scale"):
                        continue
                    if ".mlp.experts." in name:
                        # Tiled kernels retain INT8 offsets alongside FP16 scales.
                        weight += receipt["bytes"] // 2
                    else:
                        # W8 codes/scales are replaced by same-dequantized FP16.
                        weight += (
                            tensors[name.removesuffix("_scale")]["bytes"]
                            - receipt["bytes"]
                        )
        tokens = (
            min(max_context, c.sliding_window - 1)
            if c.layer_types[i] == "sliding_attention"
            else max_context
        )
        costs.append(weight + tokens * 2 * c.num_key_value_heads * c.head_dim * 2)
    globals_ = {
        name: receipt["bytes"]
        for f, shard in journal["shards"].items()
        for name, receipt in shard["tensors"].items()
        if f.startswith("global-")
    }
    first = globals_[PREFIX + "embed_tokens.weight"]
    last = sum(globals_.values()) - first
    n = c.num_hidden_layers
    # dp[stages, layers] holds the smallest achievable maximum resident bytes.
    dp = {(0, 0): (0, [])}
    for stage in range(len(devices)):
        for end in range(stage + 1, n + 1):
            candidates = []
            for begin in range(stage, end):
                if (stage, begin) not in dp:
                    continue
                cost = (
                    sum(costs[begin:end])
                    + (first if stage == 0 else 0)
                    + (last if stage == len(devices) - 1 else 0)
                )
                peak, ranges = dp[stage, begin]
                candidates.append((max(peak, cost), ranges + [(begin, end, cost)]))
            if candidates:
                dp[stage + 1, end] = min(candidates, key=lambda item: item[0])
    ranges = dp[len(devices), n][1]
    return [
        {
            "device": device,
            "first_layer": begin,
            "last_layer_exclusive": end,
            "weights_and_kv_bytes": cost,
            "required_bytes": cost + int(reserve_gib * 2**30),
        }
        for device, (begin, end, cost) in zip(devices, ranges)
    ]


class MuseRuntime:
    def __init__(
        self,
        root,
        devices=("cpu",),
        *,
        max_context=8192,
        prefill_chunk=64,
        reserve_gib=3,
        verify=True,
        progress=print,
        optimized=False,
    ):
        self.root = Path(root)
        self.progress = progress
        self.generation_stats = {}
        self.journal = json.loads((self.root / "export-progress.json").read_text())
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        if not self.journal["complete"] or index["metadata"]["format"] != FORMAT:
            raise ValueError("requires a complete supported export")
        expected = {
            name: filename
            for filename, record in self.journal["shards"].items()
            for name in record["tensors"]
        }
        if index["weight_map"] != expected:
            raise ValueError("index differs from export receipts")
        self.c = configuration(json.loads((self.root / "config.json").read_text()))
        if self.c.num_local_experts != self.journal["identity"]["plan"]["keep"]:
            raise ValueError("config expert count differs from export plan")
        if (
            not 1 <= prefill_chunk <= 128
            or not 1 <= max_context <= self.c.max_position_embeddings
        ):
            raise ValueError("invalid context or prefill size")
        self.max_context, self.prefill_chunk = max_context, prefill_chunk
        self.optimized = optimized
        if optimized and any(not str(d).startswith("npu:") for d in devices):
            raise ValueError("optimized execution requires explicit NPU devices")
        self.plan = placement_plan(
            root, devices, max_context, reserve_gib, optimized=optimized
        )
        loader = load_layer
        if optimized:
            from reap.muse_fast import load_fast_layer

            loader = load_fast_layer
            for name in (
                "npu_qwen_w4_group_matmul_310",
                "npu_qwen_w4_routed_matmul_310",
            ):
                if not hasattr(torch.ops._C_ascend, name):
                    raise RuntimeError(
                        f"optimized execution requires custom operator {name}"
                    )
        for item in self.plan:
            device = torch.device(item["device"])
            if device.type == "npu":
                with torch.npu.device(device):
                    free, _ = torch.npu.mem_get_info()
                if free < item["required_bytes"]:
                    raise RuntimeError(
                        f"insufficient free memory on {device}: need {item['required_bytes'] / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB"
                    )
        self.layers, self.globals = [], {}
        for item in self.plan:
            for i in range(item["first_layer"], item["last_layer_exclusive"]):
                self.layers.append(
                    loader(
                        root,
                        i,
                        self.c,
                        item["device"],
                        journal=self.journal,
                        verify=verify,
                    )
                )
                progress(
                    json.dumps(
                        {"event": "loaded_layer", "layer": i, "device": item["device"]}
                    )
                )
        for filename in self.journal["shards"]:
            if not filename.startswith("global-"):
                continue
            for name, tensor in checked_shard(
                root, filename, self.journal, verify=verify
            ).items():
                device = (
                    devices[0]
                    if name == PREFIX + "embed_tokens.weight"
                    else devices[-1]
                )
                if optimized and name == "lm_head.weight":
                    from reap.muse_fast import native_weight

                    self.globals[name] = native_weight(tensor, device)
                else:
                    self.globals[name] = tensor.to(device)
        self.reset()

    def reset(self):
        self.cache = [LayerCache() for _ in self.layers]
        self.seen = 0

    @torch.inference_mode()
    def forward(self, ids, *, all_logits=False):
        ids = torch.as_tensor(ids, dtype=torch.long, device="cpu")
        if (
            ids.ndim != 1
            or not 1 <= len(ids) <= self.prefill_chunk
            or self.seen + len(ids) > self.max_context
        ):
            raise ValueError("invalid token chunk or context exhausted")
        if int(ids.min()) < 0 or int(ids.max()) >= self.c.vocab_size:
            raise ValueError("token outside vocabulary")
        embedding = self.globals[PREFIX + "embed_tokens.weight"]
        h = rms(
            F.embedding(ids.to(embedding.device), embedding).unsqueeze(0),
            self.c.rms_norm_eps,
        )
        for layer, cache in zip(self.layers, self.cache):
            if h.device != layer.device:
                h = h.cpu().to(
                    layer.device
                )  # Explicit portable stage transport; no P2P assumption.
            h = layer(h, start=self.seen, cache=cache)
        h = rms(h, self.c.rms_norm_eps, self.globals[PREFIX + "norm.weight"])
        if not all_logits:
            h = h[:, -1:]
        logits = F.linear(h, self.globals["lm_head.weight"]) * self.c.output_multiplier
        logits = (
            torch.tanh(logits / self.c.final_logit_softcapping)
            * self.c.final_logit_softcapping
        )
        logits = logits.cpu()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite Muse logits; reset the request cache")
        self.seen += len(ids)
        return logits

    def generate(self, ids, max_new_tokens=32, eos=()):
        if (
            not ids
            or max_new_tokens < 1
            or len(ids) + max_new_tokens > self.max_context
        ):
            raise ValueError("prompt and generation exceed context or are empty")
        self.reset()
        started = time.monotonic()
        self.progress(
            json.dumps(
                {
                    "event": "generation_start",
                    "prompt_tokens": len(ids),
                    "max_new_tokens": max_new_tokens,
                }
            )
        )
        for start in range(0, len(ids), self.prefill_chunk):
            logits = self.forward(ids[start : start + self.prefill_chunk])
            self.progress(
                json.dumps(
                    {
                        "event": "prefill_progress",
                        "tokens": min(start + self.prefill_chunk, len(ids)),
                        "seconds": time.monotonic() - started,
                    }
                )
            )
        prefill_seconds = time.monotonic() - started
        result = []
        for step in range(max_new_tokens):
            token = int(logits[0, -1].argmax())
            result.append(token)
            if len(result) % 16 == 0:
                self.progress(
                    json.dumps(
                        {
                            "event": "decode_progress",
                            "tokens": len(result),
                            "seconds": time.monotonic() - started,
                        }
                    )
                )
            if token in eos or step == max_new_tokens - 1:
                break
            logits = self.forward([token])
        total_seconds = time.monotonic() - started
        self.generation_stats = {
            "prompt_tokens": len(ids),
            "completion_tokens": len(result),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": total_seconds - prefill_seconds,
            "total_seconds": total_seconds,
        }
        self.progress(
            json.dumps({"event": "generation_complete", **self.generation_stats})
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--reserve-gib", type=float, default=3)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.plan_only:
        print(
            json.dumps(
                placement_plan(
                    args.model,
                    args.devices,
                    args.max_context,
                    args.reserve_gib,
                    optimized=args.optimized,
                ),
                indent=2,
            )
        )
        return
    if any(d.startswith("npu:") for d in args.devices):
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        torch.npu.matmul.allow_hf32 = False
        if args.library is None:
            parser.error("NPU execution requires --library")
        torch.ops.load_library(str(args.library))
    if not args.input_ids:
        parser.error("generation requires --input-ids")
    runtime = MuseRuntime(
        args.model,
        args.devices,
        max_context=args.max_context,
        prefill_chunk=args.prefill_chunk,
        reserve_gib=args.reserve_gib,
        optimized=args.optimized,
    )
    generation = (
        json.loads((args.model / "generation_config.json").read_text())
        if (args.model / "generation_config.json").exists()
        else {}
    )
    eos = generation.get("eos_token_id", runtime.c.eos_token_id)
    eos = [eos] if isinstance(eos, int) else eos
    print(
        json.dumps(
            {
                "generated_ids": runtime.generate(
                    args.input_ids, args.max_new_tokens, eos=eos
                )
            }
        )
    )


if __name__ == "__main__":
    main()
