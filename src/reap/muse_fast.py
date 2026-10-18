"""Lossless load-time W4 tiling and device-routed Muse experts for Ascend 310P.

The physical tile ABI follows vllm-ascend's Qwen 310P pack_cube_tiles: low/high
nibbles represent channels n/n+16, biased by eight, in Cube NZ order. Exported
files are unchanged. Split-K accumulation preserves the reference runtime's
FP16 partial products and FP32 sum. No activation quantization is introduced.
"""

import torch
import torch.nn.functional as F

from reap.muse_runtime import MuseLayer, load_layer, rms


class StepPositions:
    """Share identical rotary phases across a stage for the current token chunk."""

    def __init__(self):
        self.key = None
        self.inv = None

    def get(self, config, x, start, count):
        theta = config.rope_parameters["rope_theta"]
        geometry = (x.device, config.head_dim, theta)
        key = (*geometry, x.dtype, start, count)
        if key != self.key:
            if self.inv is None or geometry != self.key[:3]:
                self.inv = (
                    1.0
                    / (
                        theta
                        ** (
                            torch.arange(0, config.head_dim, 2).float()
                            / config.head_dim
                        )
                    )
                ).to(x.device)
            positions = torch.arange(start, start + count, device="cpu").to(x.device)
            freqs = positions[:, None].float() * self.inv
            phase = torch.cat((freqs, freqs), dim=-1)
            self.values = positions, phase.cos().to(x.dtype), phase.sin().to(x.dtype)
            self.key = key
        return self.values


def pack_tiles(value, *, scales=False):
    """Encode a canonical matrix in the installed operator's lossless tile ABI."""
    if value.ndim != 2 or value.shape[0] % 32:
        raise ValueError("tile packing requires N divisible by 32")
    rows = value.shape[0] // 32
    if scales:
        return value.reshape(rows, 32, -1).transpose(1, 2).contiguous().view_as(value)
    if value.dtype != torch.int8 or value.shape[1] % 64:
        raise ValueError("tile codes require INT8 and K divisible by 128")
    nibbles = torch.stack((value & 15, (value >> 4) & 15), -1).flatten(-2) ^ 8
    planes = nibbles.reshape(rows, 2, 16, -1, 16).permute(0, 3, 4, 1, 2)
    packed = planes[..., 0, :] | (planes[..., 1, :] << 4)
    return packed.contiguous().view_as(value)


class TiledBank:
    def __init__(self, parts, inputs, outputs):
        self.parts, self.inputs, self.outputs = parts, inputs, outputs
        self.backend = None

    @classmethod
    def from_weights(cls, weights, experts, projections, device):
        key = "mlp.experts.0." + projections[0]
        inputs = weights[key + ".weight"].shape[1] * 2
        outputs = sum(
            weights["mlp.experts.0." + p + ".weight"].shape[0] for p in projections
        )
        if experts < 1 or inputs < 256 or inputs % 128 or outputs % 128:
            raise ValueError("unsupported tiled expert geometry")
        if inputs % 2048 and inputs > 2048:
            raise ValueError(
                "optimized experts require K <= 2048 or a multiple of 2048"
            )
        parts = []
        for first in range(0, inputs, 2048):
            last = min(first + 2048, inputs)
            codes, scales = [], []
            for expert in range(experts):
                prefix = f"mlp.experts.{expert}."
                code = torch.cat(
                    [
                        weights[prefix + p + ".weight"][:, first // 2 : last // 2]
                        for p in projections
                    ]
                )
                scale = torch.cat(
                    [
                        weights[prefix + p + ".weight_scale"][
                            :, first // 128 : last // 128
                        ]
                        for p in projections
                    ]
                )
                codes.append(pack_tiles(code))
                scales.append(pack_tiles(scale, scales=True))
            q = torch.stack(codes).to(device)
            s = torch.stack(scales).to(device)
            # The unsigned tiled encoding shifts symmetric zero to eight.
            offset = torch.full_like(s, 8, dtype=torch.int8)
            parts.append((q, s, offset))
        return cls(parts, inputs, outputs)

    def linear(self, x, *, expert=None, ids=None, ends=None):
        if self.backend is not None and expert is None:
            return self.backend.linear(self, x, ids=ids, ends=ends)
        if ends is not None:
            raise ValueError("grouped routes require a native backend")
        if (expert is None) == (ids is None):
            raise ValueError("specify an expert or routed IDs")
        if x.ndim != 2 or x.shape[1] != self.inputs:
            raise ValueError("invalid tiled input geometry")
        chunk = 80 if ids is not None else 128
        outputs = []
        for row in range(0, x.shape[0], chunk):
            accumulator = None
            for part, (q, scale, offset) in enumerate(self.parts):
                current = x[
                    row : row + chunk, part * 2048 : part * 2048 + q.shape[-1] * 2
                ].contiguous()
                if ids is None:
                    value = torch.ops._C_ascend.npu_qwen_w4_group_matmul_310(
                        current, q[expert], scale[expert], offset[expert], True
                    )
                else:
                    value = torch.ops._C_ascend.npu_qwen_w4_routed_matmul_310(
                        current, q, scale, offset, ids[row : row + chunk].contiguous()
                    )
                value = value.float()
                accumulator = value if accumulator is None else accumulator + value
            outputs.append(accumulator.to(x.dtype))
        return torch.cat(outputs, dim=0)


class FastMuseLayer(MuseLayer):
    def __init__(self, config, index, weights, gate_up, down):
        super().__init__(config, index, weights)
        self.gate_up, self.down = gate_up, down
        self.positions = StepPositions()
        self.norm_weights = {
            name.removesuffix(".weight"): 1.0 + value
            for name, value in weights.items()
            if name.endswith("norm.weight")
        }
        self.unit_weights = {}
        self.residual_coefficients = {}
        for name in ("post_attention", "post_feedforward"):
            gate = weights.get(name + "_residual_gate.gate")
            if gate is not None:
                w = gate / config.residual_gate_temperature
                beta = torch.sigmoid(w)
                alpha = torch.sqrt(
                    torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)
                )
                self.residual_coefficients[name] = alpha, beta

    def fast_rms(self, x, eps, weight=None):
        # The installed 310P RMS kernel produces invalid/stale values for
        # sub-block rows (observed at width eight). Only use aligned FP16 rows;
        # small synthetic heads and other dtypes retain the reference path.
        if x.device.type != "npu" or x.dtype != torch.float16 or x.shape[-1] % 16:
            return rms(x, eps, weight)
        import torch_npu

        if weight is None:
            key = x.shape[-1], x.dtype
            if key not in self.unit_weights:
                self.unit_weights[key] = torch.ones(
                    x.shape[-1], dtype=x.dtype, device=x.device
                )
            weight = self.unit_weights[key]
        return torch_npu.npu_rms_norm(x.contiguous(), weight, eps)[0]

    def norm(self, name, x, *, input_scaled=False):
        weight = self.norm_weights[name]
        if input_scaled:
            # Preserve the model's FP16 rounding *before* the RMS reduction.
            x = (x.float() * weight.float()).to(x.dtype)
            return self.fast_rms(x, self.c.post_norm_eps)
        return self.fast_rms(x, self.c.rms_norm_eps, weight)

    def attention_inputs(self, x, start):
        c, count = self.c, x.shape[1]
        q, k, v = [
            self.linear("self_attn." + name + "_proj", x)
            .view(1, count, -1, c.head_dim)
            .transpose(1, 2)
            for name in ("q", "k", "v")
        ]
        q = self.fast_rms(q, c.rms_norm_eps) * c.qk_scale_factor
        k = self.fast_rms(k, c.rms_norm_eps)
        positions, cos, sin = self.positions.get(c, x, start, count)
        if c.layer_rope_theta[self.index]:

            def rotate(t):
                half = t.shape[-1] // 2
                return (
                    t * cos + torch.cat((-t[..., half:], t[..., :half]), dim=-1) * sin
                )

            q, k = rotate(q), rotate(k)
        return q, k, v, positions

    def attention_output(self, attended, x):
        attended = attended.transpose(1, 2).contiguous()
        attended = self.fast_rms(attended, self.c.rms_norm_eps).reshape(
            1, x.shape[1], -1
        )
        attended = attended * torch.sigmoid(self.linear("self_attn.gate_proj", x))
        return self.linear("self_attn.o_proj", attended)

    def __call__(self, x, start=0, cache=None):
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or start < 0:
            raise ValueError("runtime supports one contiguous unpadded sequence")
        h = self.attention(self.norm("input_layernorm", x), start, cache)
        alpha, beta = self.residual_coefficients["post_attention"]
        x = (alpha * x + beta * self.fast_rms(h, self.c.post_norm_eps)).to(x.dtype)
        h = self.moe(self.norm("pre_feedforward_layernorm", x))
        alpha, beta = self.residual_coefficients["post_feedforward"]
        return (
            alpha * x
            + beta * self.norm("post_feedforward_layernorm", h, input_scaled=True)
        ).to(x.dtype)

    def attention(self, x, start, cache=None):
        if not getattr(self, "bounded_attention", False):
            return super().attention(x, start, cache)
        from reap.muse_attention import bounded_attention

        q, k, v, _ = self.attention_inputs(x, start)
        attended = bounded_attention(
            q,
            k,
            v,
            start=start,
            cache=cache,
            window=self.c.sliding_window
            if self.c.layer_types[self.index] == "sliding_attention"
            else None,
            max_context=self.max_context,
            prefill_chunk=self.prefill_chunk,
        )
        return self.attention_output(attended, x)

    def expert_projection(self, routes, *, ids=None, ends=None):
        if hasattr(self, "tensor_experts"):
            return self.tensor_experts.project(routes, ids=ids, ends=ends)
        metadata = {"ends": ends} if ends is not None else {"ids": ids}
        gate, up = self.gate_up.linear(routes, **metadata).chunk(2, dim=-1)
        return self.down.linear(F.silu(gate) * up, **metadata)

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
        # Preserve the reference's ascending expert accumulation order for each
        # token, including its FP16 rounding after each weighted contribution.
        ids, order = ids.sort(dim=-1)
        coefficients = coefficients.gather(1, order)
        latent = self.norm(
            "mlp.pre_expert_norm", self.linear("mlp.pre_expert_proj", flat)
        )
        if len(flat) > 1:
            # Group once, transfer route metadata once, and batch the shared
            # normalization/combine epilogue. The expert matmuls still reuse
            # weights across tokens; no activation quantization is introduced.
            host_ids = ids.flatten().cpu()
            permutation = host_ids.argsort(stable=True)
            experts, counts = host_ids[permutation].unique_consecutive(
                return_counts=True
            )
            rows = (permutation // c.num_experts_per_tok).to(x.device)
            routes = latent[rows]
            if self.gate_up.backend is not None or hasattr(self, "tensor_experts"):
                ends = host_ids.bincount(minlength=c.num_local_experts).cumsum(0)
                ends = ends.to(x.device)
                projected = self.expert_projection(routes, ends=ends)
            else:
                outputs = []
                first = 0
                for expert, count in zip(experts.tolist(), counts.tolist()):
                    gate, up = self.gate_up.linear(
                        routes[first : first + count], expert=expert
                    ).chunk(2, dim=-1)
                    outputs.append(self.down.linear(F.silu(gate) * up, expert=expert))
                    first += count
                projected = torch.cat(outputs)
            down = self.norm(
                "mlp.experts.post_expert_norm", projected, input_scaled=True
            )
            # Restore token-major, ascending-expert order before the FP16 sum.
            ordered = torch.empty_like(down)
            ordered.index_copy_(0, permutation.to(x.device), down)
            contributions = (
                ordered.view(len(flat), c.num_experts_per_tok, -1)
                * coefficients[:, :, None]
            ).to(latent.dtype)
            mixed = torch.zeros_like(latent)
            for slot in range(c.num_experts_per_tok):
                mixed = mixed + contributions[:, slot]
            return self.linear("mlp.post_expert_proj", mixed).view_as(x)
        mixed = []
        tokens_per_chunk = 80 // c.num_experts_per_tok
        if tokens_per_chunk < 1:
            raise ValueError("top-k exceeds routed operator capacity")
        for first in range(0, len(flat), tokens_per_chunk):
            current = latent[first : first + tokens_per_chunk]
            routes = (
                current[:, None, :]
                .expand(-1, c.num_experts_per_tok, -1)
                .reshape(-1, current.shape[-1])
            )
            route_ids = ids[first : first + tokens_per_chunk].flatten().to(torch.int32)
            down = self.expert_projection(routes, ids=route_ids)
            down = self.norm("mlp.experts.post_expert_norm", down, input_scaled=True)
            down = down.reshape(len(current), c.num_experts_per_tok, -1)
            contributions = (
                down * coefficients[first : first + len(current), :, None]
            ).to(current.dtype)
            total = torch.zeros_like(current)
            for slot in range(c.num_experts_per_tok):
                total = total + contributions[:, slot]
            mixed.append(total)
        return self.linear("mlp.post_expert_proj", torch.cat(mixed)).view_as(x)


def native_weight(value, device):
    """Retain FP16 matrices in NZ; keep activations/custom-kernel inputs in ND."""
    value = value.to(device)
    if value.device.type == "npu" and value.dtype == torch.float16 and value.ndim == 2:
        import torch_npu

        # Only used during single-threaded model loading. The runtime keeps
        # implicit format selection disabled for its packed custom operators.
        torch.npu.config.allow_internal_format = True
        try:
            value = torch_npu.npu_format_cast(value, 29)
        finally:
            torch.npu.config.allow_internal_format = False
    return value


def load_fast_layer(
    root, index, config, device, *, journal=None, verify=True, packed_cache=None
):
    """Convert on CPU, then transfer once; never retain duplicate NPU banks."""
    if packed_cache is not None:
        from reap.muse_packed import load_cached_layer

        return load_cached_layer(
            root, index, config, device, packed_cache, journal=journal, verify=verify
        )
    original = load_layer(root, index, config, "cpu", journal=journal, verify=verify)
    values = original.w
    gate_up = TiledBank.from_weights(
        values, config.num_local_experts, ("gate_proj", "up_proj"), device
    )
    down = TiledBank.from_weights(
        values, config.num_local_experts, ("down_proj",), device
    )
    dense = {}
    for name, value in values.items():
        if name.startswith("mlp.experts.") and not name.startswith(
            "mlp.experts.post_expert_norm."
        ):
            continue
        if name.endswith(".weight_scale"):
            continue
        scale = values.get(name + "_scale")
        if scale is not None:
            value = (value.float() * scale.float()).half()
        dense[name] = native_weight(value, device)
    return FastMuseLayer(config, index, dense, gate_up, down)
