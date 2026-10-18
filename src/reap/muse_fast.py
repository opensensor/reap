"""Lossless load-time W4 tiling and device-routed Muse experts for Ascend 310P.

The physical tile ABI follows vllm-ascend's Qwen 310P pack_cube_tiles: low/high
nibbles represent channels n/n+16, biased by eight, in Cube NZ order. Exported
files are unchanged. Split-K accumulation preserves the reference runtime's
FP16 partial products and FP32 sum. No activation quantization is introduced.
"""

import torch
import torch.nn.functional as F

from reap.muse_runtime import MuseLayer, load_layer


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

    def linear(self, x, *, expert=None, ids=None):
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

    def attention(self, x, start, cache=None):
        if not getattr(self, "flash_attention", False):
            return super().attention(x, start, cache)
        from reap.muse_attention import flash_attention

        q, k, v, _ = self.attention_inputs(x, start)
        attended = flash_attention(
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
            # This installed routed kernel is optimized for decode. Reusing an
            # expert's weight tile across its prefill tokens is much faster.
            mixed = torch.zeros_like(latent)
            host_ids = ids.cpu()
            for expert in sorted(set(host_ids.flatten().tolist())):
                tokens, slots = torch.where(host_ids == expert)
                tokens, slots = tokens.to(x.device), slots.to(x.device)
                gate, up = self.gate_up.linear(latent[tokens], expert=expert).chunk(
                    2, dim=-1
                )
                down = self.down.linear(F.silu(gate) * up, expert=expert)
                down = self.norm(
                    "mlp.experts.post_expert_norm", down, input_scaled=True
                )
                down = (down * coefficients[tokens, slots, None]).to(mixed.dtype)
                mixed.index_add_(0, tokens, down)
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
            gate, up = self.gate_up.linear(routes, ids=route_ids).chunk(2, dim=-1)
            down = self.down.linear(F.silu(gate) * up, ids=route_ids)
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


def load_fast_layer(root, index, config, device, *, journal=None, verify=True):
    """Convert on CPU, then transfer once; never retain duplicate NPU banks."""
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
