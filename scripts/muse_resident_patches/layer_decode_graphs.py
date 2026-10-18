"""Capture decode normalization and residuals around dynamic attention and TP."""

import torch
import torch.nn.functional as F
from reap.muse_fast import FastMuseLayer, TiledBank
from reap.muse_tensor import DecodeGraph

original_moe = getattr(FastMuseLayer.moe, "_muse_original", FastMuseLayer.moe)


def prepare(self, x, unused):
    c = self.c
    flat = x.reshape(-1, x.shape[-1])
    scores = torch.sigmoid(
        F.linear(flat.float(), self.w["mlp.gate.weight"].float()) * c.output_multiplier
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
    ids, order = ids.sort(dim=-1)
    coefficients = coefficients.gather(1, order)
    latent = self.norm("mlp.pre_expert_norm", self.linear("mlp.pre_expert_proj", flat))
    routes = (
        latent[:, None, :]
        .expand(-1, c.num_experts_per_tok, -1)
        .reshape(-1, latent.shape[-1])
    )
    return routes, ids.flatten().to(torch.int32), coefficients


def finish(self, projected, coefficients):
    down = self.norm("mlp.experts.post_expert_norm", projected, input_scaled=True)
    down = down.reshape(1, self.c.num_experts_per_tok, -1)
    contributions = (down * coefficients[:, :, None]).to(projected.dtype)
    total = torch.zeros(
        (1, projected.shape[-1]), dtype=projected.dtype, device=projected.device
    )
    for slot in range(self.c.num_experts_per_tok):
        total = total + contributions[:, slot]
    return self.linear("mlp.post_expert_proj", total).view(1, 1, -1)


def graph_moe(self, x):
    if x.shape[1] != 1 or not hasattr(self, "tensor_experts"):
        return original_moe(self, x)
    graphs = self.tensor_experts.graphs
    if "owner-prepare" not in graphs:
        dummy = torch.empty(0, dtype=torch.int32, device=x.device)
        graphs["owner-prepare"] = DecodeGraph.capture(
            lambda a, b: prepare(self, a, b), x, dummy
        )
    prep = graphs["owner-prepare"]
    routes, ids, coefficients = prep.run(x, prep.routes)
    projected = self.tensor_experts.project(routes, ids=ids)
    if "owner-finish" not in graphs:
        graphs["owner-finish"] = DecodeGraph.capture(
            lambda a, b: finish(self, a, b), projected, coefficients
        )
    return graphs["owner-finish"].run(projected, coefficients)


# Also works with the initially deployed tensor runtime, whose graph container
# predates explicit retention of native warmup metadata outside the graph pool.
original_linear = getattr(TiledBank.linear, "_muse_original", TiledBank.linear)
original_attention_inputs = getattr(
    FastMuseLayer.attention_inputs, "_muse_original", FastMuseLayer.attention_inputs
)
original_attention_output = getattr(
    FastMuseLayer.attention_output, "_muse_original", FastMuseLayer.attention_output
)


def retained_linear(self, x, *, expert=None, ids=None, ends=None):
    result = original_linear(self, x, expert=expert, ids=ids, ends=ends)
    if self.backend is not None and ids is not None and len(x) == 8:
        if not hasattr(self, "_decode_metadata"):
            self._decode_metadata = {}
        for key, meta in self.backend.geometry.items():
            if (
                key[0] == x.device
                and key[1] == 8
                and key[2] == self.outputs
                and key[-1] == 0
            ):
                self._decode_metadata[meta.data_ptr()] = meta
    return result


def attention_qkv(self, x, phase):
    c = self.c
    q, k, v = [
        self.linear("self_attn." + name + "_proj", x)
        .view(1, 1, -1, c.head_dim)
        .transpose(1, 2)
        for name in ("q", "k", "v")
    ]
    q = self.fast_rms(q, c.rms_norm_eps) * c.qk_scale_factor
    k = self.fast_rms(k, c.rms_norm_eps)
    if c.layer_rope_theta[self.index]:
        cos, sin = phase[0], phase[1]

        def rotate(t):
            half = t.shape[-1] // 2
            return t * cos + torch.cat((-t[..., half:], t[..., :half]), dim=-1) * sin

        q, k = rotate(q), rotate(k)
    return q, k, v


def graph_attention_inputs(self, x, start):
    if x.shape[1] != 1 or not hasattr(self, "tensor_experts"):
        return original_attention_inputs(self, x, start)
    positions, cos, sin = self.positions.get(self.c, x, start, 1)
    phase = torch.stack((cos, sin))
    graphs = self.tensor_experts.graphs
    if "attention-inputs" not in graphs:
        graphs["attention-inputs"] = DecodeGraph.capture(
            lambda a, b: attention_qkv(self, a, b), x, phase
        )
    q, k, v = graphs["attention-inputs"].run(x, phase)
    return q, k, v, positions


def graph_attention_output(self, attended, x):
    if x.shape[1] != 1 or not hasattr(self, "tensor_experts"):
        return original_attention_output(self, attended, x)
    graphs = self.tensor_experts.graphs
    if "attention-output" not in graphs:
        graphs["attention-output"] = DecodeGraph.capture(
            lambda a, b: original_attention_output(self, a, b), attended, x
        )
    return graphs["attention-output"].run(attended, x)


# Reapplying this candidate must wrap the same original implementation, not
# nest graph capture through the previous generation's wrapper.
for function, original in (
    (graph_moe, original_moe),
    (retained_linear, original_linear),
    (graph_attention_inputs, original_attention_inputs),
    (graph_attention_output, original_attention_output),
):
    function._muse_original = original


original_call = getattr(
    FastMuseLayer.__call__, "_muse_original", FastMuseLayer.__call__
)


def fused_layer(self, x, start=0, cache=None):
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != 1 or start < 0:
        return original_call(self, x, start, cache)
    if not hasattr(self, "tensor_experts") or not getattr(
        self, "bounded_attention", False
    ):
        return original_call(self, x, start, cache)
    from reap.muse_attention import bounded_attention

    graphs = self.tensor_experts.graphs
    _, cos, sin = self.positions.get(self.c, x, start, 1)
    phase = torch.stack((cos, sin))

    def inputs(value, phase):
        normalized = self.norm("input_layernorm", value)
        return normalized, *attention_qkv(self, normalized, phase)

    if "layer-inputs" not in graphs:
        graphs["layer-inputs"] = DecodeGraph.capture(inputs, x, phase)
    normalized, q, k, v = graphs["layer-inputs"].run(x, phase)
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

    def middle(attended, residual):
        h = original_attention_output(self, attended, normalized)
        alpha, beta = self.residual_coefficients["post_attention"]
        residual = (
            alpha * residual + beta * self.fast_rms(h, self.c.post_norm_eps)
        ).to(residual.dtype)
        routed, ids, coefficients = prepare(
            self, self.norm("pre_feedforward_layernorm", residual), None
        )
        return residual, routed, ids, coefficients

    if "layer-middle" not in graphs:
        graphs["layer-middle"] = DecodeGraph.capture(middle, attended, x)
    residual, routed, ids, coefficients = graphs["layer-middle"].run(attended, x)
    projected = self.tensor_experts.project(routed, ids=ids)

    def output(projected, coefficients):
        h = finish(self, projected, coefficients)
        alpha, beta = self.residual_coefficients["post_feedforward"]
        return (
            alpha * residual
            + beta * self.norm("post_feedforward_layernorm", h, input_scaled=True)
        ).to(residual.dtype)

    if "layer-output" not in graphs:
        graphs["layer-output"] = DecodeGraph.capture(output, projected, coefficients)
    return graphs["layer-output"].run(projected, coefficients)


fused_layer._muse_original = original_call


def replacements():
    return {
        "reap.muse_fast:FastMuseLayer.__call__": fused_layer,
        "reap.muse_fast:FastMuseLayer.moe": graph_moe,
        "reap.muse_fast:TiledBank.linear": retained_linear,
        "reap.muse_fast:FastMuseLayer.attention_inputs": graph_attention_inputs,
        "reap.muse_fast:FastMuseLayer.attention_output": graph_attention_output,
    }
