"""Pruning-only adapter for Hugging Face Muse Spark's eager, unquantized MoE.

The model supplies its router and normalization implementation. This module
does not import a particular Transformers release or load a whole checkpoint.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from reap.pruning_metrics import update_pruning_state_from_routed


def validate_muse_moe(moe):
    """Reject layouts that cannot be safely observed or sliced by this adapter."""
    experts, router = moe.experts, moe.gate
    count = experts.num_experts
    up, down = experts.gate_up_proj, experts.down_proj
    if up.ndim != 3 or down.ndim != 3:
        raise ValueError("Muse requires 3D fused expert banks")
    if (
        up.shape[0] != count
        or down.shape[0] != count
        or up.shape[1] != 2 * down.shape[2]
        or up.shape[2] != down.shape[1]
    ):
        raise ValueError("inconsistent Muse fused expert shapes")
    if (
        router.num_experts != count
        or router.weight.shape[0] != count
        or router.e_score_correction_bias.shape != (count,)
    ):
        raise ValueError("Muse router and expert counts disagree")
    if not 0 < router.top_k <= count:
        raise ValueError("invalid Muse top_k")
    if any(p.device.type == "meta" for p in (*moe.parameters(), *moe.buffers())):
        raise ValueError("materialize the Muse MoE before observing or pruning it")
    if any(
        p.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        for p in (up, down, router.weight)
    ):
        raise ValueError("Muse adapter requires unquantized floating-point weights")


@torch.no_grad()
def collect_muse_pruning_metrics(
    moe, hidden_states, layer_state, valid_token_mask=None
):
    """Observe actual routed outputs after Muse's shared post-expert norm.

    Only selected expert/token pairs are evaluated. Persistent summaries have
    shape [tokens, top_k]; expert activations are released after each expert.
    Router weights are taken verbatim from the model, preserving sigmoid,
    selection-only correction bias, selected-score normalization, and scaling.
    """
    validate_muse_moe(moe)
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    if valid_token_mask is not None:
        mask = valid_token_mask.reshape(-1).to(device=hidden.device, dtype=torch.bool)
        if mask.numel() != hidden.shape[0]:
            raise ValueError("valid token mask must match the hidden-state token count")
        hidden = hidden[mask]
    _, weights, ids = moe.gate(hidden)
    latent = moe.pre_expert_norm(moe.pre_expert_proj(hidden))
    norms = torch.empty_like(weights, dtype=torch.float32)
    maxima = torch.empty_like(norms)
    for expert in ids.unique(sorted=True).tolist():
        tokens, slots = torch.where(ids == expert)
        gate, up = F.linear(latent[tokens], moe.experts.gate_up_proj[expert]).chunk(
            2, dim=-1
        )
        output = F.linear(moe.experts.act_fn(gate) * up, moe.experts.down_proj[expert])
        output = moe.experts.post_expert_norm(output).float()
        norms[tokens, slots] = output.norm(dim=-1)
        maxima[tokens, slots] = output.amax(dim=-1)
    update_pruning_state_from_routed(
        layer_state,
        selected_experts=ids,
        routing_weights=weights,
        activation_norms=norms,
        activation_maxima=maxima,
        num_experts=moe.experts.num_experts,
    )


def validate_retained_experts(moe, retained_experts):
    """Return sorted IDs after validating before any model mutation."""
    validate_muse_moe(moe)
    keep = torch.as_tensor(retained_experts)
    if keep.ndim != 1 or keep.dtype not in (torch.int32, torch.int64):
        raise ValueError("retained experts must be a one-dimensional integer sequence")
    if keep.numel() < moe.gate.top_k or keep.unique().numel() != keep.numel():
        raise ValueError("retain at least top_k distinct experts")
    if keep.min() < 0 or keep.max() >= moe.experts.num_experts:
        raise ValueError("retained expert ID out of range")
    return keep.to(dtype=torch.long).sort().values


@torch.no_grad()
def prune_muse_experts(moe, retained_experts):
    """Slice an in-memory MoE for inference, preserving shared norms/projections.

    This replaces Parameters, invalidating existing optimizer references. It is
    not an in-place training operation or a shard-streaming checkpoint exporter.
    """
    keep = validate_retained_experts(moe, retained_experts)

    def sliced(parameter):
        return nn.Parameter(
            parameter.index_select(0, keep.to(parameter.device)),
            requires_grad=parameter.requires_grad,
        )

    # Prepare replacements before mutating the module.
    up = sliced(moe.experts.gate_up_proj)
    down = sliced(moe.experts.down_proj)
    router = sliced(moe.gate.weight)
    bias = moe.gate.e_score_correction_bias.index_select(
        0, keep.to(moe.gate.e_score_correction_bias.device)
    )
    config = copy.copy(moe.config)
    config.num_local_experts = len(keep)
    moe.experts.gate_up_proj = up
    moe.experts.down_proj = down
    moe.gate.weight = router
    moe.gate.e_score_correction_bias = bias
    moe.experts.num_experts = len(keep)
    moe.gate.num_experts = len(keep)
    # Do not change other layers sharing the original config during pruning.
    moe.config = config
    return keep
