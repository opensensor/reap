# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned upstream eager Muse MoE classes for offline regression tests only.

Source: huggingface/transformers d4804f9d76b8c071b993d813e9e87067b780cbee,
src/transformers/models/muse_spark/modeling_muse_spark.py.
Changes: extract five classes, remove kernel-dispatch decorator, substitute
local activation map and deferred config type annotations. Model math unchanged.
This fixture is not a production model implementation.
"""

from __future__ import annotations
from types import SimpleNamespace as MuseSparkTextConfig
import torch
from torch import nn
from torch.nn import functional as F

ACT2FN = {"silu": F.silu, "sigmoid": torch.sigmoid}


class MuseSparkTextCenteredRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # (x * w).to(bfloat16) instead of x.to(bfloat16) * w
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight).float()
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class MuseSparkTextInputScaledRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Different order from MuseSparkTextCenteredRMSNorm: first scale, then normalize.
        output = x.float() * (1.0 + self.weight).float()
        output = self._norm(
            output.type_as(x).float()
        )  # round with fp32->bf16->f32 before norm
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class MuseSparkTopkRouter(nn.Module):
    def __init__(self, config: MuseSparkTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.output_multiplier = config.output_multiplier
        self.scoring_function = ACT2FN["sigmoid"]
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))
        self.norm_topk_prob_eps = 1e-15
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        # Different from MiniMaxM2: convert to float32 before linear projection and scale by output multiplier.
        router_logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32)
        )
        router_logits = router_logits * self.output_multiplier
        routing_weights = self.scoring_function(router_logits)
        scores_for_choice = routing_weights + self.e_score_correction_bias
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        top_k_weights = routing_weights.gather(1, topk_indices)
        # Different from MiniMaxM2: add topk prob eps and scale by routed scaling factor.
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + self.norm_topk_prob_eps
        )
        top_k_weights = top_k_weights * self.routed_scaling_factor
        return router_logits, top_k_weights, topk_indices


class MuseSparkExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: MuseSparkTextConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.moe_hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.post_expert_norm = MuseSparkTextInputScaledRMSNorm(
            self.hidden_dim, eps=config.post_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts + 1
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current_state, self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )
            # Different from DeepseekV3 which has no post expert norm
            current_hidden_states = self.post_expert_norm(current_hidden_states)
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


class MuseSparkMoE(nn.Module):
    def __init__(self, config: MuseSparkTextConfig):
        super().__init__()
        self.config = config
        self.gate = MuseSparkTopkRouter(config)
        self.pre_expert_proj = nn.Linear(
            config.hidden_size, config.moe_hidden_size, bias=False
        )
        self.pre_expert_norm = MuseSparkTextCenteredRMSNorm(
            config.moe_hidden_size, eps=config.rms_norm_eps
        )
        self.experts = MuseSparkExperts(config)
        self.post_expert_proj = nn.Linear(
            config.moe_hidden_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.pre_expert_proj(hidden_states)
        hidden_states = self.pre_expert_norm(hidden_states)
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights)
        hidden_states = self.post_expert_proj(hidden_states).view(*orig_shape)
        return hidden_states
