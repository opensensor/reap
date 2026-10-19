import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from reap.args import PruneArgs
from reap.model_util import get_moe, get_model_text_config
from reap.models.muse_spark import collect_muse_pruning_metrics, prune_muse_experts
from reap.observer import MoETransformerObserver, MuseSparkObserverHookConfig
from reap.prune import prune
from reap.pruning_metrics import (
    initialize_pruning_state,
    update_pruning_state,
    update_pruning_state_from_routed,
)
from tests.muse_reference import MuseSparkMoE


def make_moe(dtype=torch.float32, config=None):
    torch.manual_seed(314)
    config = config or SimpleNamespace(
        hidden_size=8,
        moe_hidden_size=4,
        moe_intermediate_size=6,
        num_local_experts=8,
        num_experts_per_tok=2,
        output_multiplier=0.1767766953,
        routed_scaling_factor=0.7,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        hidden_act="silu",
    )
    moe = MuseSparkMoE(config).to(dtype)
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0, 0.2)
        moe.gate.weight = nn.Parameter(moe.gate.weight.float())
        # Large selection bias deliberately disagrees with uncorrected logits.
        moe.gate.e_score_correction_bias = torch.linspace(
            -0.3, 0.3, config.num_local_experts
        )
    return moe.eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_collector_matches_actual_upstream_routed_outputs(dtype):
    moe = make_moe(dtype)
    hidden = torch.randn(2, 5, 8).to(dtype)
    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 1, 1, 0, 1]], dtype=torch.bool)
    valid = hidden.reshape(-1, 8)[mask.flatten()]
    routed_outputs = []
    route = []
    norm_hook = moe.experts.post_expert_norm.register_forward_hook(
        lambda _, args, output: routed_outputs.append(output.detach().float())
    )
    gate_hook = moe.gate.register_forward_hook(
        lambda _, args, output: route.append(output)
    )
    with torch.no_grad():
        moe(valid)
    norm_hook.remove()
    gate_hook.remove()
    logits, weights, ids = route[0]
    assert not torch.equal(ids, logits.topk(2, sorted=False).indices)
    assert torch.allclose(weights.sum(-1), torch.full((len(valid),), 0.7))
    state = initialize_pruning_state(8)
    collect_muse_pruning_metrics(moe, hidden, state, mask)
    counts = torch.bincount(ids.flatten(), minlength=8)
    expected_sum = torch.zeros(8, dtype=torch.float64)
    expected_norm = torch.zeros_like(expected_sum)
    for expert, output in zip(ids.unique(sorted=True).tolist(), routed_outputs):
        # Upstream iterates slot-major one-hot routes, not token-major indices.
        slot, token = torch.where(expert == ids.T)
        expected_sum[expert] = (
            output.norm(dim=-1).double() * weights[token, slot].double()
        ).sum()
        expected_norm[expert] = output.norm(dim=-1).double().sum()
    assert state["total_tokens"] == mask.sum()
    assert torch.equal(state["expert_frequency"], counts)
    assert torch.allclose(state["weighted_ean_sum"], expected_sum, rtol=1e-5, atol=1e-6)
    assert torch.allclose(state["ean_sum"], expected_norm, rtol=1e-5, atol=1e-6)
    assert torch.allclose(
        state["reap"].mean.double(),
        expected_sum / counts.clamp_min(1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_empty_mask_and_mask_length_validation():
    moe = make_moe()
    state = initialize_pruning_state(8)
    collect_muse_pruning_metrics(moe, torch.randn(1, 3, 8), state, torch.zeros(1, 3))
    assert state["total_tokens"] == 0
    assert torch.isfinite(state["reap"].mean).all()
    with pytest.raises(ValueError, match="mask"):
        collect_muse_pruning_metrics(moe, torch.randn(1, 3, 8), state, torch.ones(2))


def test_routed_summaries_match_existing_softmax_statistics_across_batches():
    torch.manual_seed(5)
    old, new = initialize_pruning_state(5), initialize_pruning_state(5)
    for tokens in (7, 3):
        activations = torch.randn(5, tokens, 4)
        logits = torch.randn(tokens, 5)
        ids = logits.topk(2).indices
        weights = logits.softmax(-1).gather(1, ids)
        outputs = activations[ids, torch.arange(tokens)[:, None]]
        update_pruning_state(
            old,
            activations=activations,
            selected_experts=ids,
            router_logits=logits,
            num_experts=5,
        )
        update_pruning_state_from_routed(
            new,
            selected_experts=ids,
            routing_weights=weights,
            activation_norms=outputs.norm(dim=-1),
            activation_maxima=outputs.amax(dim=-1),
            num_experts=5,
        )
    for key in old:
        a = old[key].mean if hasattr(old[key], "count_shape") else old[key]
        b = new[key].mean if hasattr(new[key], "count_shape") else new[key]
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), key


@pytest.mark.parametrize(
    "bad", ["duplicates", "range", "shape", "nan", "negative", "float_ids"]
)
def test_invalid_routed_summaries_fail_before_state_mutation(bad):
    state = initialize_pruning_state(3)
    args = dict(
        selected_experts=torch.tensor([[0, 1]]),
        routing_weights=torch.tensor([[0.3, 0.4]]),
        activation_norms=torch.ones(1, 2),
        activation_maxima=torch.ones(1, 2),
        num_experts=3,
    )
    if bad == "duplicates":
        args["selected_experts"][0, 1] = 0
    if bad == "range":
        args["selected_experts"][0, 1] = 3
    if bad == "shape":
        args["activation_norms"] = torch.ones(2)
    if bad == "nan":
        args["routing_weights"][0, 0] = float("nan")
    if bad == "negative":
        args["routing_weights"][0, 0] = -1
    if bad == "float_ids":
        args["selected_experts"] = args["selected_experts"].float()
    with pytest.raises(ValueError):
        update_pruning_state_from_routed(state, **args)
    assert state["total_tokens"] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sliced_moe_matches_masked_rerouting_and_reload(dtype):
    original = make_moe(dtype)
    pruned = copy.deepcopy(original)
    keep = torch.tensor([1, 3, 5, 7])
    x = torch.randn(2, 3, 8).to(dtype)
    shared_config = pruned.config
    shared_names = [
        n for n in pruned.state_dict() if "proj" not in n or "experts" not in n
    ]
    preserved = {
        n: p.clone()
        for n, p in pruned.state_dict().items()
        if "gate." not in n and (n in shared_names or "post_expert_norm" in n)
    }

    def masked_router(hidden):
        hidden = hidden.reshape(-1, 8).float()
        logits = (
            torch.nn.functional.linear(hidden, original.gate.weight.float())
            * original.gate.output_multiplier
        )
        scores = logits.sigmoid()
        choices = scores + original.gate.e_score_correction_bias
        choices[:, [0, 2, 4, 6]] = -torch.inf
        ids = choices.topk(2, sorted=False).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-15) * 0.7
        return logits, weights, ids

    with torch.no_grad(), patch.object(original.gate, "forward", masked_router):
        expected = original(x)
    prune_muse_experts(pruned, keep.flip(0))
    assert shared_config.num_local_experts == 8
    assert pruned.config.num_local_experts == 4
    assert pruned.gate.num_experts == pruned.experts.num_experts == 4
    assert pruned.gate.top_k == 2
    assert torch.equal(
        pruned.gate.e_score_correction_bias, original.gate.e_score_correction_bias[keep]
    )
    for name, value in preserved.items():
        assert torch.equal(pruned.state_dict()[name], value), name
    assert torch.allclose(pruned(x), expected, rtol=1e-5, atol=1e-6)
    reloaded = make_moe(dtype, config=copy.copy(pruned.config))
    reloaded.load_state_dict(pruned.state_dict(), strict=True)
    assert torch.equal(pruned(x), reloaded(x))


@pytest.mark.parametrize("keep", [[0], [0, 0], [-1, 2], [0, 8], [0.0, 1.0], [[0, 1]]])
def test_bad_retained_ids_do_not_mutate_weights(keep):
    moe = make_moe()
    before = {name: value.clone() for name, value in moe.state_dict().items()}
    with pytest.raises(ValueError):
        prune_muse_experts(moe, keep)
    for name, value in before.items():
        assert torch.equal(moe.state_dict()[name], value)


class MuseSparkForCausalLM(nn.Module):
    """Small native-MoE wrapper exercising REAP discovery and export dispatch."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.layers:
            layer.mlp = make_moe()
        self.config = self.model.layers[0].mlp.config

    def forward(self, x):
        for layer in self.model.layers:
            x = layer.mlp(x)
        return x

    def save_pretrained(self, path):
        save_file(self.state_dict(), path / "model.safetensors")
        config = get_model_text_config(self)
        (path / "config.json").write_text(json.dumps(vars(config)))


class MuseSparkForConditionalGeneration(MuseSparkForCausalLM):
    def __init__(self):
        super().__init__()
        text = self.model
        self.model = nn.Module()
        self.model.language_model = text
        self.config = SimpleNamespace(text_config=self.config)

    def forward(self, x):
        for layer in self.model.language_model.layers:
            x = layer.mlp(x)
        return x


@pytest.mark.parametrize(
    "model_cls", [MuseSparkForCausalLM, MuseSparkForConditionalGeneration]
)
def test_observer_prune_export_and_reload(model_cls, tmp_path):
    model = model_cls()
    inputs = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    observer = MoETransformerObserver(model, MuseSparkObserverHookConfig())
    before = model(inputs).detach()
    observer.reset()
    with observer.set_attention_mask(mask):
        assert torch.equal(model(inputs), before)
    data = observer.report_state()
    observer.close_hooks()
    assert set(data) == {0, 1}
    assert all(row["total_tokens"] == 7 for row in data.values())
    args = PruneArgs(prune_method="reap")
    prune(data, model, args, 4, tmp_path)
    assert get_model_text_config(model).num_local_experts == 4
    assert json.loads((tmp_path / "config.json").read_text())["num_local_experts"] == 4
    state = load_file(tmp_path / "model.safetensors")
    reloaded = model_cls()
    for i in range(2):
        prune_muse_experts(get_moe(reloaded, i), [0, 1, 2, 3])
    reloaded.load_state_dict(state, strict=True)
    assert torch.equal(model(inputs), reloaded(inputs))


def test_incomplete_or_invalid_calibration_rejected_before_pruning(tmp_path):
    model = MuseSparkForCausalLM()
    observer = MoETransformerObserver(model, MuseSparkObserverHookConfig())
    model(torch.randn(1, 5, 8))
    data = observer.report_state()
    observer.close_hooks()
    args = PruneArgs(prune_method="reap")
    for invalid in (
        {0: data[0]},
        {**data, 1: {**data[1], "reap": torch.full((8,), float("nan"))}},
    ):
        with pytest.raises(ValueError):
            prune(invalid, model, args, 4, tmp_path)
        assert get_moe(model, 0).experts.num_experts == 8
        assert not list(tmp_path.iterdir())


def test_merging_and_unimplemented_layerwise_replay_rejected():
    from reap.layerwise_observer import LayerwiseMoEObserver

    with pytest.raises(ValueError, match="pruning metrics only"):
        MuseSparkObserverHookConfig(record_pruning_metrics_only=False)
    with pytest.raises(NotImplementedError, match="RoPE/NoPE"):
        LayerwiseMoEObserver(MuseSparkForCausalLM(), MuseSparkObserverHookConfig())


def test_meta_selection_bias_rejected_before_pruning():
    moe = make_moe()
    bank = moe.experts.gate_up_proj
    moe.gate.e_score_correction_bias = moe.gate.e_score_correction_bias.to("meta")
    with pytest.raises(ValueError, match="materialize"):
        prune_muse_experts(moe, [0, 1, 2, 3])
    assert moe.experts.gate_up_proj is bank
