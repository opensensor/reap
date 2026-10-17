# Experimental Muse Spark pruning support

This implementation collects pruning statistics from unquantized eager Muse MoEs,
slices retained experts in memory, and streams text calibration from indexed
safetensors on CPU. It is not a complete 816B checkpoint conversion or serving
pipeline.

## Implemented

- A pruning-only observer configuration for `MuseSparkForCausalLM` and
  `MuseSparkForConditionalGeneration`, including the latter's nested text model.
- Actual model router weights and selected IDs, preserving sigmoid scoring,
  selection-only correction bias, selected-score normalization, and routed scaling.
  `renormalize_router_weights` does not override Muse's native coefficients.
- Expert outputs measured **after** the shared post-expert norm and **before**
  the weighted mixture/shared output projection. Only selected expert/token pairs
  are recomputed; summaries are `[tokens, top_k]`, avoiding a dense
  `[experts, tokens, hidden]` activation allocation.
- Existing REAP statistics, including conditional-mean `reap`, contribution sum
  `weighted_ean_sum`, and selection frequency. The existing softmax observer path
  for other models is unchanged.
- Consistent slicing of both fused banks, router rows, and selection-bias entries.
  Shared norms/projections and top-k are preserved. Retained IDs are sorted into
  original order; score ties retain lower original IDs.
- Uniform in-memory model pruning/export dispatch, with all-layer coverage,
  count, top-k, and finite-saliency checks before pruning begins. Config updates
  target `text_config` for the multimodal wrapper.
- A separate CPU streaming runner that constructs one meta-initialized embedding,
  decoder layer, or final norm at a time. `load_state_dict(assign=True)` preserves
  source precision, including FP32 routers and residual gates. Leaving a module
  context releases its weight mappings before the next module is loaded.
- Native per-layer full/sliding attention masks and RoPE/NoPE handling. Padding
  statistics always use the original token mask, not a causal-mask row.
- Atomic, checksummed activation/score artifacts and resumable layer boundaries,
  guarded by a single-writer file lock and checkpoint/input/backend identity.

The observer recomputes the routed expert outputs after the native forward; this
is a correctness-first implementation, not an optimized single-pass observer.
The per-MoE pruning utility allocates replacement banks and replaces Parameters;
it is intended for inference/export, not reuse with an existing optimizer.

## APIs

For a materialized Muse MoE and its real pre-MoE hidden states:

```python
from reap.models.muse_spark import collect_muse_pruning_metrics, prune_muse_experts
from reap.pruning_metrics import initialize_pruning_state

state = initialize_pruning_state(moe.experts.num_experts)
collect_muse_pruning_metrics(moe, hidden_states, state, valid_token_mask=padding_mask)

# REAP: state["reap"].mean
# Contribution weighting: state["weighted_ean_sum"]
# padding_mask uses True/1 for real tokens; None means all tokens are real.
# Only prune after calibration/validation; the operation changes this module.
prune_muse_experts(moe, retained_expert_ids)
```

For a model that already fits in memory and has a compatible native implementation:

```python
from reap.observer import MoETransformerObserver, MuseSparkObserverHookConfig

observer = MoETransformerObserver(model, MuseSparkObserverHookConfig())
with observer.set_attention_mask(batch.get("attention_mask")):
    model(**batch)
observations = observer.report_state()
observer.close_hooks()
```

`reap.prune.prune` accepts these observations with `PruneArgs(prune_method="reap")`
or `PruneArgs(prune_method="weighted_ean_sum")`. It currently requires observations
for every text layer and uniform expert counts. Calling `prune_muse_experts` on one
MoE updates that module's config copy; it does not update an enclosing model's config.

## Explicit limitations

Do **not** use the existing `reap.layerwise_prune` command to load the 816B model:
it materializes the whole checkpoint in host RAM before layer replay. Moreover,
its replay cache reuses first-layer attention kwargs. `LayerwiseMoEObserver`
continues to reject Muse; use the separate `reap.muse_streaming` entry point below,
which implements and tests these semantics without the legacy whole-model loader.

The upstream dependency pins remain unchanged. Transformers 4.55 does not
provide Muse. Production loading needs a separately validated Muse-capable
Transformers environment; changing the project's global pins would also require
testing its existing model implementations, evaluation dependencies, and vLLM.
The pruning adapter itself imports no Muse-specific Transformers implementation.
The streaming runner deliberately uses native Muse classes and native mask/RoPE
builders from the compatible environment.

Merging metrics, packed NVFP4/int4 weights, adaptive per-layer expert counts,
shard-streaming export, full-model generation quality, and Ascend/CANN kernels
are not supported by this change. Registering the model attributes does not
establish support for other REAP merging/conversion paths.

## Streamed CPU calibration

Install the separate native-Muse environment without changing REAP's legacy pins:

```sh
uv venv --python 3.14 .venv-muse
uv pip install --python .venv-muse/bin/python -r requirements-muse.txt
```

The tested backend is Transformers commit
`d4804f9d76b8c071b993d813e9e87067b780cbee` with Torch 2.11.0. Native integration
tests require this environment; they skip when the legacy environment lacks Muse.

Create a local JSON corpus with a `calibration` list of text strings. The CLI uses
the checkpoint's tokenizer locally, prepends BOS when configured, and processes
one snippet per batch. It does not apply a chat template or download data. For
instruction/trajectory-specific formatting, use the Python API with prepared
`input_ids`, optional binary `attention_mask`, and optional `position_ids` tensors.

```sh
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  .venv-muse/bin/python -m reap.muse_streaming \
  --source /path/to/Muse-Spark-Open-816B-A42B \
  --corpus /path/to/corpus.json --samples 64 --max-tokens 256 --threads 8 \
  --output /path/to/calibration-run --stop-after-layer 2
```

`--stop-after-layer` is an exclusive count: `2` processes layers 0 and 1. Repeat
with identical inputs/settings and `--resume`, omitting the stop option to process
the remaining layers. A completed run applies the final text norm and sets
`complete: true`; it does **not** compute logits, perplexity, or coding scores.

Output layout:

- `progress.json`: committed next layer, source/backend/input identity, checksums,
  timings, weight bytes, resident memory before/after release, and peak process RSS.
- `scores/layer-NNNNN.safetensors`: REAP means, contribution sums, frequency,
  and the other pruning statistics. Dense layers receive a marker instead.
- `activations-<stage>/batch-NNNNN.safetensors`: hidden states for the committed
  stage. During a transition, current and next-stage activations coexist. Previous
  stages are retired after the progress manifest commits; interrupted stages are
  recomputed on resume. Initial embedding and final norm interruptions also resume.

Resume requires the same source path, config/index, shard sizes/mtimes, tokenized
inputs, runner/scoring code, relevant backend source/version, and CPU thread count.
Saved activations and scores are checked with SHA-256. Source shard size/mtime
checks are a fast identity check, **not** a cryptographic hash of the entire 1.6TB
checkpoint. Keep the source read-only and unchanged for the duration of a run.
Output must be outside the checkpoint directory. A new run requires an empty
output directory; resuming requires its committed manifest.

This runner is Linux/CPU-only, uses eager attention, and supports the default Muse
RoPE with one global base plus per-layer enable/disable. Distinct nonzero per-layer
bases and other RoPE variants are rejected. Eager attention has quadratic sequence
memory cost; validate short contexts first. Weights are bounded to one layer, and
activation tensors are processed one batch at a time, but RSS also includes Python,
Torch workspaces, and resident file-backed pages. All input token IDs are retained
in host memory. No offloaded generation engine or accelerator backend is included.

## Validation

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_muse_spark.py tests/test_pruning_metrics.py
```

Tests use small unquantized eager MoEs extracted from the upstream implementation,
with no model downloads or gated weights. They verify FP32/BF16 statistics against
actual native forward outputs, biased routing and scaling, padding exclusion,
invalid-input rejection, masked rerouting versus physical slicing, shared-weight
preservation, both wrapper layouts, and safetensors save/reload dispatch. The
wrapper models used for export tests are small test harnesses, not complete native
Transformers models; full `from_pretrained` reload remains a separate milestone.

The test reference is pinned to Hugging Face Transformers commit
[`d4804f9d76b8c071b993d813e9e87067b780cbee`](https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/src/transformers/models/muse_spark/modeling_muse_spark.py).
`tests/muse_reference.py` retains the Apache-2.0 notice and documents its extraction
and removal of kernel-dispatch decorators. It is not a production model loader.

Native streamed-calibration validation:

```sh
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  .venv-muse/bin/python -m pytest -q tests/test_muse_streaming.py
```

These tests instantiate a real native `MuseSparkTextModel`, save its tensors across
several indexed shards, and compare streamed final hidden states and every pruning
statistic with native full forwards. FP32 and mixed BF16/FP32 results must match
exactly, including left/right padding, explicit positions, alternating full/sliding
attention, RoPE/NoPE, and sequences past the sliding window. They also cover module
release, interruption/restart, changed-input/artifact rejection, and writer locking.

## Remaining milestones

1. Calibrate all 62 layers on representative coding data; compare REAP and
   contribution weighting using held-out loss/logit drift and executable coding
   tasks. Near-constant post-normalization expert norms mean REAP may closely
   track conditional average routing weight on Muse.
2. Stream a separate pruned checkpoint with tensor slicing, index/config updates,
   retained-ID provenance, interrupted-run handling, and native reload parity.
3. Evaluate W4 quantization and the Ascend backend independently. Pruning quality,
   quantization quality, simulator parity, and actual device allocation/throughput
   are separate checks. Keeping top-8 does not halve active expert computation.
