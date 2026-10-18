# Experimental Muse Spark pruning support

This implementation collects pruning statistics from unquantized eager Muse MoEs,
slices retained experts in memory, and streams text calibration from indexed
safetensors on CPU. It also exports a frozen pruning plan through ModelSlim into
a custom W4-expert/W8-attention deployment format. An initial standalone text
runtime supports sequential layer placement and greedy HTTP serving. Full-size
generation remains unvalidated; stock vLLM cannot load this custom checkpoint.

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

Merging metrics, quantized source checkpoints, adaptive per-layer expert counts,
full-model generation quality, and vLLM/distributed TP/EP integration
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

## Whole-model coding screen

`reap.muse_experiment` runs a CPU-only, resumable screening experiment. It first
calibrates every layer on the calibration split, freezes independent per-layer
expert selections, and then replays the validation split through the baseline and
each candidate. Defaults compare 128 and 144 retained experts using `reap` and
`weighted_ean_sum` (the sum of router-weighted output norms). Top-k is unchanged.
Candidates physically slice the in-memory layer; no checkpoint is exported.

The output head is loaded only after the decoder and final norm have finished.
Logits are computed in chunks of 32 valid token positions, using the native output
multiplier and tanh softcap in the checkpoint dtype, then FP32 log-softmax. Reports
include token-weighted next-token NLL/perplexity, change in NLL, baseline-to-candidate
KL divergence, top-1 agreement, and logit RMSE. Padding and transitions across
padding are excluded. Loss includes all valid text tokens, including task prompts;
it is **not** assistant-only loss, generation accuracy, or executable coding success.

The pilot corpus builder combines complete short Magicoder task/solution records
in Python, C++, TypeScript, and Rust; complete Glaive tool-use conversations;
repository test snippets; and complete short test diffs. It uses plain text with
BOS rather than a Muse chat template. Long instruction records are excluded;
only repository test files are chunked. This biases the pilot toward short tasks
and cannot establish long-context or agentic coding quality.

Download the pinned public inputs once; subsequent preparation and evaluation are
offline. Keep datasets and experiment outputs outside the repository:

```sh
hf download ise-uiuc/Magicoder-OSS-Instruct-75K --type dataset \
  --revision 5f839b1f368a76b161028bb9edff055db34022b2 \
  --include README.md --include data-oss_instruct-decontaminated.jsonl \
  --local-dir /data/magicoder
hf download glaiveai/glaive-function-calling-v2 --type dataset \
  --revision e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac \
  --include README.md --include glaive-function-calling-v2.json \
  --local-dir /data/glaive

PYTHONPATH=src python -m reap.muse_corpus \
  --magicoder /data/magicoder/data-oss_instruct-decontaminated.jsonl \
  --glaive /data/glaive/glaive-function-calling-v2.json \
  --repo /path/to/vllm-ascend --revision FULL_GIT_COMMIT \
  --model /path/to/Muse-Spark-Open-816B-A42B \
  --output /data/coding-v2.json --max-tokens 512

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m reap.muse_experiment \
  --source /path/to/Muse-Spark-Open-816B-A42B --corpus /data/coding-v2.json \
  --output /data/coding-screen-v2 --threads 8
```

Use the separate Muse environment described above. Repeat the last command with
`--resume` after an interruption. `--stop-after-calibration-layer N` can validate
an initial portion of calibration before resuming the full experiment. Inputs,
thread count, code, plans, source metadata, and activation checksums are verified.
Freeze a checkout for long experiments: editing the runner changes its identity
and deliberately prevents resuming with different code. Earlier streaming runs
must resume with the runner version that created them.

The builder saves exact token IDs, source revisions/file hashes, category labels,
and record IDs. Its deterministic split groups Magicoder by seed, Glaive by first
user request, and repository records by file. It excludes exact normalized
text/prompt duplicates and exact token duplicates. This is not semantic
deduplication or proof of independence from Muse pretraining. Published coding
benchmark test sets are not used in this calibration recipe.

The initial local corpus contains 96 calibration records (39,588 tokens) and 24
validation records (9,825 tokens), at most 512 tokens each. That is a screening
pilot, not a final calibration budget. Inspect per-layer coverage in `plans/*.json`,
especially never-observed or sparsely observed retained experts, before trusting
the rankings. The script rejects incomplete calibration and corrupt scores.

`experiment.json` records the active phase and completed evaluation results.
Each phase has the same layer-boundary progress manifest as the streaming runner.
Batch progress messages are informational; only committed layers are resumable.
The phases are `calibration`, `baseline`, and one directory per ranking/count.
Each validation phase writes `evaluation.json`, including per-record/category
metrics. Completion does not automatically select a winner or export weights.

All candidates reread the source layer shards, and pruning temporarily holds both
source and selected banks. This uses more peak RAM than baseline replay, but does
not materialize the full model. Cold storage and CPU compute can make the full
experiment take hours. Native integration tests compare streamed logits and
pruned forwards against a complete `MuseSparkForConditionalGeneration` in FP32
and mixed BF16/FP32, and verify loss/KL across chunk sizes and resumed candidates:

```sh
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q \
  tests/test_muse_streaming.py tests/test_muse_spark.py tests/test_pruning_metrics.py \
  tests/test_muse_evaluate.py tests/test_muse_corpus.py tests/test_muse_experiment.py
```

## Coding pilot and remaining quality checks

The 62-layer coding pilot selected contribution-weighted pruning to 128 experts
per layer for deployment bring-up. Calibration used 96 records (39,588 tokens),
and validation used 24 records (9,825 tokens), with a 512-token record limit.
Before quantization, validation perplexity increased from 2.24332 to 2.30044
(+2.55%); mean teacher KL was 0.03142 and top-1 agreement was 95.11%.
These are teacher-forced diagnostics, not executable coding accuracy.

The separate ModelSlim export completed all 62 layers and 189 shards, and its
transfer to the Ascend host passed SHA-256 verification. Remaining quality work
includes longer contexts, quantized-versus-pruned logit/loss comparisons, and
executable coding tasks. Pruning quality, quantization quality, simulator parity,
and device allocation/throughput are separate checks. Keeping top-8 does not
halve active expert computation.

## ModelSlim export for Ascend bring-up

`reap.muse_export` reads the original BF16 checkpoint and a validated, frozen
pruning plan. It slices retained experts directly from the source shards, then
uses ModelSlim IR symmetric min/max round-to-nearest quantization on CPU. This
initial export does not use activation calibration or measure quantized quality.

```sh
PYTHONPATH=src:/path/to/msmodelslim .venv-muse/bin/python -m reap.muse_export \
  --source /models/Muse-Spark-Open-816B-A42B \
  --plan /results/plans/weighted_ean_sum-128.json \
  --output /models/Muse-Spark-Contribution128-W4A16-W8Attn --threads 8
# Repeat with --resume after interruption; --stop-after-layer 1 is a smoke export.
```

The format is `muse_spark_w4a16_g128_w8attn_v1`:

- Experts: signed W4, group size 128 along K, FP16 scales. Each byte packs even K
  in the low nibble and odd K in the high nibble. Zero offsets are implicit.
- Attention projections, including the output gate: signed W8 with FP32
  per-output-channel scales. Activations remain floating point.
- Other BF16 weights become FP16; source FP32 router/gate values remain FP32.
  Router rows and correction biases follow the retained expert mapping. Shared
  norms/projections and top-k are preserved.
- Each layer has two expert shards split by retained expert ID and one dense
  shard. Global embeddings, final norm, and output head have separate shards.
  This layout anticipates EP2, but does not itself implement TP/PP execution.
- The exporter fingerprints the source, plan, exporter, and ModelSlim backend;
  resume verifies completed shard checksums. It publishes the index last and
  explicitly marks `deployment.json` as `serving_ready: false`.

The deployment candidate is contribution-weighted pruning with 128 retained
experts per layer (top-8 routing unchanged), selected from the coding pilot.
Full-model loading and HTTP coding generation have now passed on six Ascend
310P chips; the measured bring-up results are below. The intended optimized
layout is TP2/PP3/EP2. The first standalone runtime instead uses sequential
complete-layer placement across six devices (PP6), with CPU transport at stage
boundaries. Production performance and coding quality remain unqualified. The
export's `runtime_required` note describes the original TP2/PP3/EP2 target, not a hard format restriction.

`reap.muse_qualify prepare --export EXPORT --output PROBE` extracts three real
quantized expert projections and CPU FP16 references. On the Ascend host,
`reap.muse_qualify run --data PROBE --library EXTENSION.so --output RESULT.json`
compares an existing W4 operator against those references at token counts
1, 17, 128, and 129. `--tokens`, `--rows`, and `--projection` select smaller cases
for CANN simulator runs. The adapter chunks K at 2048 and M at 128 to fit
the existing Qwen operator, accumulating FP16 partial products in FP32. This
adds launches and partial-product rounding; it is an initial bring-up path. The
result records checksums and loaded simulator libraries. An existing Qwen extension may reject Muse dimensions: a successful
import or simulator startup alone does not qualify the kernel. This probe tests
same-quantized matrix multiplication, not end-to-end model quality.

## Standalone Ascend runtime

`reap.muse_runtime` loads the packed export without importing native Muse classes
on the serving host. Its text equations are checked against the pinned native
implementation, including the centered/input-scaled norms, FP32 router,
selection-only bias, normalized expert outputs, gated residuals, output softcap,
and per-layer RoPE/NoPE. A bounded sliding-window cache retains only the previous
window minus one tokens; full-attention layers retain the complete prefix.
Inputs are one contiguous unpadded text sequence. Prefill is chunked at 64 tokens
by default, with a maximum chunk size of 128. Reset clears request caches.

Resident expert weights remain packed W4. W8 attention uses a bounded fallback:
dequantize 512 output rows to FP16, multiply, then release the temporary weights.
The initial implementation prioritizes a verifiable path to generation; it has
no fused MoE dispatch, sampling, continuous batching, TP/EP collectives, or
performance claim. Complete layers are assigned to contiguous stages by a
minimax planner that includes global weights, bounded KV storage, and a default
3 GiB workspace reserve per device. Startup checks free memory before loading.
It does not stop existing services to obtain that memory.

Inspect the full export's placement without loading its weights:

```sh
PYTHONPATH=src python -m reap.muse_runtime --model /models/Muse-Spark-Contribution128-W4A16-W8Attn \
  --devices npu:0 npu:1 npu:2 npu:3 npu:4 npu:5 --plan-only
```

On a host with the validated Torch/NPU and custom operator build, source CANN's
`set_env.sh`, set `ASCEND_CUSTOM_OPP_PATH` to the operator vendor directory, and
prepend its `op_api/lib` to `LD_LIBRARY_PATH`. Preserve CANN's existing
`PYTHONPATH` when adding REAP's `src`. Set
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` before starting Python.
On the full export, the default allocator exhausted chip 3 with 36.21 GiB live
but 44.39 GiB reserved; a representative expert-bank allocation with expandable
segments held 3.094 GiB live and 3.104 GiB reserved. Then launch:

```sh
python -m reap.muse_serve --model /models/Muse-Spark-Contribution128-W4A16-W8Attn \
  --devices npu:0 npu:1 npu:2 npu:3 npu:4 npu:5 \
  --library /path/to/validated/vllm_ascend_C.so \
  --max-context 8192 --prefill-chunk 64 --host 127.0.0.1 --port 8002
```

The tested host environment uses Torch 2.13.0, torch_npu 2.13.0.rc1, CANN 9.1.0,
Transformers 5.17.0 for tokenization, FastAPI 0.136.3, and Uvicorn 0.53.0. The
runtime does not require ModelSlim to be installed. Quantization and serving
should run in separate processes: importing ModelSlim patches Torch/Pydantic.

The initial API provides `/health`, `/v1/models`, `/v1/completions`, and
`/v1/chat/completions`. Requests use `temperature: 0`, `n: 1`, and no streaming.
Chat accepts string content, uses the checkpoint's chat template with default
`reasoning_effort: low`, and separates Muse's self/user channels into
`reasoning_content` and `content`. Tools, multimodal content, sampling, and other
unsupported request fields are rejected. One request runs at a time; a busy
runtime returns HTTP 503. Health becomes available only after weights load.

For a server bound to loopback, forward the port from your client machine:

```sh
ssh -N -L 8002:127.0.0.1:8002 ASCEND_HOST
```

Then send a greedy request (the initial runtime can take minutes):

```sh
curl http://127.0.0.1:8002/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"def add(a, b):\n    ","max_tokens":16,"temperature":0}'
```

Validation so far:

- A small exported native model matches dequantized native full forwards,
  cached decoding beyond the sliding window, and greedy generation/reset.
- A complete real exported layer, including W8 attention and W4 MoE, passed
  physical 310P forwards for an 11-token coding input both together and token
  by token. Maximum absolute error versus native CPU was 0.0078125 and RMSE
  about 0.000689 (tolerance `rtol=0.01, atol=0.005`). The CPU runtime matched
  the native reference exactly for this input.
- A separate **synthetic** six-layer model placed one layer on each of the six
  physical chips, matched native CPU logits within tolerance, and generated the
  same three greedy token IDs. Its HTTP completions and chat routes returned
  successful responses. This is transport/API validation, not evidence of
  full-size Muse generation or coding quality.

`reap.muse_runtime_qualify prepare` builds the real first-layer native reference;
`run` verifies full and cached execution on the host.

On 2026-10-18, runtime commit `6796c68` loaded the complete 189-shard export on
six physical 310P3 chips using expandable allocator segments. The first real
`/v1/completions` request completed successfully: prompt `def add(a, b):` plus
newline/indentation generated `return a + b` and a subtraction function.
The resulting Python parsed and passed eight arithmetic assertions (positive,
negative, zero, and fractional inputs), after a strict AST allowlist check.
This is a trivial executable smoke, not a coding benchmark.

That request used seven prompt tokens and generated 16 tokens in 116.27 seconds,
including prompt processing and first-request overhead. Its aggregate output
rate was about 0.138 tokens/second; this is not a steady-state decode benchmark.
During generation, `npu-smi` reported server process memory of approximately
34.6–40.6 GiB per chip, with the highest total device usage about 41.8 GiB.
That initial deployment exposed an 8,192-token limit. The optimized path and
larger-cache qualification are documented below; long-context quality remains untested. Quantized coding quality, longer-context correctness,
and performance optimization remain outstanding.

The full-model `/v1/chat/completions` smoke also returned HTTP 200, correctly
separating reasoning and answer text. It used 162 prompt tokens and generated
64 tokens in 540.71 seconds. The output allowance ended during the final function
(`finish_reason: length`); this is chat-format validation, not a completed chat
coding result. Reasoning consumes the same output allowance: request more than
64 output tokens for useful chat answers, and inspect `finish_reason`.

`reap.muse_deploy` can supervise deployment without stopping another
service. It waits for the transfer's `transfer-verified.json`, checks all six
NPUs against the placement budget, launches the server, verifies that health
belongs to its child process, and captures a nonempty 16-token coding completion.
Only then does its atomic status change to `serving`. It records loading/failure
states and server logs, and terminates only its own child if the launch or smoke
fails. This final smoke establishes generation availability, not coding quality.


## Optimized 310P execution

Pass `--optimized` to `muse_serve`, `muse_runtime`, or `muse_deploy` to select the
optimized execution path. The original path remains available for comparison.
The on-disk export and quantization recipe are unchanged.

At load time, expert banks are split along K and losslessly encoded into the
installed operator's Cube tile layout. Gate and up projections share one bank.
Decode routes experts on-device and preserves ascending-expert FP16 accumulation
order; prefill groups tokens by expert on the host to reuse decoded tiles.
The installed routed kernel was substantially slower for multi-token groups,
so it is used only for single-token decoding.

Attention weights are dequantized once to the same FP16 values used by the
reference path. Dense FP16 matrices, including the output head, are explicitly
stored in FRACTAL_NZ format; implicit format selection remains disabled for
activations and packed custom operators. The placement budget includes the
larger attention weights and retained tiled zero-point metadata. This uses more
resident memory to eliminate repeated dequantization and format conversions.

Generation logs now report prompt processing progress, output-token progress,
and separate prefill/decode durations. A CPU-only host can calculate optimized
NPU placement with `--plan-only --optimized`, without importing torch_npu.

The complete optimized model was measured against three fixed coding completion
requests on the same six chips. Responses matched the reference text exactly:

| Request | Reference seconds | Optimized seconds | Speedup |
| --- | ---: | ---: | ---: |
| Add, 7 prompt / 16 output tokens | 115.28 | 24.99 | 4.61x |
| Reverse, 6 prompt / 8 output tokens | 62.87 | 13.85 | 4.54x |
| Factorial, 15 prompt / 8 output tokens | 72.88 | 16.70 | 4.37x |

A text-chat request with 162 prompt tokens finished after 72 generated tokens in
133.50 seconds, returned a complete addition function, and passed four arithmetic
assertions. These are latency and regression smokes, not a coding benchmark.
`scripts/benchmark_muse_http.py` records requests, responses, timings, and optional
baseline text comparisons.

## Bounded grouped attention and larger contexts

Add `--bounded-attention --max-context 98304` to `--optimized` for the expanded
context path. Adaptive query subchunks bound temporary attention score and softmax
storage, reaching four queries at a 98,304-token prefix. Query heads are grouped against 16 KV heads without expanding keys and
values to 128 heads. Full-attention caches allocate their configured capacity
once per request and append in place; their valid prefixes are made contiguous
at most once per layer forward. Sliding caches retain 2,047 tokens plus room for
one prefill chunk, included in the placement budget. All KV values remain FP16.
Native RoPE and the checkpoint's 2,048-token sliding window are unchanged.

The checkpoint's native position limit is 1,048,576. The six-chip memory plan
fits **98,304 total prompt plus generated tokens** with the existing 3 GiB
workspace reserve per chip. This is a configured deployment target, not a claim
that the complete window has passed retrieval or coding quality evaluation.
A 1M FP16 KV cache for full-attention layers alone requires 64 GiB and does not fit
the remaining resident memory; it needs an offload or cache-compression strategy.

The installed 310P fused-attention kernel passed isolated 65,536-token and real
first-layer checks, but timed out during full-model generation on device 1
(`IncreFlashAttention`, ACL error 507014), requiring an in-band card reset.
Production uses standard grouped matrix operations instead. The fused launch
option is removed so the isolated check cannot be mistaken for deployment
qualification. Its 65,536-token kernel limit does not apply to grouped attention.

CPU tests compare grouped outputs against an independent repeated-KV reference,
including causal masks, sliding-window rollover, chunk boundaries, stable cache
allocation, and NaN-poisoned excluded capacity. On hardware, a four-query 98,304-token comparison matched the grouped reference
exactly. Real-layer prefill errors were 0.0009766 for sliding attention and zero
for full attention; cached native-reference maximum error was 0.0078125, with
RMSE 0.000689, matching the reference runtime's tolerance. Full-model results
are recorded after deployment qualification. Capacity and trivial coding checks do not
establish long-context retrieval or coding quality.

The grouped path also passed synthetic cached and full forwards across all six
chips from a worker thread (maximum logit error 0.0006104) and generated the same
three greedy token IDs as native CPU. All 83 focused Muse tests passed.

The complete `ab28485` runtime loaded all 62 layers with
`--optimized --bounded-attention --max-context 98304 --prefill-chunk 64`.
Its first 16-token completion finished in 26.12 seconds, including 5.58 seconds
of prefill, and allocated the configured full-capacity caches. Repeated coding
checks preserved all three baseline responses exactly:

| Request | Reference seconds | 96K-configured seconds | Speedup |
| --- | ---: | ---: | ---: |
| Add, 7 prompt / 16 output tokens | 115.28 | 25.16 | 4.58x |
| Reverse, 6 prompt / 8 output tokens | 62.87 | 13.91 | 4.52x |
| Factorial, 15 prompt / 8 output tokens | 72.88 | 16.96 | 4.30x |

These requests use short prompts with the larger cache capacity configured; they
are not measurements of 96K prompt processing speed or quality.

The expanded-context chat check finished normally with 162 prompt and 84 output
tokens in 151.77 seconds. Its final addition function passed four arithmetic
assertions and matched the earlier final answer, although the reasoning text and
length differed (previously 72 output tokens). Grouped attention preserves the
quantized weights but changes floating-point operation shapes, so it does not
promise bitwise-identical generation for every prompt.

A separate full-model prompt crossed the sliding-cache boundary: 2,127 prompt
tokens plus 16 generated tokens completed in 317.19 seconds (296.06 prefill,
21.12 decode). Its returned addition function passed syntax validation and four
arithmetic assertions. This establishes full-model cache rollover and generation
beyond 2,048 tokens; the complete 96K window has not been evaluated for retrieval,
coding quality, or end-to-end prompt latency.

## Qwen and GLM performance transfers

Runtime `cd18b08` builds on the existing Qwen W4 tiling and resident NZ weights:

- Prefill sorts routes once, gathers all expert inputs together, then batches
  post-expert normalization and output combination. Expert projections still
  reuse their weights across tokens. The final sum retains ascending expert
  order and FP16 rounding after every contribution.
- `--optimized --bounded-attention` accepts prefill chunks up to 512 tokens.
  Individual expert GEMMs remain split into at most 128 rows, respecting the
  installed operator's limit. The larger sliding-cache append area is included
  in placement planning; maximum context stays at 98,304.
- Grouped attention batches more queries for short prefixes, with a fixed score
  budget of at most `32 * min(max_context, 98304)` elements per KV head. It keeps
  the previous four-query geometry at a full 96K prefix and never expands KV
  heads. Injected operators retain their previous conservative geometry.
- Each stage shares rotary phases for the current position range. Centered norm
  weights and residual-gate coefficients are prepared once instead of rebuilt
  during every forward.
- Aligned FP16 rows use the installed NPU RMS operator. A hardware ablation found
  invalid, unstable results for width-eight rows; unaligned rows and other dtypes
  use the reference math. The real Muse widths (64, 4096, 8192) passed checks at
  both RMS epsilon values, including small inputs.

These changes preserve the checkpoint, W4 expert codes/scales, top-eight routing,
FP16 activations and KV caches, native RoPE, and sliding-window semantics.
Fused RMS and different GEMM/attention shapes can change FP16 rounding, so the
runtime does not promise identical tokens for every prompt.

On the real first layer, median decode time decreased from 21.59 to 20.23 ms.
The 64-token prefill decreased from 169.75 to 152.38 ms; 512 tokens took 366.67 ms,
about 3.7 times the old 64-token per-layer throughput. These isolated-layer
measurements do not establish full-model throughput. Native cached-layer maximum
error remained 0.0078125. The corrected six-chip synthetic run matched the native
greedy IDs `[26, 4, 9]`, with maximum logit error 0.0006104. All 89 focused tests
passed, including large expert groups, causal masks, poisoned unused cache
capacity, position-cache reuse, and placement admission.
`scripts/qualify_muse_rms.py --device npu:0 --output rms-checks.json` reproduces
the RMS hardware gate with CANN sourced and `PYTHONPATH=src`; all 84 combinations
passed on this host, each repeated three times, including weighted normalization
and zero inputs.

The frozen runtime completed full-model deployment with
`--optimized --bounded-attention --max-context 98304 --prefill-chunk 512`.
The same requests were compared with the previous 96K runtime (`ab28485`):

| Request | Previous seconds | New seconds |
| --- | ---: | ---: |
| Add, 7 prompt / 16 output | 25.16 | 23.31 |
| Reverse, 6 prompt / 8 output | 13.91 | 12.85 |
| Factorial, 15 prompt / 8 output | 16.96 | 15.51 |
| Chat, 162 prompt / 84 output | 151.77 | 122.96 |
| Sliding rollover, 2,127 prompt / 16 output | 317.19 | 128.69 |

All three completions, the full chat message (including reasoning), and the
long-prompt completion matched the previous text. Chat and long-prompt addition
functions each passed four arithmetic assertions. On the 2,127-token prompt,
prefill fell from 296.06 to 109.24 seconds: **7.18 to 19.47 prompt tokens/s,
2.71x faster**. Decode improved from 0.733 to 0.784 tokens/s on the short add
request and 0.710 to 0.771 on the longer prompt. Decode rates count the generated
tokens after the first token, which comes from prefill. HTTP times include both
phases. These are single-request measurements; they do not measure concurrent
request throughput or full-96K retrieval/quality. Final health, configured cache
capacity, and all 86 frozen source/operator file hashes were checked.

The Qwen/GLM sources also contain model-specific recurrent attention, sparse
attention, MTP state handling, and native W4A8 paths. Those are not drop-in Muse
operations: W4A8 changes activation precision, and MTP needs a compatible draft
model plus accepted/rejected-cache handling. This deployment retains W4A16 and
sequential layer placement. Cross-request batching and TP/EP execution require
a separate scheduler and distributed model adapter; they are not enabled by
increasing the prefill chunk.

### Resident experiments and packed weight loading

The optimized single-process server supports authenticated local controls for
switching Python functions and versioned native W4 modules while all six stages
keep their weights allocated. Enable `--resident-token-file /private/token` with
a mode-0600 file containing a random secret of at least 32 characters. Controls
require both a loopback TCP peer and the bearer secret; forwarded headers are
not trusted. Inference continues to use the same single-request lock. A busy
runtime returns 503 to a control request, so the client can retry after inference
finishes. There is no background scheduler or graph capture to drain.

Use `--port 8001 --additional-port 8002` to serve both listeners from the same
process, model and request lock. Additional ports do not create extra replicas
or increase concurrency.

```bash
python scripts/muse_resident_control.py --token-file /private/token status
python scripts/muse_resident_control.py --token-file /private/token \
  load-native /absolute/path/manifest.json
python scripts/muse_resident_control.py --token-file /private/token \
  switch --native muse_w4_v2_k1024
# Restore original Python and the original W4 backend:
python scripts/muse_resident_control.py --token-file /private/token switch
```

Native loading checks the library and kernel SHA256 before mutation, rejects
namespace/name collisions, prepares a kernel on each stage, and compares actual
resident gate/up weights against the installed operators for decode and grouped
prefill, including empty expert groups and more than 128 rows. Loading leaves
serving dispatch unchanged. A failure after native mutation blocks inference
until restart; device code and Torch registrations cannot safely be unloaded.
An identical successful manifest can be loaded again without re-registration.

A dispatch switch compares complete logits for a four-token prefill and a cached
one-token decode, requiring exact equality, unchanged weight storage addresses,
and device synchronization. Failure restores the prior functions/backend and
rechecks its logits before allowing requests. These are small transactional
checks, not a substitute for a full coding or long-context evaluation. Health
reports the PID, active module, generation and weight-storage digest. Native
modules stay loaded until process exit even after dispatch rollback.

A trusted Python candidate defines `replacements()` returning a dictionary from
an allowed `module:Class.method` name to a callable. Pass it with `switch
--candidate candidate.py` and repeat `--native NAME` to retain a native backend.
Omitting either option restores that component to its startup implementation.
Allowed functions are listed in `reap.muse_resident.TARGETS`; initially they cover
MoE, RMS, attention inputs/outputs and tiled expert dispatch. Candidates must
only define functions: no tensor mutation, model construction or device-state
changes during import or `replacements()`. This is trusted development code,
not a sandbox. Weight layouts, model construction and attention cache allocation
still require a restart.

Build an independently named direct-launch module on the target CANN/Torch
installation after sourcing CANN's environment:

```bash
python scripts/muse_native/build.py \
  --qwen-source /path/to/pinned/qwen-checkout \
  --catlass-include /path/to/catlass/include \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --output /new/absolute/build-directory --namespace muse_w4_v2 --tile 1024
```

The builder copies and fingerprints its actual external Qwen/Catlass inputs;
upstream sources retain their licenses. Its bridge uses explicit ACLRTC device
code, bypassing OPP discovery. The kernel reuses the existing resident W4 layout
and FP16 split-K rounding; tile 1024 doubles the unpacking batch relative to the
512 control. Grouped prefill processes sorted routes in one launch per split-K
part instead of launching separately for each expert. Decode reuses the same
interface with device route IDs. No activation or weight precision is reduced.

For subsequent startups, enable `--packed-cache /path/to/cache`. Each layer is
stored as safetensors with a SHA256 receipt, keyed by source shard receipts,
config digest and packing ABI. It contains tiled W4 banks and dequantized dense
FP16 matrices. Cache hits validate those bytes instead of rereading and repacking
the source expert shards. Cache creation validates the canonical shards; corrupt
cache entries fail closed. Files are written under a per-layer lock and published
by atomic directory rename. The canonical model export is unchanged. NPUs still
apply their dense NZ format during transfer. Explicit aligned host staging avoids
slow driver transfers from large mmap views.

The cache can be prepared on CPU while a previous model remains served:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m reap.muse_packed \
  --model /path/to/export --cache /path/to/cache
```

The packed cache needs approximately another full checkpoint's disk capacity.
The first build includes packing, writing and hashing costs; startup acceleration
must be measured on cache hits. Hot switches do not read model weights from disk.

`--deterministic` enables Torch/CANN deterministic algorithms. In the real
layer-0 qualification, the existing non-deterministic path varied by up to
0.0009765625 across repeated 512-token MoE calls. Deterministic execution removed
this variation: all baseline/native comparisons were exact at 1, 7, 128 and 512
tokens, with no material timing penalty in that test. Native-1024 MoE medians
were 14.46 vs 17.82 ms at one token and 289.74 vs 335.21 ms at 512 tokens. These
are isolated real-layer results, not full-model throughput claims.

After qualification, `--native-manifest /absolute/path/manifest.json` restores the
selected native module on future startups. It requires resident controls and
runs the same per-device and full-logit gates before opening the listeners.
The server fails startup if qualification fails. Keep a launcher without this
flag for baseline recovery.

The repeatable real-layer gate is `scripts/qualify_muse_native.py`; pass the
model, native manifest, original operator library, device and output path. It
checks every repeated result exactly under deterministic execution and reports
synchronized medians separately for each prompt size.

### Resident deployment qualification (2026-10-18)

The six-chip server now listens on loopback ports **8001 and 8002** with one
resident model, 98,304-token total context and 512-token prefill chunks. Native
`muse_w4_v2_k1024` is active. Loading and qualifying the module on all six devices
took **0.261 seconds**; the first full-logit dispatch switch took **6.729 seconds**.
Native switching, a Python MoE wrapper, baseline rollback and native restoration
all preserved the server PID and the digest of checkpoint storage addresses.

Same-process deterministic comparisons, with unchanged generated text:

| Measurement | Original backend | Native 1024 |
| --- | ---: | ---: |
| Decode, short add request | 0.789 tok/s | **0.940 tok/s** |
| Prefill, 2,127-token prompt | 19.459 tok/s | **22.519 tok/s** |
| Decode after that prompt | 0.776 tok/s | **0.922 tok/s** |
| Short add, complete HTTP request | 23.180 s | **19.338 s** |
| Long prompt, complete HTTP request | 128.636 s | **110.730 s** |

The three completion regressions, 162-prompt/84-output-token chat (including
reasoning text), and 2,127-token rollover completion all match the previous
validated server exactly. Chat latency is 102.625 seconds versus the previous
122.961 seconds. Chat and long-prompt addition functions each pass four arithmetic
assertions. These remain single-request coding smokes; full-model 96K quality
and concurrent-request throughput have not been evaluated.

All 62 layers loaded with verified packed-cache hits. The cache occupies
241,589,798,672 bytes (about 225 GiB). A separate real layer-0 CPU preparation
comparison found every dense tensor and tiled bank exactly equal to canonical
loading. Across two repetitions, median preparation fell from 5.342 to 2.478
seconds (**2.16x**). This excludes NPU transfer/NZ conversion and is not a complete
startup speed measurement; the initial server startup overlapped cache builders.

Validation includes **93 focused tests**, the reusable real-layer native gate,
all-six-device resident projection checks, Python rollback and native failure
handling. Frozen current and next-start runtime inventories verified 341 and
342 source/operator/build-input files respectively. The prepared next-start
launcher uses runtime `9988d30` and `--native-manifest` to requalify and activate
the selected module before opening listeners; the running process retains its
original `33230a4` Python image and the hot-loaded native module.

### Tensor-parallel experts and decode graphs

`--tensor-experts` distributes the output channels of every expert projection
across all configured NPUs. Attention, dense projections, and KV remain on
contiguous stages. The memory planner balances those stages again after dividing
expert weights; it includes the requested full/sliding KV capacity and reserve.
Unequal 128-channel partitions rotate between devices across layers. Gates and
up projections use matching partitions; down projections gather complete output
vectors before the shared normalization and ordered expert sum.

This keeps all 128 experts, top-8 routing, W4 weights, FP16 activations, and the
existing split-K rounding. It requires neither activation quantization nor
cross-device floating-point reductions. Two host gathers connect the parallel
expert projections. Dedicated device workers execute all NPU shards together using private host
staging allocations;
single-token decode replays captured projection graphs. Grouped prefill retains
the eager native path and handles up to 512 tokens per chunk.

For additional decode graph coverage, pass the included trusted candidate:

```bash
python -m reap.muse_serve \
  --model /path/to/export --library /path/to/vllm_ascend_C.so \
  --devices npu:0 npu:1 npu:2 npu:3 npu:4 npu:5 \
  --optimized --bounded-attention --tensor-experts --deterministic \
  --max-context 98304 --prefill-chunk 512 \
  --packed-cache /path/to/cache \
  --resident-token-file /private/token \
  --native-manifest /absolute/path/manifest.json \
  --resident-patch-file scripts/muse_resident_patches/tensor_decode_graphs.py \
  --port 8001 --additional-port 8002
```

The candidate captures the decode router, expert combination, attention input
projections/RoPE, and attention output projections. Attention scores, masks, and
KV appends remain dynamic. Every replay updates activation, route, coefficient,
and rotary buffers. The startup option applies the same exact-logit gate as a
hot switch before opening listeners. The candidate can also be installed with
`muse_resident_control.py switch --native NAME --candidate FILE`.

Captures share one memory pool and an explicit capture stream per device.
Individual graph pools exhausted the driver's virtual address reservations at
full-model scale with the expandable allocator, despite sufficient weight
memory. Captured outputs keep their own live tensor references, and ordered
replay permits scratch reuse. Native geometry tensors created before capture
are retained even if varying prefill sizes evict their lookup-cache entries.
Hot switches reset graphs on their owning devices, discard retired pool handles,
and recapture into fresh pools after switching Python or native code; storage
fingerprints include every nonowner expert shard.

### Six-chip tensor deployment qualification (2026-10-18)

Frozen runtime `2f816ed` is serving on ports **8001 and 8002**, with native
`muse_w4_v2_k1024` and the additional decode graph candidate active. All 62 layers
loaded from verified packed-cache hits. The deployment keeps 128 experts, top-8
routing, the same quantized weights, FP16 KV, 98,304-token total context and
512-token prefill chunks. Dense/KV stage boundaries are 0/7/19/31/43/55/62;
every expert projection uses all six chips.

| Measurement | Sequential native | Tensor experts | Tensor + decode graphs |
| --- | ---: | ---: | ---: |
| Short-request decode | 0.940 tok/s | 1.879 tok/s | **1.980 tok/s** |
| Chat decode | 0.935 tok/s | 1.874 tok/s | **1.977 tok/s** |
| Decode after 2,127 prompt tokens | 0.922 tok/s | 1.786 tok/s | **1.902 tok/s** |
| Prefill, 2,127-token prompt | 22.519 tok/s | 31.496 tok/s | **31.891 tok/s** |
| Short add, complete HTTP request | 19.338 s | 9.122 s | **8.709 s** |
| Chat, complete HTTP request | 102.625 s | 49.016 s | **46.641 s** |
| Long prompt, complete HTTP request | 110.730 s | 75.935 s | **74.586 s** |

Decode counts cached steps (`completion_tokens - 1`), since prefill produces
the first token. These are single-request timings, not concurrent throughput.
The roughly 2.1x decode gain remains slow for interactive coding.

All three completion regressions, the entire 162-prompt/84-output-token chat
message including reasoning, and the 2,127-token rollover response match the
previous validated server exactly. The two generated addition functions pass
eight arithmetic assertions in total. Python patch reapplication, Python
rollback, native rollback and restoration pass exact-logit gates while retaining
the same process and checkpoint storage addresses. Both ports share that process.

Validation includes **103 focused tests**, exact real-layer checks at
1/7/128/512 tokens, changing decode inputs/routes, 1,200 simultaneous captured
graphs across six chips, and four reset/recapture cycles over 62 graph groups.
All 340 frozen runtime inventory entries passed SHA256 verification. The next
startup launcher reapplies the native manifest and trusted Python candidate
through the startup qualification gate. The sequential native deployment remains
available as a rollback launcher.

This is a throughput and regression qualification of the existing export, not
a broad coding quality evaluation. Full-model context regression reaches 2,127
prompt tokens; full 96K retrieval/quality is still unmeasured. Dynamic attention
and dense stages remain sequential, and expert projections still gather through
host memory. Further work should measure those costs before changing kernels or
numerics. Local deployment evidence is in `../results/muse-opt-v4/final-summary.json`
and the two validation summaries alongside it.

### Native token streaming and vLLM migration

Runtime `fa9e722` adds `stream: true` for completion and text-chat requests.
Each generated token reaches an SSE response during generation. Chat deltas
separate `reasoning_content` and `content`, withhold partial Muse channel markers
and incomplete Unicode, and finish with the normal stop/length reason and
`[DONE]`. `stream_options: {"include_usage": true}` adds a final usage event.
Disconnects cancel at a generation boundary; the request lock stays held until
the worker has stopped and reset its KV cache. Streaming preserves greedy token
selection and the single-request admission policy. It does not add tool calls.

The Kilo provider is `muse-ascend/Muse-Spark-Contribution128-W4A16-W8Attn`, with
the `muse-chat` agent for text/code discussion. The workstation forwards
`127.0.0.1:18001` to threadripper's loopback port 8001 through the user service
`muse-kilo-tunnel.service`. No buffered compatibility bridge is needed for the
streaming runtime. Kilo's configuration reserves 4,096 output tokens within the
98,304-token context and keeps the existing default model unchanged.
A direct native-streaming Kilo smoke (`kilo run --pure --agent muse-chat`)
returned `OK` with 689 input and 34 output tokens in 41.22 seconds including
CLI startup. The Kilo token count includes the generated reasoning.

Local source inspection found that QFN's `tools/sidecard/qfn_resident_server.py`
wraps the real vLLM API/engine factory and subprocess executor. Its worker
extension in `qfn_resident_worker.py` adds model-runner RPCs, graph retirement,
collective lifetimes and prefix-state management. Those controls are reusable
patterns; they do not themselves implement a model's forward or weight loader.
The inspected local vLLM and vllm-ascend model registries have no Muse adapter.

Moving Muse onto that engine requires the following work:

1. Implement/register Muse's text-model forward, including latent projections,
   selection-only router bias, expert output normalization and gated residuals.
   Bind the existing export receipts and packed W4/W8 loading to vLLM's loader.
2. Select a supported parallel layout before changing execution. The current
   ragged six-rank expert shards are not stock equal-width tensor parallelism.
   Muse's 128 query heads also fail the inspected vLLM configuration check for
   TP=6. TP=2/PP=3 satisfies head divisibility but reduces the number of chips
   working on each layer; it needs a latency/memory comparison before adoption.
   Retaining all-six-chip expert execution needs a custom parallel adapter.
3. Integrate full and sliding FP16 KV with vLLM's cache manager and attention
   metadata; qualify chunked prefill, prefix reuse, cancellation and interleaved
   requests. The current single-request AppendCache cannot be shared unchanged
   by a continuous batching scheduler.
4. Reuse the QFN resident/collective lifecycle after validating the communication
   operation. Muse currently concatenates independent output shards; replacing
   this with an all-reduce would change the arithmetic. Capture graph-safe
   gathers, then gate complete logits and executable coding responses before
   comparing single-request latency and concurrent throughput.

The next native-kernel experiment flattened decode row/tile scheduling to reduce
idle cores on 640-channel shards. It passed exact hardware checks and full-model
resident switch gates, but median short decode changed only from 1.996 to
2.008 tok/s over three repetitions (0.6%). It was not promoted. A separate
grouped-prefill table-reuse variant failed exact output checks and was rejected
before resident loading. The streaming runtime retains the established native
1024 backend and skips redundant device-context visits for prepared decode graphs.

Runtime `fa9e722` was qualified on ports 8001 and 8002 after host recovery on
2026-10-18. All 62 layers loaded from verified packed-cache entries, and all
421 frozen runtime files passed SHA256 checks. Native SSE passes exact response,
reasoning, usage and finish-reason comparisons on three coding completions,
a chat exchange and the 2,127-token prompt. Disconnecting during decode releases
the request lock and permits the next exact response without a worker restart.
The 109 focused Muse tests and CPU socket/disconnect tests also pass.

The short completion delivers its first text at **1.166 seconds**, followed by
16 incremental text events over 8.716 seconds, at **1.986 decode tok/s**.
Chat's first visible reasoning/text arrives at **6.271 seconds** and its decode
rate is **1.992 tok/s**. The 2,127-token prompt measures **31.425 prefill tok/s**
and **1.919 decode tok/s**, with first text at 67.690 seconds. These are streamed
single-request measurements; streaming improves visibility, not token production
speed. Full-model context quality beyond the existing 2,127-token regression
remains unmeasured.

Recovery corrected a host startup script that expected four Ascend device nodes
and therefore failed before completing permissions for the six-chip system.
The bring-up service now expects six; initialization and FP16 matrix checks pass
on all six chips. The earlier reload encountered a Linux `xas_find_conflict`
fault during a file-cache read; its cause has not been established. Recovery
and validation evidence is under `../runs/muse-opt-v5` and
`../results/muse-opt-v5/stream-validation-summary.json`.

Decode attribution after recovery measures about 285 ms/token in expert
projection and its host gathers, within 322 ms/token for the full MoE block.
Attention inputs, cache/scores and output measure about 70, 43 and 98 ms/token.
These timings synchronize section boundaries and include profiling overhead;
they are not an end-to-end throughput benchmark. Profiling was removed afterward.

A resident candidate with reusable rank-local pinned transfer buffers passed
exact-logit and generated-text checks but changed three-run median short decode
from 2.009 to 1.984 tok/s. It was rejected, and the original Python patch was
restored with the same worker PID and weight storage identity. An isolated
32-channel shard-alignment probe also passed exact output checks at 8/137/512
routed rows, including empty expert groups. Its largest decode shards took about
7% less time, but 512-row gate/up prefill was about 4.5% slower. This microbenchmark
uses synthetic eight-expert banks and the unpromoted flattened kernel; it is not
a full-model speedup. The live layout remains on 128-channel shard alignment.
Evidence is in `decode-profile.json`, `staged-transfer-performance.json` and
`balanced32-qualification.json` under `../results/muse-opt-v5`.

### Reusing prompt prefixes

`muse_serve` and `muse_deploy` accept `--prefix-cache` with bounded attention.
After a successful request, the worker keeps one prompt prefix ending at a
prefill-chunk boundary. An identical prompt shorter than one chunk can also
reuse its cached logits. Extensions of partial chunks are recomputed, preserving
the uncached suffix's GEMM and attention shapes. Generated tokens are never
included in the retained prompt prefix.

Full-attention KV shares its existing append-only allocation. Sliding attention
retains a separate copy of only the valid window, then restores it before reuse;
this remains correct after decode has moved the active sliding buffer. Placement
reserves space for old and replacement sliding snapshots to coexist. The total
context limit and FP16 KV precision are unchanged. Cancelled/failed requests and
resident model changes discard the prefix. Health reports `prefix_cache`, and
generation logs report `cached_prompt_tokens`; cold throughput and reused-prefix
latency must be reported separately.

Runtime `ef5ef31` now serves both ports with prefix reuse enabled. All 358 frozen
runtime files and 62 packed-cache layers verified. The placement plan includes
the sliding snapshots and the existing 3 GiB reserve per device; it retains the
98,304-token capacity, original graph patch and established native W4 backend.

The controlled 2,127-token prompt reused 2,048 tokens on its second request:

| Measurement | Cold | Cached prefix |
| --- | ---: | ---: |
| First streamed text | 66.758 s | 3.016 s |
| Prefill time | 66.753 s | 3.010 s |
| Total HTTP response | 74.483 s | 10.838 s |
| Decode rate | 1.942 tok/s | 1.917 tok/s |

Both requests produced the same 16 output tokens, usage and finish reason.
Changing the suffix while retaining those 2,048 tokens also matched a cold run
exactly: prefill fell from 66.496 to 3.106 seconds. This is avoided prompt work,
not an improvement in cold prefill throughput or decode. An identical seven-token
prompt reused its cached logits and delivered first text in 7.75 ms.

Direct Kilo streaming returned `OK` in both runs. Its 689-token prompt reused
512 tokens, reducing prefill from 21.080 to 5.041 seconds. CLI wall time was
43.442 seconds cold and 25.208 seconds warm; output counts differed (38 versus
34), so this is an integration smoke rather than a controlled total-latency
comparison. HTTP usage still reports the full prompt token count; cache hits
are visible in server generation logs.

All 112 focused Muse tests pass. The prefix test additionally compares exact
CPU logits across aligned extensions, divergent prompts, partial chunks and
sliding-window rollover. Hardware checks cover exact streaming completion/chat
responses and changed suffixes, plus cancellation followed by a successful cold
request. Failed/cancelled requests and resident changes invalidate the cache.
The cache retains one prefix and serving remains single-request. Full-model
96K retrieval and quality remain unmeasured. Evidence is under
`../results/muse-opt-v6`, including `stream-validation-summary.json`,
`prefix-changed-suffix.json`, `kilo-prefix.json` and `final-health.json`.

Additional decode/prefill candidates were evaluated without replacing the live
kernel. Asynchronous pinned transfers passed parity but reduced median decode
from 2.003 to 1.884 tok/s. Mask-based W4 unpack measured 1.993 to 2.020 tok/s;
that small, unbracketed gain was insufficient to promote. Wider Cube tiles and
a corrected tile-loop variant passed synthetic operator checks with negligible
decode benefit. Dense input padding also produced no useful gain. A 1,024-token
prefill chunk was exact but improved isolated whole-layer time by less than 1%;
a 2,048-token chunk changed some FP16 outputs. A direct peer-copy probe failed
with an SDMA error despite reported peer capability; the serving worker remained
functional. None of these candidates is active. `experiments.json` records the
results and their scope.

### Reducing decode launches

`layer_decode_graphs.py` captures input normalization with QKV, attention output
with the first residual and router, and expert combination with the final
residual. Cross-graph tensors retain fixed storage until graph invalidation.
Prefill keeps its existing path. The resident controller also qualifies and
rolls back `FastMuseLayer.__call__` replacements, allowing whole-layer graph
experiments without reloading weights.

Single-token grouped attention skips construction/application of its all-false
mask: the append-only full cache contains past/current positions, and the sliding
cache retains at most the previous window minus one positions. Multi-token and
injected-operator paths retain explicit masks. FP16 attention arithmetic, KV
precision, expert selection and context capacity are unchanged. CPU tests compare
exact masked/unmasked outputs with poisoned unused cache capacity, and emulate
static graph storage across changed inputs, requests and graph recapture.

The expanded-graph patch also submits each expert rank once for both projection
phases. A barrier gathers the gate/up shards before down projection; down reuses
the rank's existing device route IDs instead of copying them from the host again.
Failures abort the barrier and drain all rank futures before cache cleanup or
resident switching. In one resident worker, three-repeat short decode medians
were 2.017 tok/s before, 2.093 tok/s with the combined patch, and 2.002 tok/s after
restoring the old path (4.15% over the mean of the baseline medians). Responses
and resident probe logits matched exactly. The earlier isolated layer improvement
of 11–13% did not transfer directly to full-model throughput; expanded graphs
plus mask removal alone measured only about 1.4% on the short full-model test.
Evidence is under `../results/muse-opt-v7`.

### Function-tool request compatibility

Chat completions accept OpenAI `tools`, `tool_choice: "auto"` (the default) or
`"none"`, `parallel_tool_calls`, assistant `tool_calls`, and tool-result messages
with `tool_call_id`. Definitions reach Muse's native chat template. Incoming JSON
argument strings become mappings as that template requires. Generated native
ATEM calls become API `tool_calls` with JSON arguments and `finish_reason:
"tool_calls"`; text and reasoning retain their streaming behavior. Function-call
deltas are emitted once the generated call is complete, so partial ATEM markup
is never sent as a tool invocation. The client executes tools.

`tool_choice: "required"` and named forced choices return an explicit 400 because
constrained tool selection is not implemented. Tool fields on text-completion
requests also return 400. This compatibility fix does not drop the client's tool
definitions or change the Kilo configuration. Tests cover the reported ordinary
chat request with tools, streamed calls, typed arguments, history round trips,
and the existing disconnect/cache behavior.
