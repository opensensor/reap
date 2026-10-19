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

Runtime `ef5ef31` qualified both ports with prefix reuse enabled. All 358 frozen
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

Runtime `2765412` serves ports 8001 and 8002 with the combined decode patch and
function-tool compatibility. All 62 packed-cache layers loaded; the frozen image
contains 360 verified files. The 127 focused tests pass. Live SSE completion/chat
messages, reasoning, usage and finish reasons match the earlier reference exactly.
Long-prompt cold prefill remains **31.892 tok/s**, with **1.993 decode tok/s** after
2,127 prompt tokens. Reusing 2,048 prompt tokens reduces first text from **66.698
to 2.892 seconds** and total response time from **74.222 to 10.479 seconds**, with
identical output. These prefix gains remain separate from raw decode throughput.

A live `what are you?` request with `tools` and `tool_choice: "auto"` returned a
normal answer instead of 422. Another request emitted `lookup({"query":"abc"})`
as an API tool call; a matching tool-result message produced a final answer
containing the supplied result. These are interface smokes, not a broad tool-use
quality evaluation. Additional benchmark requests were stopped to release the
single-request worker for interactive use. The latest deployment did not repeat
the hardware disconnect or complete the changed-suffix cold/warm comparison;
CPU tests and the previous deployment's checks remain available. Evidence is in
`tools-validation-summary.json`, `stream-validation-summary.json`,
`pipeline-combined-performance.json` and `final-health.json` under
`../results/muse-opt-v7`.


### Shared pinned prefill and host-driver investigation (2026-10-18)

Patch `3612570` now runs in the existing `2765412` process on ports 8001/8002.
Prefill copies each expert phase's activations and routes into one reusable pinned
host snapshot shared across all six ranks. Uploads remain concurrent. Every rank
completes its CPU result copy before reuse; operator errors synchronize that rank
and all futures are drained before propagating the failure. Separate rank pools
have separate staging buffers. The original MoE arithmetic and decode path remain
unchanged. Weight storage identities and the server PID were preserved.

| Measurement | Baseline | Shared pinned prefill |
| --- | ---: | ---: |
| Cold prefill, 400 tokens | 49.011 tok/s | **59.906 tok/s** |
| Cold prefill, 2,127 tokens | 35.376 tok/s | **55.565 tok/s** |
| First text, 2,127 tokens | 60.130 s | **38.284 s** |
| Decode after 2,127 tokens | 2.002 tok/s | 2.002 tok/s |

The 400-token result uses three repeats per candidate and baseline measurements
before and after: **22.23%** higher prefill throughput.
The 2,127-token result is one same-process cold comparison, with zero cached prompt
tokens in both runs. Each streamed completion matches the earlier reference
exactly. A warm request reuses 2,048 tokens and returns first text in
**2.758 seconds**. Client disconnect, subsequent exact completion,
and repeated reset/recapture checks pass. There are **132 passing Muse tests**.
Full-model 96K retrieval/quality and concurrent request scheduling remain untested.

The faster pageable-sharing experiment was rejected after `aclrtMemcpy` error
107017 during repeated resident probes. Kernel tracing exposed repeated partial
pins: **19 of 65 requested pages** from a valid 256 KiB anonymous buffer. The
vendor wrapper unpins the partial result and retries the whole range five times,
then reports zero and returns `-EFAULT`; its log obscured the partial progress.
This does not establish why Linux stops at that page. Serializing uploads lost
the performance gain. Shared pinned staging passed the checks above without new
Ascend kernel errors and retains concurrent uploads.

The separate host-driver fork has diagnostic commit
[`5f2a44a`](https://github.com/opensensor/ascend-310p-host-driver/commit/5f2a44a):
raw negative errors and exhausted partial-pin results are logged, with a targeted
BPF trace and pin-ownership regression tests. All 18 driver tests and verification
of 31 Linux 7.0.0-38 modules pass. Those kernel changes are **not installed** and
are not a validated fix for the underlying partial-pin failure.

The frozen next-start runtime is `runtime-3612570`, with 425 verified files and
launcher `launch-muse-prefill-v8.sh`; the currently loaded base image stays
`2765412` to avoid reloading weights. Native `muse_w4_v2_k1024`, context 98,304,
prefill chunk 512, tools and streaming remain enabled. Rollback uses
`launch-muse-tools-v7.sh` or its original resident Python patch. Evidence is in
`prefill-shared-safe-performance.json`, `shared-pinned-long-validation.json`,
`driver-debug.json` and `final-health.json` under `../results/muse-opt-v8`.


### Pinned decode and balanced W4 tiles (2026-10-18)

Patch `f7b08a0` adds reusable pinned decode input/result buffers, avoiding repeated
pageable allocations and copies at each expert phase. Blocking decode transfers
won the comparison with asynchronous transfers; prefill retains its independently
validated asynchronous uploads. Uneven rank widths have separate reusable result
buffers. All futures are drained on failure before any shared buffer is reused.
Identical rotary-phase tensors are shared across a stage, and layers without RoPE
skip their construction. Rank workers retain their initialized device context.

Native `muse_w4_v9_k1024` balances routed row/tile tasks across cores and replaces
arithmetic nibble extraction with masks and exact FP16 scaling. FP16 dequantization,
Cube products, split-K order and expert sums remain unchanged. Reproduction uses
`scripts/muse_native/build.py --unpack mask` with the pinned external Qwen/Catlass
inputs; the build checks the upstream blocks before transforming them and records
all source and binary hashes. External headers are not vendored here.

| Measurement | Same-round baseline | Deployed candidate |
| --- | ---: | ---: |
| Short decode, median of three | 2.092 before / 2.091 after | **2.183 tok/s** |
| Decode after 2,127 prompt tokens | 1.997 | **2.085 tok/s** |
| Cold 2,127-token prefill, one pair | 42.380 | 41.980 tok/s |
| First text, cold / 2,048-token cached prefix | 50.194 s cold | 50.672 / **2.735 s** |

The repeated short decode gain is **4.38%** over the mean of the baseline medians.
Long-prompt outputs match the reference exactly, including all 16 SSE text chunks.
Disconnect recovery and three additional reset/recapture cycles pass. There are
**132 passing Muse tests**. No prefill gain is claimed for this change: the
55.565 tok/s measured in the previous round was not reproduced in this round's
control. Full-model 96K retrieval/quality remains unmeasured; configuration and
KV allocation limits remain 98,304 tokens with 512-token prefill chunks.

The device trace covers three decode tokens. The seven staged dense FP16 matrix
operations consume about 184 ms per token; W4 kernel durations sum to 2.216 s
across all six devices over those three tokens. Cross-device and host timings
must not be added as serial wall time. FP32 router caching and combined local
Q/K/V/gate projection did not improve on the simpler phase-cache candidate.
Asynchronous shared pinned decode regressed. Reusing kernel masks across row/tile
tasks added no reliable gain and is not selected.

A separate dense-projection experiment split Q and attention-gate output channels
across all six devices. Eight changing real-weight inputs matched exactly; a
microbenchmark reduced their combined latency from 1.50 to 1.03 ms. Full-model
resident qualification and three exact-output repetitions reached **2.302 tok/s**
versus 2.179 tok/s controls (**5.66%** above the otherwise optimized runtime).
However, retaining duplicate projection shards left only about **707 MiB** free
on the tightest chip. This variant is **not deployed**: it consumes the memory
needed for the long-context workspace. The next implementation should shard these
weights during loading, rather than retaining both the original owner matrices
and distributed copies, and must qualify prefill as well as decode. The prototype
is retained in `../runs/muse-opt-v9/dense-tp.py` for that work.

The frozen next-start image is `runtime-f7b08a0` (425 verified files), with launcher
`launch-muse-decode-v9.sh`. The current process retains base image `2765412` and its
weight identities; ports 8001/8002, native tool handling and streaming remain in
place. Rollback uses `runtime-3612570` / `launch-muse-prefill-v8.sh` and native
`muse_w4_v2_k1024`. No kernel modules were replaced. Evidence is under
`../results/muse-opt-v9`, including `final-decode-performance.json`,
`long-validation.json`, `dense-tp-performance.json` and `trace/summary.json`.
The live identity request with tools, native `lookup` call, and tool-result
response also pass through SSE (`tools-validation-summary.json`). These interface
smokes do not establish broad coding or tool-selection quality.

### Load-time query/gate sharding (2026-10-18)

`8dc6f8f` adds optional `--tensor-attention` to the server and deployment launcher.
It requires tensor experts and bounded attention. Query and attention-gate output
channels are distributed across the existing six ranks during loading, including
packed-cache loads. The dense owner never receives a second copy of those two
matrices. Every rank keeps its query/gate rows together in NZ format; ordered
host gathers reconstruct queries followed by gates. FP16 weights, reductions,
routing and KV precision are unchanged.

Decode reuses the expert pool and graph lifecycle. Prefill uses the same shards
with dynamic row counts and bounded pinned staging. Futures are drained on
failure before host buffers can be reused. Resident storage identity includes
all six query/gate shards. The memory planner redistributes their existing
bytes and rebalances the remaining dense/KV stages to
`0/5/17/30/43/57/62`; it preserves 98,304-token context and the 3 GiB reserve.
This removes the previous prototype's 16,642,998,272 duplicate weight bytes.

The frozen `runtime-8dc6f8f` contains 427 verified files and uses the unchanged
`muse_w4_v9_k1024` native backend. The new launcher is
`/srv/ai/storage/experiments/muse-contribution128-w4/launch-muse-dense-v10.sh`;
`launch-muse-decode-v9.sh` remains the rollback. All 62 packed layers were cache
hits. After capture, the tightest chip had approximately 3.6 GiB free instead of
the duplicate-weight prototype's approximately 707 MiB.

Validation includes 138 focused tests, real layer-0 query/gate projections at
1/7/128/512 tokens with changing inputs and graph resets, and exact comparisons
against saved pre-reload outputs at every one of the 62 layers for four-token
prefill plus cached single-token decode. Startup also compares all logits before
and after activating the resident graphs.

| Measurement | Previous deployment | Load-time sharding |
| --- | --- | --- |
| Short decode, median of three | 2.184 tok/s | **2.313 tok/s** |
| Decode after 2,127 prompt tokens | 2.082 tok/s | **2.204 tok/s** |
| Cold 2,127-token prefill | 49.753 s | 50.633 s |
| Repeated-prefix request, 16 output tokens | 10.028 s | 9.624 s |

Short decode improves 5.91% in this before/reload comparison. The same-process
prototype's previous balanced controls independently measured a 5.66% gain.
Cold prefill is slightly slower; no prefill gain is claimed. The real projection
microbenchmark also showed this tradeoff: query/gate decode fell from about
1.54 ms to 0.89 ms, while the 512-token operation rose from about 5.38 ms to
6.31 ms. Full-model text and usage match the saved short/long completion fixtures.
Evidence and experiment scripts are under `../results/muse-opt-v10` and
`../runs/muse-opt-v10`, mirrored remotely under `opt-v10`.

A follow-up host-affinity trial compared three repetitions per setting with
restored-default controls. Pinning rank threads to CPUs 0–5 and the inference
thread to CPU 6 reached 2.386 tok/s versus 2.311/2.310 tok/s controls (+3.24%).
Spreading those threads across CPUs 0/4/8/12/16/20/24 measured 2.295 tok/s and
was rejected. Pinning multi-token prefill also increased its observed cold time
to 51.883 seconds, so the deployment override is limited to single-token decode.
Multi-token prefill restores the original thread masks.

The resident patch defaults to no affinity override. A frozen deployment copy
can append `CPU_AFFINITY = (0, 1, 2, 3, 4, 5, 6)` for this 3970X host. It validates
all target masks before changing any thread, restores changes on syscall failure,
and restores saved masks when the override is disabled. This setting is specific
to the measured host; it is not a general CPU layout recommendation.

The decode-only affinity trial used patch `b09148c`, hot-loaded into the
`8dc6f8f` process. It measured **2.385 tok/s** short decode, **2.293 tok/s** after
2,127 prompt tokens, 50.708-second cold prefill, and 9.406-second repeated-prefix
requests. The overall short-decode gain over this round's starting deployment is
**9.20%**. These are single-request measurements; cold prefill remains slightly
slower than the original deployment, and no prefill throughput gain is claimed.
The optional-affinity checks bring the focused suite to **141 passing tests**.

That trial froze `runtime-b09148c` with 428 verified files, including the
configured frozen patch `layer_decode_graphs_affinity.py`. The primary launcher
is `launch-muse-affinity-v10.sh`; `launch-muse-dense-v10.sh` retains sharding with
default CPU scheduling, and `launch-muse-decode-v9.sh` retains the pre-sharding
runtime. The active configured patch SHA256 is
`69746f2f574ffef997d1e094a8df7ff99811d5500bf9f64e195e571db0e91570`.

### Prefill staging after profiling (2026-10-18)

The next trace sampled the first four layers of a 512-token prefill chunk.
Summed host self durations included 132.6 ms in concatenation and 60.5 ms in
fill operations; the six-device native W4 tasks summed to 1,041.0 ms. Device and
worker times overlap and must not be read as serial request wall time. The trace
identified repeated large host allocations/copies around expert projection.

Patch `bae3318` copies routed activations and metadata directly into reusable
pinned staging. It gathers expert outputs into separate reusable pinned up/down
buffers with `torch.cat(..., out=...)`, then uses the up gather directly as the
down-phase input. This removes a pageable `.cpu()` temporary, repeated gather
allocation/initialization and a redundant host copy. Global deterministic
execution and uninitialized-memory policy are unchanged. All rank futures finish
before a gather can be overwritten; the final upload is blocking. Tests vary
inputs, grouped/routed metadata and row counts while checking numerical ordering,
buffer reuse and bounded cache cardinality.

Cold prefill qualification, with 2,127 prompt tokens and identical completions:

| Trial | Control before | Candidate | Control after |
| --- | --- | --- | --- |
| Reusable gathers | 50.648 s | 41.564 s | 51.320 s |
| Direct pinned staging plus gathers | 51.069 s | **35.574 s** | 50.780 s |

The combined prefill change improves throughput **43.15%** over the mean of its
surrounding controls. A separate final-source run reproduced **35.567 seconds
(59.802 tok/s)**. Final short-decode median is **2.380 tok/s**, versus this turn's
2.184 tok/s starting point (**8.98%** faster); decode after 2,127 tokens is
**2.291 tok/s**. A repeated-prefix request takes **9.136 seconds** total. These
remain single-request measurements; the long-prompt trial uses one request per
control/candidate position, followed by a separate final-source repeat.

The selected next-start image is `runtime-bae3318`, containing 428 verified files.
`launch-muse-prefill-v10.sh` activates native `muse_w4_v9_k1024` and the configured
resident patch with decode CPU affinity. Its SHA256 is
`2a3f1bad1bc4ae7fb0fbd164d0ed78338d2351284454a022729646343a4f9a44`.
The current process retains the `8dc6f8f` load-time layout and hot-loads this patch.
`launch-muse-affinity-v10.sh` rolls back the prefill transport change, while
`launch-muse-decode-v9.sh` restores the pre-sharding runtime. Context remains
98,304 tokens, with the same three-GiB reserve. The focused suite passes all
**141 tests**. Trace CSVs and summaries are in `../results/muse-opt-v10/trace-prefill`;
controlled trials are `reuse-prefill-performance.json`,
`direct-prefill-performance.json` and `final-performance.json`.

### Fine expert partitions and affinity cleanup (2026-10-18)

`ce9b4aa` partitions expert outputs at the physical 32-channel NZ boundary.
For six ranks and 4,096 channels, the widest shard falls from 768 to 704
channels; the other four shards have 672 channels. Gate/up shards retain both
corresponding channel ranges, and the memory planner uses the same partitioning.
Every weight and reduction is preserved. The packed-cache ABI, 98,304-token
context, 512-token prefill chunks, and 3 GiB reserve are unchanged.

The direct-launch W4 kernel accepts these complete NZ tiles. The vendor reference
binding still requires 128-channel output alignment, so reference/fallback calls
pad only their temporary operands and crop before split-K accumulation. Serving
retains the narrow resident banks. Native `muse_w4_v11_k1024` also carries the
core's next tile assignment across nonempty prefill expert groups, avoiding a
repeated partial tile batch on the same cores. Improving the old 640-channel
shards alone gave only a 0.3% full-model prefill gain because the wider ranks
still set the completion time.

The first full-model comparison, before the affinity cleanup below, measured:

| Measurement | Previous layout | Fine partitions |
| --- | ---: | ---: |
| Short decode, median of three | 2.393 tok/s | **2.473 tok/s** |
| Cold prefill, 2,127 tokens | 35.549 s | **34.175 s** |
| Decode after that prompt | 2.272 tok/s | **2.335 tok/s** |
| Warm request, 2,048 cached tokens | 9.191 s | **8.855 s** |

All 124 saved layer outputs (four-token prefill and cached one-token decode
through all 62 layers) matched exactly. Standalone qualification covers the
672/704/1344/1408-channel banks, changed inputs and routes, grouped prefill,
empty groups, and the padded vendor reference. Streaming, cancellation recovery,
repeated recapture, and native tool call/result handling also passed on this
layout. These are single-request measurements and small correctness fixtures,
not a broad coding or full-model 96K retrieval evaluation.

A subsequent four-thread prefill experiment exposed an affinity lifecycle bug:
the startup decode probe could leave the future server thread pinned to CPU 6.
New API and OpenMP workers inherited that single-CPU mask. A live CPU sample
showed all four compute threads sharing CPU 6 at about 24% CPU each; cold prefill
regressed from 34.035 to 171.358 seconds. `66b641c` restores saved rank/caller
masks during request-cache cleanup, resident-probe cleanup, and pool shutdown.
It preserves prefix snapshots during successful request cleanup and tolerates
already-exited streaming threads. The CPU suite now passes 148 tests, including
probe failure and parent-mask inheritance regressions.

Other rejected trials included reusable decode gathers, alternative SMT/CPU
placements, captured host copies, and serial asynchronous rank launches. The
serial launch trial lost 25% decode throughput. A wider one-row Cube reduction
was exact but had no consistent standalone speedup. Retaining one expert row per
core without a tile barrier failed the isolated correctness gate and was never
loaded into the server. Adding the barrier restored exactness but gave mixed,
small gains. The fresh trace's first-position stage-copy outliers include
profiler startup overhead; they are not steady-state transport measurements.
Evidence and experiment scripts are under `../results/muse-opt-v11` and
`../runs/muse-opt-v11`.

The first reload of the affinity cleanup stopped on the packed-cache checksum
for layer 21. Two ordinary reads returned the same incorrect SHA256;
`POSIX_FADV_DONTNEED` followed by a fresh read returned the original receipt's
SHA256. The other 61 cached layers also verified. No files were rewritten and no
integrity checks were bypassed. This isolates the observed mismatch to cached
host pages, but does not identify its cause or establish a driver fix. The
failed startup log and hashes are retained in `pagecache-integrity-event.json`
and `cache-audit.json` alongside the experiment results.

After the cleanup reload, all 124 layer outputs again matched exactly. Short
median decode measured 2.473 tok/s; 2,127-token cold prefill took 33.935 seconds
(62.68 tok/s). The guarded four-thread trial used the full 0–63 CPU mask and took
33.513 seconds, versus 34.025/34.051-second one-thread controls. Its 1.57% prefill
throughput gain accompanied a roughly 0.9% decode regression, so the selected
configuration keeps one CPU thread. This trial establishes that the earlier
single-CPU inheritance failure is fixed, not that more CPU threads are a useful
serving optimization.

The final serving image is `runtime-66b641c`, launched by
`launch-muse-affinity-v11.sh`. Both ports share worker 1068710 and the original
resident patch SHA256 `2a3f1bad1bc4ae7fb0fbd164d0ed78338d2351284454a022729646343a4f9a44`.
All 428 frozen files verified. The final idle check found all 154 host threads
back on CPU mask 0–63 and no new Ascend kernel errors since this successful
reload. SSE cold/warm output, disconnect recovery, and three recapture cycles
passed. Warm first text arrived in 2.449 seconds.

A real Kilo session with only read/edit enabled completed read → edit → final
confirmation in 160.598 seconds. It changed a temporary `add(a, b)` implementation
from subtraction to addition, and an independent execution returned 5 for
`add(2, 3)`. The three requests took 79.852 / 54.858 / 25.764 seconds server-side;
the two follow-ups each reused 2,048 prompt tokens. This is a tiny integration
check, not evidence of general coding quality. The current API usage object does
not expose cached-token detail, so Kilo's displayed cache-read count remains
zero even though the server records prefix reuse.

The user's global Kilo model now advertises tool support. A `muse-code` primary
agent enables read/edit/write/glob/grep/bash, with other tools disabled and
existing permission rules preserved. The separate `muse-chat` entry remains
text-only. The six-tool agent was configuration-validated; the executed edit
smoke used the isolated two-tool agent. A private timestamped config backup was
created. Larger coding tasks, the complete six-tool workflow, and further decode
optimization remain to be evaluated.

### Ordered row ownership and Kilo usage reporting (2026-10-19)

A new trace samples three decode positions after a 2,127-token prompt. Summed
six-device W4 task duration is 2.093 seconds and dense NZ matmul duration is
0.574 seconds. These sums overlap across devices. Blocking host copies include
pending device compute; the roughly 16 ms vocabulary-logit copy waits are not
pure PCIe transfer time. Startup also distorts the trace's first position.
Unprofiled controls measured 2.323/2.333 tok/s decode after the prompt.

`ef23d84` lets each core retain one routed row's expert geometry and unpack
tables across all output tiles when the routed row count equals the core count.
The pinned Qwen header transformation orders the previous MTE3 store before
MTE2 reuses shared buffers. The earlier unordered version failed isolated
correctness checks; it was never used for serving. The ordered version matches
all 21 tested shape/mode cases and 56 additional changing-input/route cases,
including grouped prefill and narrow 672/704/1344/1408-channel banks. Its compiled
C++ input and transformed header match the repository sources byte for byte.

Three-repeat short-decode medians were 2.438 / **2.490** / 2.441 tok/s for
control / candidate / control: **2.05% higher throughput**. After a 2,127-token
prompt, decode was 2.314 / **2.369** / 2.327 tok/s. Cold prefill was essentially
unchanged at 33.982 / 33.936 / 33.923 seconds. A separate pinned nonblocking
input-copy experiment preserved output but lost 6.41% decode throughput and
was reverted. The selected Python patch is unchanged.

`151ee6e` reports `prompt_tokens_details.cached_tokens` in both normal responses
and final SSE usage frames. Usage is captured before releasing the generation
lock, so cleanup or a later request cannot overwrite the count. The CPU suite
passes 150 tests, including request-local usage isolation on both response paths.
Trace tables, receipts, and controlled measurements are in
`../results/muse-opt-v12`; scripts are in `../runs/muse-opt-v12`.

The selected frozen runtime is `runtime-ef23d84`, using native
`muse_w4_v12_rows_ordered_k1024` and the unchanged configured Python patch
`2a3f1bad1bc4ae7fb0fbd164d0ed78338d2351284454a022729646343a4f9a44`.
`launch-muse-ordered-v12.sh` is the next-start launcher. All 428 files verified
before launch; all 62 cached layers loaded without checksum failures. The new
worker reproduced all 124 saved layer outputs exactly. Final short-decode median
was 2.486 tok/s. Cold prefill took 34.495 seconds for 2,127 tokens; long-prompt
decode was 2.349–2.370 tok/s and the repeated-prefix request took 8.801 seconds.
These post-reload values are separate measurements from the controlled native
comparison above. SSE usage matched server cache counts, and cancellation
recovery plus three recapture cycles passed.

The actual six-tool `muse-code` configuration then completed a read → edit →
confirmation smoke in **222.768 seconds**. Its first request had 4,535 prompt
tokens and took 84.628 seconds to prefill. Follow-up requests reused 4,096 and
4,608 tokens; Kilo now displayed those exact cache-read counts. Only read/edit
were exercised, and an independent execution confirmed `add(2, 3) == 5` after
the edit. This larger prompt differs from the earlier two-tool smoke, so their
end-to-end times are not an optimization comparison. The integration works,
but this latency still falls short of a responsive coding assistant.

A follow-up grouped-prefill trial retained geometry/unpack tables across each
core's assigned output tiles, with the existing per-tile path preserved for
small K where output staging overlaps those tables. All 29 shape/mode checks
and 56 changed-input/route checks matched exactly. Standalone grouped kernels
improved roughly 1–3%, but full-model cold prefill was 34.229 / 34.111 / 34.284
seconds for control / candidate / control: only **0.43%** higher throughput.
Short decode fell about **0.66%**. The trial was rejected and the selected native
backend restored through its exact-logit gate; its evidence is retained in
`retained-groups-performance.json`.

The final idle check verified all 428 frozen files and both serving ports on
worker 1154211. All 151 remaining threads had CPU mask 0–63; no new Ascend
kernel errors were found since this reload. Resident weight storage identities
were unchanged through the trials. No driver modules were replaced.

### Vocabulary sharding and pinned rank results (2026-10-19)

The next profile covers layers 0–3 in the 512-token chunk starting at position
3,584 of a synthetic 4,254-token completion fixture. Unprofiled cold prefill
controls took 74.507 and 73.962 seconds. Summed device durations included
150.8 ms of W4 tasks and 75.0 ms of FP32 softmax; devices overlap, and blocking
host copies include pending compute. FP16-output softmax and half-to-float
softmax trials changed values, so neither replaces the existing FP32 path.

`c061ea5` adds optional `--tensor-output` to the serving/deployment commands.
Vocabulary rows are partitioned at 32-channel boundaries and loaded directly
into six NZ matrices. The original owner matrix is omitted. Each rank keeps the
existing FP16 projection, output multiplier and tanh softcap arithmetic. One
ordered CPU gather returns independent logits, so later requests cannot overwrite
a prefix snapshot. Decode captures the projections in the existing graph pools;
all futures drain on failure and reset invalidates the head graphs. Resident
storage identity includes every vocabulary shard.

The memory planner redistributes the existing 3,310,354,432 head bytes; it adds
no duplicate weight storage. Dense/KV stages become `0/4/15/26/38/50/62` while
retaining 98,304-token context, 512-token prefill chunks and the 3 GiB reserve.
The full 202,048-row hardware probe matched every logit at 1/4/7 tokens with
changing inputs and recapture. Single-token projection plus CPU result transfer
fell from approximately **16.6 ms to 3.3 ms**. This is an isolated head measurement,
not a fivefold full-model speedup.

The same commit copies prefill rank outputs directly into reusable pinned
buffers. Each rank/phase/shard width retains one row count; gathers finish before
reuse. Controlled cold 2,127-token prefill measured 34.021 / **33.513** / 34.004
seconds (**1.49%** throughput gain). A 4,254-token follow-up with 2,048 cached
tokens measured 42.381 / **41.818** / 42.417 seconds (**1.39%** gain). Short decode
was 0.20% lower in that trial. All completion outputs matched. The CPU suite
passes **153 tests**, including independent output lifetime, changed inputs,
recapture, complete future draining, shard identity and exact placement totals.
Evidence and scripts are under `../results/muse-opt-v13` and `../runs/muse-opt-v13`.

The frozen serving image `runtime-c061ea5` contains 362 verified source/operator
files and is launched by `launch-muse-output-v13.sh`. The native W4 backend remains
`muse_w4_v12_rows_ordered_k1024`; the configured Python patch SHA256 is
`df9ce19202fe49c45daceadfbf428e3cf32f19c0b8cec2934723527caaabdd2c`.
All 62 packed layers loaded from cache without checksum failures. Every one of
the 124 saved layer outputs matched exactly after the layout change.

Final short-decode median was **2.568 tok/s**, versus the preceding same-fixture
controls' 2.482/2.473 tok/s. This before/reload comparison includes both the head
change and the new placement; it is separate from the isolated head benchmark.
Cold 2,127-token prefill took **33.765 seconds**, decode was **2.447 tok/s**, and
the 2,048-cached-token request took **8.533 seconds** total. Cold 4,254-token
prefill took **73.460 seconds**, versus 74.507/73.962-second prior controls;
decode was **2.424 tok/s**. Reusing 4,096 tokens reduced that request to
**10.042 seconds** total. All completion text and usage checks passed.

SSE returned the exact 16-token fixture, with cached first text in 2.435 seconds.
Cached usage, disconnect recovery and three recapture cycles passed. The final
idle check found both ports on worker 1246486, all 362 frozen files intact,
all 151 host threads restored to CPU mask 0–63, unchanged resident storage
identities during qualification, and no new Ascend kernel errors since startup.
No driver modules were changed. Rollback remains `launch-muse-ordered-v12.sh`.

The actual six-tool Kilo agent again completed read → edit → confirmation and
an independent check of the edited function returned 5 for `add(2, 3)`.
This run took **232.586 seconds** and generated 314 tokens, versus the previous
222.768 seconds and 280 tokens. The initial prompt again contained 4,535 tokens;
its prefill took 83.391 seconds, and follow-ups reused 4,096/4,608 tokens.
The generated lengths differ, so this is a functional integration result, not
a controlled end-to-end speedup. It does **not** establish responsive Kilo use
or broad coding quality. Only read/edit actions were exercised.

### Peer allocation diagnosis and transport qualification (2026-10-19)

Fresh-process probes reproduced the device 2 → 1 SDMA translation failure
(`507013`) with ordinary allocation, and with expandable allocation plus
successful explicit peer enablement in both directions. Device 2 → 0 also
failed; 0 → 2 passed. Performing the successful reverse copy first did not
repair 2 → 0. These failures do not establish that all peer links are unusable.

A synchronous ACL copy exposed `devmm_d2d_pa_to_bar_dma`: the destination
physical address could not be translated into its PCIe BAR (`-EFAULT`). The
host driver's `devdrv_devmem_addr_d2h` rejects addresses outside the advertised
RX ATU ranges. Allocating the destination with
`ACL_MEM_MALLOC_HUGE_FIRST_P2P` instead of ordinary tensor storage made the
2 → 0 asynchronous copy exact at 128 KiB and 4 MiB. No driver module change
was needed for this test. Peer capability and successful enablement alone do
not establish that an arbitrary tensor allocation is peer-addressable.

The loaded model had exhausted P2P huge pages on the inspected devices;
device 1 had only about 1.2 MB reported free in its P2P normal-page pool, and
even a 128 KiB P2P allocation failed there. With the idle model unloaded,
reserving a bounded 16 MiB arena on each device succeeded. All 36 directed
source/destination combinations then passed four exact 128 KiB copies each:
24 inter-card peer paths, six local paths and six same-card pinned-host
fallbacks. The first arena probe required a correction to initialize ACL
contexts before calling the native allocator; it stopped before any copy.

The six-rank transport microbenchmark used 15 repetitions, discarded four
warmups, and bracketed the candidate with host-transfer controls. Values below
are means of the six owner devices' medians; they are not model throughput.

| Operation | Rows × width | Host control | Peer candidate |
| --- | --- | ---: | ---: |
| Broadcast | 8 × 4096 | 0.488 ms | 0.515 ms |
| Gather | 8 × 4096 | 0.440 ms | 0.602 ms |
| Broadcast | 512 × 4096 | 1.274 ms | 2.105 ms |
| Gather | 512 × 4096 | 1.031 ms | 0.669 ms |

The peer path preserved every value but regressed decode-sized transfers and
broadcasts. It is not enabled for serving. The large-gather result warrants a
separate prefill experiment, with arenas reserved before loading weights and
the same-card fallback retained. Current serving continues to use pinned-host
transport. Evidence and standalone reproduction scripts are in
`results/muse-opt-v14` and `runs/muse-opt-v14` beside the checkout, mirrored on
Threadripper under `opt-v14`.

The subsequent reload correctly stopped at a layer-55 cache integrity failure.
Two cached reads agreed on the wrong hash, while a simultaneous `O_DIRECT`
disk read matched the receipt. Comparing both paths identified exactly one
changed byte and one changed bit: file offset 3,699,389,219 was `0x9c` in cache
and `0x98` on disk. Both versions of the affected 4 KiB page were preserved.
`POSIX_FADV_DONTNEED` followed by a fresh read restored the expected checksum;
no file bytes or receipts were rewritten and no integrity check was bypassed.
This is the second observed page-cache integrity incident, on a different layer
from the earlier layer-21 event. Its cause remains unresolved; a one-bit change
does not establish a DMA/driver fault or a RAM fault. The host exposes no EDAC
memory-controller counters at the inspected sysfs path.

### Constant-offset W4 specialization (2026-10-19)

`629e5a1` adds `--offsets eight` to the native builder. The biased symmetric
NZ layout represents zero as eight, so the specialized kernel replaces the
per-group offset broadcasts with one tile-wide FP16 addition of negative eight.
It omits offset loads/casts while preserving scale multiplication, Cube products,
FP16 partial-product rounding and the ordered FP32 split-K sum.

The separate `muse-w4-offset8-v1` ABI validates every offset tensor before its
first native use. Validation retains tensor identity and rechecks versioned
tensors after in-place changes. Inference tensors follow the runtime's existing
immutable-weight contract. The original `muse-w4-v1` ABI still supports tensor
offsets. Tests cover rejection before any native launch, changed offsets, valid
inference tensors and the generic backend's unchanged behavior. The full prior
suite plus the added integration test pass **155 tests**.

The guarded candidate passed 21 operator shape/mode cases and 56 additional
changing-input/route cases. A same-process full-model comparison then measured:

| Measurement | Control before | Offset-eight kernel | Control after |
| --- | ---: | ---: | ---: |
| Short decode, median of three | 2.573 tok/s | **2.622 tok/s** | 2.569 tok/s |
| Decode after 2,127 tokens | 2.429 tok/s | **2.471 tok/s** | 2.439 tok/s |
| Cold 2,127-token prefill | 34.714 s | 34.432 s | 34.035 s |

Short decode improves **1.97%** over the mean of the two control medians.
Long-prompt decode improves about **1.51%**. Prefill is effectively unchanged:
the candidate is 0.17% slower than the mean control, within the observed control
variation. Every saved completion in the comparison matched exactly. The trial
returned to the control before final qualification.

Final qualification selected `muse_w4_v14_offset8_guarded_k1024` in the frozen
`runtime-629e5a1` image. All 611 source/operator files verify, all 62 packed
layers loaded from cache, and all 124 saved layer outputs match exactly. The
selected launcher is `launch-muse-offset-v14.sh`; rollback uses
`launch-muse-output-v13.sh`. The current worker was started with the control
native and switched through the exact-logit gate; the launcher now selects the
qualified offset-eight native for subsequent starts.

The final short-decode median is **2.624 tok/s**. Cold 2,127-token prefill takes
**33.753 seconds**, followed by **2.473 tok/s** decode; the repeated-prefix
request takes **8.428 seconds** total. Cold 4,254-token prefill takes
**73.506 seconds**, followed by **2.437 tok/s** decode; its repeated-prefix
request takes **10.064 seconds**. These final measurements are separate from
the paired controls above. Saved completions remain exact.

Streaming returns 16 exact text chunks and reports 2,048 cached tokens on the
warm request, with first text in **2.394 seconds**. Disconnect recovery and
three recapture cycles pass. Both ports serve the same worker; all 151 host
threads have CPU mask 0–63 after cleanup, storage identities are unchanged,
and no new Ascend kernel errors were found since this reload. Context remains
98,304 tokens. The actual Kilo read/edit smoke was not repeated in this round;
its previous 232.586-second result remains the latest end-to-end evidence.
