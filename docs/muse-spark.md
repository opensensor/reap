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
context path. Four-query subchunks bound temporary attention score and softmax
storage. Query heads are grouped against 16 KV heads without expanding keys and
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
