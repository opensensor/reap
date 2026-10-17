"""CPU, text-only, resumable calibration of indexed unquantized Muse checkpoints.

This entry point loads one embedding/decoder/norm module at a time. It never
constructs a whole model or writes model weights. Requires native Muse classes
from a separately installed, compatible Transformers version.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import resource
import re
import tempfile
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from reap.models.muse_spark import collect_muse_pruning_metrics
from reap.metrics import OnlineStatsTracker
from reap.pruning_metrics import initialize_pruning_state


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _native_backend():
    try:
        from transformers.models.muse_spark import modeling_muse_spark as modeling
        from transformers.models.muse_spark.configuration_muse_spark import (
            MuseSparkTextConfig,
        )
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Muse streaming needs a Muse-capable Transformers environment; "
            "see docs/muse-spark.md for the tested revision."
        ) from exc
    return (
        modeling,
        MuseSparkTextConfig,
        create_causal_mask,
        create_sliding_window_causal_mask,
    )


class MuseShardSource:
    """Read indexed safetensors without materializing unrelated tensors.

    CPU tensors are mmap-backed. On leaving a module context its parameters
    become meta tensors, releasing source mappings even if the module object is
    retained. Nested materialization is rejected. Source dtype is preserved.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        config_bytes = (self.root / "config.json").read_bytes()
        index_bytes = (self.root / "model.safetensors.index.json").read_bytes()
        raw = json.loads(config_bytes)
        self.weight_map = json.loads(index_bytes)["weight_map"]
        text = raw.get("text_config", raw)
        if raw.get("quantization_config") or text.get("quantization_config"):
            raise ValueError("streaming calibration requires an unquantized checkpoint")
        self.modeling, config_class, self.full_mask, self.sliding_mask = (
            _native_backend()
        )
        self.config = config_class.from_dict(text)
        self.config._attn_implementation = "eager"
        self.config._experts_implementation = "eager"
        self.config.use_cache = False
        prefixes = [
            p
            for p in ("model.language_model.", "model.", "")
            if p + "embed_tokens.weight" in self.weight_map
        ]
        if len(prefixes) != 1:
            raise ValueError("expected exactly one Muse text embedding prefix")
        self.prefix = prefixes[0]
        if (
            len(self.config.layer_types) != self.config.num_hidden_layers
            or len(self.config.layer_rope_theta) != self.config.num_hidden_layers
        ):
            raise ValueError(
                "Muse per-layer attention configuration has the wrong length"
            )
        if set(self.config.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError("unsupported text attention type")
        # The native implementation uses one global RoPE object and a per-layer
        # enable/disable flag. Reject differing nonzero bases rather than silently
        # treating them as a different mathematical model.
        theta = self.config.rope_parameters["rope_theta"]
        if any(value not in (0, theta) for value in self.config.layer_rope_theta):
            raise ValueError("distinct nonzero per-layer RoPE bases are not supported")
        if self.config.rope_parameters["rope_type"] != "default":
            raise ValueError("streaming currently supports default Muse RoPE only")
        self.files = {}
        stats = {}
        for name in sorted(set(self.weight_map.values())):
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root) or path.suffix != ".safetensors":
                raise ValueError(
                    "shard paths must stay within the checkpoint directory"
                )
            st = path.stat()
            self.files[name] = path
            stats[name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        self.identity = {
            "source": str(self.root),
            "config_sha256": _digest(config_bytes),
            "index_sha256": _digest(index_bytes),
            "shards": stats,
            "transformers": importlib.metadata.version("transformers"),
            "modeling_sha256": _digest(
                Path(inspect.getfile(self.modeling)).read_bytes()
            ),
            "masking_sha256": _digest(
                Path(inspect.getfile(self.full_mask)).read_bytes()
            ),
            "runner_sha256": _digest(Path(__file__).read_bytes()),
            "adapter_sha256": _digest(
                Path(inspect.getfile(collect_muse_pruning_metrics)).read_bytes()
            ),
            "pruning_metrics_sha256": _digest(
                Path(inspect.getfile(initialize_pruning_state)).read_bytes()
            ),
            "metrics_sha256": _digest(
                Path(inspect.getfile(OnlineStatsTracker)).read_bytes()
            ),
            "config_class_sha256": _digest(
                Path(inspect.getfile(config_class)).read_bytes()
            ),
            "safetensors": importlib.metadata.version("safetensors"),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        }
        self.active = None
        self.load_log = []

    @contextmanager
    def _module(self, factory, prefix):
        if self.active is not None:
            raise RuntimeError("release the current module before loading another")
        self.active = prefix
        module = None
        try:
            with torch.device("meta"):
                module = factory()
            expected = module.state_dict()
            wanted = {prefix + key for key in expected}
            available = {key for key in self.weight_map if key.startswith(prefix)}
            if wanted != available:
                raise ValueError(
                    f"checkpoint/module keys differ for {prefix}: missing={wanted - available}, extra={available - wanted}"
                )
            values = {}
            for shard in sorted({self.weight_map[key] for key in wanted}):
                with safe_open(
                    self.files[shard], framework="pt", device="cpu"
                ) as reader:
                    for full_name in sorted(wanted):
                        if self.weight_map[full_name] != shard:
                            continue
                        value = reader.get_tensor(full_name)
                        key = full_name[len(prefix) :]
                        if value.shape != expected[key].shape:
                            raise ValueError(f"unexpected shape for {full_name}")
                        if value.dtype not in (
                            torch.float32,
                            torch.float16,
                            torch.bfloat16,
                        ):
                            raise ValueError(
                                f"unsupported source dtype for {full_name}: {value.dtype}"
                            )
                        values[key] = value
            size = sum(t.numel() * t.element_size() for t in values.values())
            module.load_state_dict(values, strict=True, assign=True)
            del expected, values, value
            module.eval().requires_grad_(False)
            self.load_log.append({"prefix": prefix, "weight_bytes": size})
            yield module
        finally:
            if module is not None:
                module.to_empty(device="meta")
            self.active = None
            gc.collect()

    def embedding(self):
        c = self.config
        return self._module(
            lambda: self.modeling.MuseSparkTextNormedEmbedding(
                c.vocab_size, c.hidden_size, c.pad_token_id, c.rms_norm_eps
            ),
            self.prefix + "embed_tokens.",
        )

    def layer(self, index):
        if not 0 <= index < self.config.num_hidden_layers:
            raise ValueError("layer index out of range")
        return self._module(
            lambda: self.modeling.MuseSparkTextDecoderLayer(self.config, index),
            self.prefix + f"layers.{index}.",
        )

    def final_norm(self):
        return self._module(
            lambda: self.modeling.MuseSparkRMSNorm(
                self.config.hidden_size, eps=self.config.rms_norm_eps
            ),
            self.prefix + "norm.",
        )

    def replay_kwargs(self, hidden, batch, layer_index):
        """Use native mask/RoPE builders with this layer's attention semantics."""
        positions = batch["position_ids"]
        builder = (
            self.sliding_mask
            if self.config.layer_types[layer_index] == "sliding_attention"
            else self.full_mask
        )
        mask = builder(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=batch["attention_mask"],
            past_key_values=None,
            position_ids=positions,
        )
        embeddings = None
        if self.config.layer_rope_theta[layer_index]:
            rotary = self.modeling.MuseSparkTextRotaryEmbedding(self.config)
            embeddings = rotary(hidden, positions)
        return {
            "attention_mask": mask,
            "position_ids": positions,
            "position_embeddings": embeddings,
            "past_key_values": None,
        }


def _prepare_batches(batches, config):
    result = []
    for batch in batches:
        ids = torch.as_tensor(batch["input_ids"], device="cpu")
        if ids.dtype != torch.long or ids.ndim != 2 or not all(ids.shape):
            raise ValueError(
                "input_ids must be a nonempty [batch, sequence] int64 tensor"
            )
        if ids.min() < 0 or ids.max() >= config.vocab_size:
            raise ValueError("token ID outside vocabulary")
        mask = torch.as_tensor(
            batch.get("attention_mask", torch.ones_like(ids)), device="cpu"
        )
        if (
            mask.shape != ids.shape
            or not ((mask == 0) | (mask == 1)).all()
            or not mask.bool().any(-1).all()
        ):
            raise ValueError(
                "attention_mask must be binary, token-aligned, with a valid token per row"
            )
        positions = torch.as_tensor(
            batch.get("position_ids", torch.arange(ids.shape[1]).unsqueeze(0)),
            device="cpu",
        )
        if (
            positions.dtype != torch.long
            or positions.shape not in ((1, ids.shape[1]), ids.shape)
            or positions.min() < 0
        ):
            raise ValueError(
                "position_ids must have shape [1 or batch, sequence] and nonnegative int64 values"
            )
        result.append(
            {
                "input_ids": ids.contiguous(),
                "attention_mask": mask.bool().contiguous(),
                "position_ids": positions.contiguous(),
            }
        )
    if not result:
        raise ValueError("calibration requires at least one batch")
    return result


def _batches_digest(batches):
    h = hashlib.sha256()
    for batch in batches:
        for key, value in sorted(batch.items()):
            h.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
            h.update(value.numpy().tobytes())
    return h.hexdigest()


def _atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _save_json(path, value):
    _atomic(path, lambda p: p.write_text(json.dumps(value, indent=2) + "\n"))


def _save_tensors(path, values):
    _atomic(
        path,
        lambda p: save_file(
            {k: v.detach().cpu().contiguous() for k, v in values.items()}, p
        ),
    )


def _rss_bytes():
    # Linux RSS (including resident mmap pages), not virtual address space.
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
        "SC_PAGE_SIZE"
    )


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _activation_hashes(output, stage, count):
    return {
        str(i): _file_digest(_activation_path(output, stage, i)) for i in range(count)
    }


def _activation_path(output, stage, batch):
    return output / f"activations-{stage}" / f"batch-{batch:05d}.safetensors"


def _retire_stage(output, stage, count):
    for index in range(count):
        _activation_path(output, stage, index).unlink(missing_ok=True)
    directory = output / f"activations-{stage}"
    if directory.exists() and not any(directory.iterdir()):
        directory.rmdir()


def _remove_interrupted_temporaries(output):
    """Remove only this writer's tempfile names inside its artifact directories."""
    directories = [output, output / "scores"]
    directories.extend(
        p
        for p in output.iterdir()
        if re.fullmatch(r"activations-(input|layer-\d{5}|final)", p.name)
    )
    for directory in directories:
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in directory.iterdir():
            if re.fullmatch(r"\.partial-[a-z0-9_]{8}", path.name) and path.is_file():
                path.unlink()


@torch.inference_mode()
def run_calibration(
    source, batches, output, *, stop_after_layer=None, resume=False, progress=None
):
    """Observe baseline text layers, atomically committing at layer boundaries.

    `stop_after_layer` is an exclusive count. Resume starts from the last complete
    layer; a partially processed layer is recomputed. Activation files are one
    batch each, and only current/next layer files coexist during a transition.
    """
    output = Path(output).resolve()
    if output == source.root or output.is_relative_to(source.root):
        raise ValueError("calibration output must be outside the source checkpoint")
    batches = _prepare_batches(batches, source.config)
    stop = (
        source.config.num_hidden_layers
        if stop_after_layer is None
        else stop_after_layer
    )
    if not 0 <= stop <= source.config.num_hidden_layers:
        raise ValueError("stop_after_layer must be between zero and the layer count")
    identity = {
        "source": source.identity,
        "batches_sha256": _batches_digest(batches),
        "schema": 1,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another calibration writer holds this output directory"
            ) from exc
        manifest = output / "progress.json"
        if manifest.exists():
            if not resume:
                raise ValueError("output already contains a run; use resume=True")
            state = json.loads(manifest.read_text())
            if state["identity"] != identity:
                raise ValueError(
                    "resume identity differs (checkpoint, inputs, backend, or runner)"
                )
            next_layer = state["next_layer"]
            expected_stage = (
                "final"
                if state["complete"]
                else f"layer-{next_layer - 1:05d}"
                if next_layer
                else "input"
            )
            if (
                not isinstance(next_layer, int)
                or not 0 <= next_layer <= source.config.num_hidden_layers
                or state["stage"] != expected_stage
                or state["batches"] != len(batches)
                or [row["layer"] for row in state["layers"]] != list(range(next_layer))
                or (state["complete"] and next_layer != source.config.num_hidden_layers)
            ):
                raise ValueError("invalid committed progress manifest")
            if state.get("initialized"):
                if (
                    _activation_hashes(output, state["stage"], len(batches))
                    != state["activation_hashes"]
                ):
                    raise ValueError("committed activation checksum differs")
                for row in state["layers"]:
                    if (
                        _file_digest(
                            output / "scores" / f"layer-{row['layer']:05d}.safetensors"
                        )
                        != row["score_sha256"]
                    ):
                        raise ValueError("committed score checksum differs")
            # Recover from a crash after commit but before retiring the old stage,
            # or discard partial outputs for a layer that must be recomputed.
            _remove_interrupted_temporaries(output)
            for directory in output.iterdir():
                match = re.fullmatch(
                    r"activations-(input|layer-\d{5}|final)", directory.name
                )
                if directory.is_dir() and match and match[1] != state["stage"]:
                    _retire_stage(output, match[1], len(batches))
        else:
            if resume:
                raise ValueError("no committed progress to resume")
            if any(p.name != ".lock" for p in output.iterdir()):
                raise ValueError("new run requires an empty output directory")
            state = {
                "identity": identity,
                "next_layer": 0,
                "stage": "input",
                "complete": False,
                "initialized": False,
                "batches": len(batches),
                "tokens": sum(int(b["attention_mask"].sum()) for b in batches),
                "layers": [],
            }
            _save_json(manifest, state)
        if not state["initialized"]:
            with source.embedding() as embedding:
                for i, batch in enumerate(batches):
                    _save_tensors(
                        _activation_path(output, "input", i),
                        {"hidden": embedding(batch["input_ids"])},
                    )
            state.update(
                initialized=True,
                activation_hashes=_activation_hashes(output, "input", len(batches)),
            )
            _save_json(manifest, state)
        if state["complete"]:
            return state
        if stop < state["next_layer"]:
            raise ValueError("stop_after_layer precedes committed progress")
        for index in range(state["next_layer"], stop):
            start = time.monotonic()
            stage = f"layer-{index:05d}"
            old_stage = state["stage"]
            metrics = None
            with source.layer(index) as layer:
                if source.config.mlp_layer_types[index] == "sparse":
                    metrics = initialize_pruning_state(layer.mlp.experts.num_experts)
                for batch_index, batch in enumerate(batches):
                    hidden = load_file(
                        _activation_path(output, old_stage, batch_index)
                    )["hidden"]
                    handle = None
                    if metrics is not None:

                        def observe(module, args, result):
                            collect_muse_pruning_metrics(
                                module, args[0], metrics, batch["attention_mask"]
                            )

                        handle = layer.mlp.register_forward_hook(observe)
                    try:
                        result = layer(
                            hidden, **source.replay_kwargs(hidden, batch, index)
                        )
                        _save_tensors(
                            _activation_path(output, stage, batch_index),
                            {"hidden": result},
                        )
                    finally:
                        if handle is not None:
                            handle.remove()
                    del hidden, result
                resident = _rss_bytes()
            report = (
                {}
                if metrics is None
                else {
                    k: v.mean if isinstance(v, OnlineStatsTracker) else v
                    for k, v in metrics.items()
                }
            )
            # Dense layers have no pruning scores; an explicit scalar marks them.
            _save_tensors(
                output / "scores" / f"layer-{index:05d}.safetensors",
                report or {"dense_layer": torch.tensor(True)},
            )
            state["layers"].append(
                {
                    "layer": index,
                    "elapsed_seconds": time.monotonic() - start,
                    "weight_bytes": source.load_log[-1]["weight_bytes"],
                    "score_sha256": _file_digest(
                        output / "scores" / f"layer-{index:05d}.safetensors"
                    ),
                    "rss_after_forward_bytes": resident,
                    "rss_after_release_bytes": _rss_bytes(),
                    "process_peak_rss_bytes": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * 1024,
                }
            )
            state.update(
                next_layer=index + 1,
                stage=stage,
                activation_hashes=_activation_hashes(output, stage, len(batches)),
            )
            _save_json(manifest, state)
            _retire_stage(output, old_stage, len(batches))
            if progress:
                progress(state["layers"][-1])
        if stop == source.config.num_hidden_layers:
            old_stage = state["stage"]
            with source.final_norm() as norm:
                for i in range(len(batches)):
                    hidden = load_file(_activation_path(output, old_stage, i))["hidden"]
                    _save_tensors(
                        _activation_path(output, "final", i), {"hidden": norm(hidden)}
                    )
                    del hidden
            state.update(
                stage="final",
                complete=True,
                activation_hashes=_activation_hashes(output, "final", len(batches)),
            )
            _save_json(manifest, state)
            _retire_stage(output, old_stage, len(batches))
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="JSON with a calibration list of raw text snippets",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.max_tokens, args.samples, args.threads) <= 0:
        parser.error("max-tokens, samples, and threads must be positive")
    torch.set_num_threads(args.threads)
    source = MuseShardSource(args.source)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.source / "tokenizer.json"))
    texts = json.loads(args.corpus.read_text())["calibration"][: args.samples]
    bos = [] if source.config.bos_token_id is None else [source.config.bos_token_id]
    batches = [
        {
            "input_ids": torch.tensor(
                [
                    bos
                    + tokenizer.encode(text, add_special_tokens=False).ids[
                        : args.max_tokens - len(bos)
                    ]
                ],
                dtype=torch.long,
            )
        }
        for text in texts
    ]
    state = run_calibration(
        source,
        batches,
        args.output,
        stop_after_layer=args.stop_after_layer,
        resume=args.resume,
        progress=lambda row: print(json.dumps(row), flush=True),
    )
    print(
        json.dumps(
            {
                "next_layer": state["next_layer"],
                "complete": state["complete"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
