"""Output-sharded experts with host gathers and per-device decode graphs.

Dense layers and KV stay on the sequential stages. Every expert projection uses
all devices. Output partitioning preserves the original split-K reductions and
FP16 rounding; it does not approximate routing, weights, or normalization.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from reap.muse_fast import FastMuseLayer, TiledBank, native_weight


_GRAPH_CONTEXTS = {}


def graph_context(device):
    # A separate expandable allocator pool per graph reserves a device-sized
    # virtual segment. Hundreds of graphs exhaust the driver's address space.
    # Captures/replays are ordered and outputs stay owned by each DecodeGraph;
    # sharing one pool and capture stream per device safely reuses scratch.
    device = torch.device(device)
    if device not in _GRAPH_CONTEXTS:
        with torch.npu.device(device):
            _GRAPH_CONTEXTS[device] = (
                torch.npu.graph_pool_handle(),
                torch.npu.Stream(device=device),
            )
    return _GRAPH_CONTEXTS[device]


def partitions(size, ranks, rotation=0, alignment=128):
    if ranks < 1 or alignment < 1 or size % alignment or size // alignment < ranks:
        raise ValueError("invalid aligned partition geometry")
    blocks, extra = divmod(size // alignment, ranks)
    widths = [
        alignment * (blocks + ((r - rotation) % ranks < extra)) for r in range(ranks)
    ]
    start, result = 0, []
    for width in widths:
        result.append((start, start + width))
        start += width
    return result


def slice_bank(bank, ranges, device):
    """NZ rows are contiguous physical blocks only at 32-channel boundaries."""
    if not ranges or any(
        lo < 0 or hi > bank.outputs or lo >= hi or lo % 32 or hi % 32
        for lo, hi in ranges
    ):
        raise ValueError("bank slices must contain complete NZ blocks")
    parts = []
    for codes, scales, _ in bank.parts:
        # cat materializes aligned host staging, including single-range slices.
        q, s = [
            torch.cat([v[:, lo:hi] for lo, hi in ranges], dim=1).contiguous().to(device)
            for v in (codes, scales)
        ]
        parts.append((q, s, torch.full_like(s, 8, dtype=torch.int8)))
    return TiledBank(parts, bank.inputs, sum(hi - lo for lo, hi in ranges))


@dataclass
class DecodeGraph:
    inputs: torch.Tensor
    routes: torch.Tensor
    graph: object
    output: torch.Tensor
    stream: object
    retained: tuple = ()

    @classmethod
    def capture(cls, function, inputs, routes):
        import torch_npu  # noqa: F401

        with torch.inference_mode(), torch.npu.device(inputs.device):
            inputs, routes = inputs.clone(), routes.clone()
            function(inputs, routes)
            torch.npu.synchronize(inputs.device)
            graph = torch.npu.NPUGraph()
            # torch.npu.graph's default capture stream is process-global, not
            # device-local. Explicit streams are required for all six captures.
            pool, stream = graph_context(inputs.device)
            with torch.npu.graph(graph, pool=pool, stream=stream):
                output = function(inputs, routes)
            graph.replay()
            torch.npu.synchronize(inputs.device)
        return cls(inputs, routes, graph, output, stream)

    def run(self, inputs, routes):
        self.inputs.copy_(inputs)
        self.routes.copy_(routes)
        self.graph.replay()
        return self.output


class TensorExperts:
    def __init__(self, ranks, pool, *, decode_graphs=True):
        self.ranks, self.pool = ranks, pool
        self.decode_graphs = decode_graphs
        self.graphs = {}

    def invalidate_graphs(self):
        for item in self.graphs.values():
            with torch.npu.device(item.inputs.device):
                torch.npu.synchronize(item.inputs.device)
                item.graph.reset()
                # The allocator retires this pool when its last graph resets.
                # Reusing the old pool token then hits use_count==0 or stale
                # virtual-memory handles. A new capture generation needs a
                # fresh pool/stream; all layers are invalidated before replay.
                _GRAPH_CONTEXTS.pop(item.inputs.device, None)
        self.graphs.clear()

    @staticmethod
    def operation(rank, phase, inputs, routes, grouped):
        bank = rank.gate_up if phase == "up" else rank.down
        if grouped and bank.backend is None:
            pieces, first = [], 0
            for expert, last in enumerate(routes.cpu().tolist()):
                if last > first:
                    pieces.append(bank.linear(inputs[first:last], expert=expert))
                first = last
            result = torch.cat(pieces)
            if phase == "up":
                gate, up = result.chunk(2, dim=-1)
                return F.silu(gate) * up
            return result
        metadata = {"ends" if grouped else "ids": routes}
        if phase == "up":
            gate, up = rank.gate_up.linear(inputs, **metadata).chunk(2, dim=-1)
            return F.silu(gate) * up
        return rank.down.linear(inputs, **metadata)

    def prepare_graphs(self, inputs, routes):
        # Captures must be serial. All later invocations reuse static buffers;
        # changing token activations AND expert IDs are copied before replay.
        if all(
            (i, phase) in self.graphs
            for i in range(len(self.ranks))
            for phase in ("up", "down")
        ):
            return
        for i, rank in enumerate(self.ranks):
            with torch.npu.device(rank.device):
                for phase in ("up", "down"):
                    key = i, phase
                    if key not in self.graphs:
                        self.graphs[key] = DecodeGraph.capture(
                            lambda x, ids, r=rank, p=phase: self.operation(
                                r, p, x, ids, False
                            ),
                            inputs.to(rank.device),
                            routes.to(rank.device),
                        )
                        backend = (
                            rank.gate_up.backend if phase == "up" else rank.down.backend
                        )
                        if backend is not None:
                            # Native geometry tensors are allocated during warmup,
                            # outside the graph pool. Pin them through replay even
                            # when varied prefill sizes evict the backend's LRU.
                            self.graphs[key].retained = tuple(backend.geometry.values())

    def project(self, inputs, *, ids=None, ends=None):
        if (ids is None) == (ends is None):
            raise ValueError("provide routed IDs or grouped ends")
        device = inputs.device
        host_inputs = inputs.cpu()
        host_routes = (ids if ids is not None else ends).cpu()
        use_graph = self.decode_graphs and ids is not None and len(inputs) == 8
        if use_graph:
            self.prepare_graphs(host_inputs, host_routes)

        def run(i, phase, value):
            rank = self.ranks[i]
            with torch.inference_mode(), torch.npu.device(rank.device):
                # Give each device worker independent, aligned host storage.
                # Avoid sharing pageable DMA registration across contexts;
                # each permanent worker retains its own host allocation arena.
                value, routes = value.clone(), host_routes.clone()
                if use_graph:
                    result = self.graphs[i, phase].run(value, routes)
                else:
                    result = self.operation(
                        rank,
                        phase,
                        value.to(rank.device),
                        routes.to(rank.device),
                        ends is not None,
                    )
                return result.cpu()

        for phase in ("up", "down"):
            futures = [
                self.pool.submit(run, i, phase, host_inputs)
                for i in range(len(self.ranks))
            ]
            # Wait for every rank even on error before allowing reset/hot swap.
            results, failures = [], []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append(exc)
            if failures:
                raise failures[0]
            host_inputs = torch.cat(results, dim=-1)
        return host_inputs.to(device)


def shard_layer(
    cpu, devices, owner, pool, *, decode_graphs=True, tensor_attention=False
):
    if (
        cpu.gate_up.outputs != 2 * cpu.down.inputs
        or cpu.gate_up.inputs != cpu.down.outputs
    ):
        raise ValueError("unsupported expert projection geometry")
    from reap.muse_dense import QG_NAMES, TensorAttention

    gate_ranges = partitions(cpu.down.inputs, len(devices), cpu.index)
    down_ranges = partitions(cpu.down.outputs, len(devices), cpu.index)
    ranks = []
    for device, (lo, hi), (down_lo, down_hi) in zip(devices, gate_ranges, down_ranges):
        middle = cpu.down.inputs
        gate_up = slice_bank(
            cpu.gate_up, [(lo, hi), (middle + lo, middle + hi)], device
        )
        down = slice_bank(cpu.down, [(down_lo, down_hi)], device)
        if str(device) == str(owner):
            weights = {
                name: native_weight(value.clone(), device)
                for name, value in cpu.w.items()
                if not tensor_attention or name not in QG_NAMES
            }
            layer = FastMuseLayer(cpu.c, cpu.index, weights, gate_up, down)
            rank = layer
        else:
            rank = SimpleNamespace(
                device=torch.device(device), w={}, gate_up=gate_up, down=down
            )
        ranks.append(rank)
    if str(owner) not in [str(d) for d in devices]:
        raise ValueError("dense owner is not a tensor rank")
    layer.tensor_experts = TensorExperts(ranks, pool, decode_graphs=decode_graphs)
    if tensor_attention:
        layer.tensor_attention = TensorAttention.from_layer(cpu, layer.tensor_experts)
    return layer


class RankPool:
    """One permanent host thread/context per NPU, with parallel rank execution."""

    def __init__(self, devices):
        self.workers = []
        for rank, device in enumerate(devices):
            device = getattr(device, "device", device)
            device = f"npu:{device}" if isinstance(device, int) else str(device)

            def initialize(device=device):
                if device.startswith("npu:"):
                    torch.npu.set_device(device)

            self.workers.append(
                ThreadPoolExecutor(
                    max_workers=1,
                    initializer=initialize,
                    thread_name_prefix=f"muse-rank-{rank}",
                )
            )

    def submit(self, function, rank, *args):
        return self.workers[rank].submit(function, rank, *args)

    def shutdown(self, wait=True):
        for worker in self.workers:
            worker.shutdown(wait=wait)


def make_pool(devices):
    return RankPool(devices)


def tensor_plan(
    config,
    journal,
    stages,
    max_context,
    prefill_chunk,
    reserve_gib,
    *,
    prefix_cache=False,
    tensor_attention=False,
):
    """Account for every shard, dense NZ matrix, KV slot, and global tensor."""
    from reap.muse_runtime import PREFIX

    ranks = len(stages)
    dense = [0] * config.num_hidden_layers
    globals_ = [0] * ranks
    experts = [0] * ranks
    attention = [0] * ranks
    for index in range(config.num_hidden_layers):
        tensors = journal["shards"][f"dense-{index:03d}.safetensors"]["tensors"]
        dense[index] += sum(t["bytes"] for t in tensors.values())
        for name, record in tensors.items():
            if name.endswith(".weight_scale"):
                dense[index] += (
                    tensors[name.removesuffix("_scale")]["bytes"] - record["bytes"]
                )
        if tensor_attention:
            from reap.muse_dense import QG_NAMES

            for name in QG_NAMES:
                key = f"{PREFIX}layers.{index}.{name}"
                record = tensors[key]
                rows, columns = record["shape"]
                # Dense W8 codes/scales have already been counted as FP16.
                cost = rows * columns * 2
                dense[index] -= cost
                for r, (lo, hi) in enumerate(partitions(rows, ranks, alignment=32)):
                    attention[r] += (hi - lo) * columns * 2
        tokens = (
            min(max_context, config.sliding_window - 1 + prefill_chunk)
            if config.layer_types[index] == "sliding_attention"
            else max_context
        )
        dense[index] += tokens * 2 * config.num_key_value_heads * config.head_dim * 2
        if prefix_cache:
            from reap.muse_prefix import snapshot_reserve_bytes

            dense[index] += snapshot_reserve_bytes(config, index, max_context)
        for ep in (0, 1):
            for name, record in journal["shards"][
                f"experts-{index:03d}-ep{ep}.safetensors"
            ]["tensors"].items():
                size = (
                    config.moe_hidden_size
                    if ".down_proj." in name
                    else config.moe_intermediate_size
                )
                cost = record["bytes"]
                if name.endswith(".weight_scale"):
                    cost += cost // 2  # UINT4's signed-zero offsets are INT8.
                for r, (lo, hi) in enumerate(partitions(size, ranks, index)):
                    experts[r] += cost * (hi - lo) // size
    for filename, shard in journal["shards"].items():
        if filename.startswith("global-"):
            for name, record in shard["tensors"].items():
                globals_[0 if name == PREFIX + "embed_tokens.weight" else -1] += record[
                    "bytes"
                ]
    # Expert bytes are now spread across every device. Rebalance dense stages
    # separately, including the large embedding and output head at the ends.
    dp = {(0, 0): (0, [])}
    for rank in range(ranks):
        for end in range(rank + 1, config.num_hidden_layers + 1):
            candidates = []
            for begin in range(rank, end):
                if (rank, begin) not in dp:
                    continue
                cost = (
                    sum(dense[begin:end])
                    + globals_[rank]
                    + experts[rank]
                    + attention[rank]
                )
                peak, ranges = dp[rank, begin]
                candidates.append((max(peak, cost), ranges + [(begin, end, cost)]))
            if candidates:
                dp[rank + 1, end] = min(candidates, key=lambda item: item[0])
    ranges = dp[ranks, config.num_hidden_layers][1]
    return [
        dict(
            stage,
            first_layer=begin,
            last_layer_exclusive=end,
            expert_shard_bytes=experts[r],
            attention_shard_bytes=attention[r],
            dense_and_kv_bytes=cost - experts[r],
            weights_and_kv_bytes=cost,
            required_bytes=cost + int(reserve_gib * 2**30),
        )
        for r, (stage, (begin, end, cost)) in enumerate(zip(stages, ranges))
    ]
