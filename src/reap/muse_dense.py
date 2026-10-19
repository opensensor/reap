"""Lossless output-channel sharding of query and attention-gate projections."""

import torch
import torch.nn.functional as F

QG_NAMES = ("self_attn.q_proj.weight", "self_attn.gate_proj.weight")


class TensorAttention:
    """Share the expert rank pool, graph lifecycle, and bounded host staging.

    A request owns this pool until all rank futures finish. Only decode captures
    graphs; prefill uses the same weights with dynamic row counts. Original
    owner weights are never copied to the device when this mode is loaded.
    """

    def __init__(self, experts):
        self.experts = experts
        if not hasattr(experts.pool, "attention_staging"):
            experts.pool.attention_staging = {}

    @classmethod
    def from_layer(cls, cpu, experts):
        from reap.muse_fast import native_weight
        from reap.muse_tensor import partitions

        query, gate = (cpu.w[name] for name in QG_NAMES)
        if query.shape != gate.shape or query.ndim != 2:
            raise ValueError("query and gate must have equal matrix geometry")
        for rank, (lo, hi) in zip(
            experts.ranks, partitions(query.shape[0], len(experts.ranks), alignment=32)
        ):
            rank.attention_qg_weight = native_weight(
                torch.cat((query[lo:hi], gate[lo:hi])), rank.device
            )
        return cls(experts)

    def snapshot(self, key, value):
        buffers = self.experts.pool.attention_staging
        buffer = buffers.get(key)
        if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
            buffer = torch.empty(
                value.shape,
                dtype=value.dtype,
                device="cpu",
                pin_memory=value.device.type == "npu",
            )
            buffers[key] = buffer
        buffer.copy_(value)
        return buffer

    def project(self, x):
        from reap.muse_tensor import DecodeGraph

        experts = self.experts
        host = self.snapshot("inputs", x)
        use_graph = experts.decode_graphs and x.numel() == x.shape[-1]
        if use_graph:
            for i, rank in enumerate(experts.ranks):
                key = i, "dense-qg"
                if key not in experts.graphs:
                    with torch.npu.device(rank.device):
                        weight = rank.attention_qg_weight
                        experts.graphs[key] = DecodeGraph.capture(
                            lambda a, b, w=weight: F.linear(a, w),
                            host.to(rank.device),
                            torch.empty(0, dtype=torch.int32, device=rank.device),
                        )

        def run(i):
            rank = experts.ranks[i]
            try:
                with torch.inference_mode():
                    if use_graph:
                        graph = experts.graphs[i, "dense-qg"]
                        output = graph.run(host, graph.routes)
                    else:
                        output = F.linear(
                            host.to(rank.device), rank.attention_qg_weight
                        )
                    # Keep the two uneven shard widths, but only one row count
                    # per width: arbitrary prefill sizes cannot grow this cache.
                    result = self.snapshot((i, output.shape[-1]), output)
                    return result.chunk(2, dim=-1)
            except BaseException:
                if str(rank.device).startswith("npu:"):
                    torch.npu.synchronize(rank.device)
                raise

        futures = [experts.pool.submit(run, i) for i in range(len(experts.ranks))]
        results, failures = [], []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
        return torch.cat([v[0] for v in results] + [v[1] for v in results], dim=-1).to(
            x.device
        )
