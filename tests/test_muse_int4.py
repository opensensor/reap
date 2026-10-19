from types import SimpleNamespace
from collections import OrderedDict
import hashlib
from threading import RLock

import pytest
import torch

from reap.muse_fast import pack_tiles
from reap.muse_int4 import (
    DirectA8,
    DirectA8Bank,
    NativeA8Bank,
    canonical_codes,
    check_direct_manifest,
    pack_part,
)


@pytest.mark.parametrize("n,k", [(32, 256), (672, 2048), (704, 512), (1408, 1024)])
@pytest.mark.parametrize("alignment", [32, 64, 128])
def test_native_layout_preserves_weights_and_padding(n, k, alignment):
    torch.manual_seed(n + k)
    canonical = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8)
    scales = torch.randn(n, k // 128).half()
    tiled = pack_tiles(canonical)
    assert torch.equal(canonical_codes(tiled), canonical)
    q, s, offsets, sums = pack_part(
        tiled[None],
        pack_tiles(scales, scales=True)[None],
        torch.full((1, n, k // 128), 8, dtype=torch.int8),
        alignment=alignment,
    )
    padded = q.shape[1]
    decoded = q[0].reshape(padded // 16, k // 128, 2, 16, 32)
    decoded = decoded.permute(0, 3, 1, 2, 4).contiguous().reshape(padded, k // 2)
    assert torch.equal(decoded[:n], canonical)
    assert not decoded[n:].any()
    restored_scale = s[0].reshape(padded // 16, k // 128, 16).transpose(1, 2)
    restored_scale = restored_scale.reshape(padded, k // 128)
    assert torch.equal(restored_scale[:n], scales)
    assert not restored_scale[n:].any()
    assert not offsets.any()
    restored_sum = sums[0].reshape(padded // 16, k // 128, 16).transpose(1, 2)
    restored_sum = restored_sum.reshape(padded, k // 128)
    fields = torch.stack((canonical & 15, (canonical >> 4) & 15), -1)
    signed = torch.where(fields >= 8, fields - 16, fields)
    assert torch.equal(restored_sum[:n], signed.reshape(n, -1, 128).sum(-1).half())
    assert not restored_sum[n:].any()


def test_native_layout_rejects_asymmetric_and_nonfinite_weights():
    q = torch.zeros(1, 32, 128, dtype=torch.int8)
    s = torch.ones(1, 32, 2, dtype=torch.float16)
    o = torch.full_like(s, 8, dtype=torch.int8)
    o[0, 0, 0] = 7
    with pytest.raises(ValueError, match="symmetric"):
        pack_part(q, s, o)
    o.fill_(8)
    s[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pack_part(q, s, o)


def test_native_adapter_rejects_invalid_split_geometry():
    part = (
        torch.zeros(1, 32, 128, dtype=torch.int8),
        torch.ones(1, 32, 2, dtype=torch.float16),
        torch.full((1, 32, 2), 8, dtype=torch.int8),
    )
    with pytest.raises(ValueError, match="width mismatch"):
        NativeA8Bank(SimpleNamespace(inputs=512, outputs=32, parts=[part]), "cpu")
    with pytest.raises(ValueError, match="output geometry"):
        NativeA8Bank(SimpleNamespace(inputs=256, outputs=64, parts=[part]), "cpu")


def test_native_adapter_preserves_split_sum_and_crops_padding(monkeypatch):
    part = (
        torch.zeros(1, 32, 128, dtype=torch.int8),
        torch.ones(1, 32, 2, dtype=torch.float16),
        torch.full((1, 32, 2), 8, dtype=torch.int8),
    )
    bank = NativeA8Bank(
        SimpleNamespace(inputs=768, outputs=32, parts=[part, part, part]), "cpu"
    )
    x = torch.arange(768, dtype=torch.float16).expand(8, -1)
    routes = torch.zeros(8, dtype=torch.int32)
    packed_inputs = []
    values = iter((2048, 1, -2048))

    def pack(value):
        packed_inputs.append(value.clone())
        return (value,) * 4

    def matmul(*arguments):
        assert arguments[-1] is routes
        result = torch.full((8, 128), next(values), dtype=torch.float16)
        result[:, 32:] = float("nan")
        return result

    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_qwen_w4_a8_pack_310", pack, raising=False
    )
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_qwen_w4_a8_int4_matmul_310", matmul, raising=False
    )
    actual = bank.linear(x, ids=routes)
    assert torch.equal(actual, torch.ones(8, 32, dtype=torch.float16))
    assert all(
        torch.equal(value, x[:, i * 256 : (i + 1) * 256])
        for i, value in enumerate(packed_inputs)
    )
    with pytest.raises(ValueError, match="route geometry"):
        bank.linear(x, ids=routes.long())


def test_direct_manifest_checks_quantizer_before_loading(tmp_path, monkeypatch):
    files = {}
    for name in ("library", "kernel", "quantizer"):
        p = tmp_path / name
        p.write_bytes(name.encode())
        files[name] = dict(
            path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()
        )
    manifest = dict(
        name="muse_a8_test",
        namespace="muse_a8_test",
        abi="muse-w4a8-zero64-v1",
        **files,
    )
    assert check_direct_manifest(manifest)
    loaded = []
    monkeypatch.setattr(torch.ops, "load_library", loaded.append)
    (tmp_path / "quantizer").write_bytes(b"changed")
    with pytest.raises(ValueError, match="quantizer hash mismatch"):
        DirectA8(manifest, [])
    assert not loaded


def test_direct_bank_omits_offsets_and_accounts_padding():
    q = torch.zeros(1, 672, 128, dtype=torch.int8)
    s = torch.ones(1, 672, 2, dtype=torch.float16)
    o = torch.full_like(s, 8, dtype=torch.int8)
    bank = DirectA8Bank(
        SimpleNamespace(inputs=256, outputs=672, parts=[(q, s, o)]), "cpu", None
    )
    assert len(bank.parts[0]) == 3
    assert bank.parts[0][0].shape == (1, 704, 128)
    assert (
        sum(v.numel() * v.element_size() for v in bank.parts[0])
        == 704 * 256 * (64 + 4) // 128
    )


def test_direct_launch_descriptor_crop_and_geometry_lifetime():
    backend = DirectA8.__new__(DirectA8)
    device = torch.device("cpu")
    backend.kernels = {device: ("matmul", "pack")}
    backend.empty = {device: torch.empty(0, dtype=torch.uint8)}
    backend.geometry = OrderedDict()
    backend.geometry_lock = RLock()
    q = torch.zeros(1, 64, 128, dtype=torch.int8)
    scale = torch.ones(1, 64, 2, dtype=torch.float16)
    bank = SimpleNamespace(inputs=512, outputs=32, parts=[(q, scale, scale)] * 2)
    inputs = torch.ones(8, 512, dtype=torch.float16)
    routes = torch.zeros(8, dtype=torch.int32)
    calls = []

    def launch(kernel, arguments, cores):
        calls.append(kernel)
        assert cores == 8
        if kernel == "pack":
            assert arguments[-1].tolist() == [16]
            for t in arguments[1:5]:
                t.zero_()
        else:
            assert arguments[-1].tolist() == [8, 1, 64, 256, 8, 1, 1]
            assert arguments[7] is routes
            arguments[8].fill_(2)
            arguments[8][:, 32:] = float("nan")

    backend.launch = launch
    assert torch.equal(
        backend.linear(bank, inputs, ids=routes),
        torch.full((8, 32), 4, dtype=torch.float16),
    )
    assert calls == ["pack", "matmul", "pack", "matmul"]
    # Graph owners retain these tensors even after varied prefill evicts the LRU.
    retained = tuple(backend.geometry.values())
    for i in range(300):
        backend.metadata((device, i + 20, 0, 256, 0, -1), [i])
    assert len(backend.geometry) == 256
    assert retained[0].tolist() == [16]
    assert retained[1].tolist() == [8, 1, 64, 256, 8, 1, 1]
    calls.clear()
    bank.inputs = 768
    with pytest.raises(ValueError, match="split geometry"):
        backend.linear(bank, inputs, ids=routes)
    assert not calls


def test_direct_builder_refuses_changed_upstream_anchors():
    import runpy
    from pathlib import Path

    builder = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/muse_native/build_int4.py")
    )
    offset = "        DataCopy(metadata[N], zw_[index], metadataCopy);"
    assert "Duplicate(metadata[N]" in builder["zero_offsets"](offset)
    for text in ("", offset + offset):
        with pytest.raises(ValueError, match="offset load"):
            builder["zero_offsets"](text)
    pack = "KERNEL_TYPE_AIC_ONLY\n  auto td = reinterpret_cast<__gm__ PackTiling*>(tiling);"
    assert "InitSocState" in builder["direct_quantizer"](pack)
    for text in ("", pack + pack):
        with pytest.raises(ValueError, match="pack entry"):
            builder["direct_quantizer"](text)


def test_a8_plan_matches_actual_sharded_allocations_and_preserves_kv(monkeypatch):
    from reap.muse_fast import TiledBank
    from reap.muse_runtime import placement_plan
    from reap.muse_tensor import shard_layer, tensor_plan

    with pytest.raises(ValueError, match="optimized tensor experts"):
        placement_plan("unused", ["cpu"], experimental_a8=True)
    monkeypatch.setattr(
        "reap.muse_tensor.native_weight", lambda value, device: value.to(device)
    )
    c = SimpleNamespace(
        num_hidden_layers=6,
        moe_hidden_size=256,
        moe_intermediate_size=256,
        sliding_window=2048,
        num_key_value_heads=2,
        head_dim=64,
        layer_types=["sliding_attention", "full_attention"] * 3,
    )

    def bank(n):
        q = torch.zeros(2, n, 128, dtype=torch.int8)
        s = torch.ones(2, n, 2, dtype=torch.float16)
        return TiledBank([(q, s, torch.full_like(s, 8, dtype=torch.int8))], 256, n)

    shards, expected = {}, [0] * 6
    backend = object()
    for index in range(6):
        cpu = SimpleNamespace(
            c=c,
            index=index,
            w={"input_layernorm.weight": torch.zeros(256).half()},
            gate_up=bank(512),
            down=bank(256),
        )
        layer = shard_layer(cpu, ["cpu"] * 6, "cpu", None, expert_backend=backend)
        for rank, r in enumerate(layer.tensor_experts.ranks):
            assert r.gate_up.backend is backend and r.down.backend is backend
            expected[rank] += sum(
                v.numel() * v.element_size()
                for b in (r.gate_up, r.down)
                for part in b.parts
                for v in part
            )
        shards[f"dense-{index:03d}.safetensors"] = {"tensors": {"norm": {"bytes": 512}}}
        for ep in (0, 1):
            tensors = {}
            for projection in ("gate", "up", "down"):
                name = f"mlp.experts.{ep}.{projection}_proj"
                tensors[name + ".weight"] = {"bytes": 256 * 128}
                tensors[name + ".weight_scale"] = {"bytes": 256 * 2 * 2}
            shards[f"experts-{index:03d}-ep{ep}.safetensors"] = {"tensors": tensors}
    stages = [dict(device=f"npu:{r}") for r in range(6)]
    base = tensor_plan(c, {"shards": shards}, stages, 98304, 512, 3, prefix_cache=True)
    candidate = tensor_plan(
        c,
        {"shards": shards},
        stages,
        98304,
        512,
        3,
        prefix_cache=True,
        experimental_a8=True,
    )
    assert [p["expert_shard_bytes"] for p in candidate] == expected
    assert sum(p["dense_and_kv_bytes"] for p in base) == sum(
        p["dense_and_kv_bytes"] for p in candidate
    )
    assert all(
        p["required_bytes"] - p["weights_and_kv_bytes"] == 3 * 2**30 for p in candidate
    )
