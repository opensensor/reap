from types import SimpleNamespace

import pytest
import torch

from reap import muse_fast
from reap.muse_fast import TiledBank, pack_tiles
from reap.muse_int4 import DirectA8Bank
from reap.muse_int4_cache import load_native_layer
from reap.muse_runtime import MuseRuntime
from reap.muse_tensor import shard_layer


def fixture(tmp_path, monkeypatch):
    model, cache = tmp_path / "model", tmp_path / "native-cache"
    model.mkdir()
    (model / "config.json").write_text("{}")
    journal = {
        "shards": {
            name: {"sha256": "source-digest"}
            for name in (
                "dense-000.safetensors",
                "experts-000-ep0.safetensors",
                "experts-000-ep1.safetensors",
            )
        }
    }
    c = SimpleNamespace(
        moe_hidden_size=256, moe_intermediate_size=256, num_local_experts=2
    )

    def bank(n):
        codes = torch.randint(-128, 128, (2, n, 128), dtype=torch.int8)
        scales = torch.rand(2, n, 2).half()
        q = torch.stack([pack_tiles(x) for x in codes])
        s = torch.stack([pack_tiles(x, scales=True) for x in scales])
        return TiledBank([(q, s, torch.full_like(s, 8, dtype=torch.int8))], 256, n)

    cpu = SimpleNamespace(
        c=c,
        index=0,
        w={"input_layernorm.weight": torch.randn(256).half()},
        gate_up=bank(512),
        down=bank(256),
    )
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return cpu

    monkeypatch.setattr(muse_fast, "load_fast_layer", load)
    monkeypatch.setattr(
        "reap.muse_tensor.native_weight", lambda value, device: value.to(device)
    )
    return model, cache, journal, cpu, calls


def test_native_cache_hits_skip_conversion_preserve_every_shard_and_bind_topology(
    tmp_path, monkeypatch
):
    model, cache, journal, cpu, calls = fixture(tmp_path, monkeypatch)
    backend = object()
    direct = shard_layer(cpu, ["cpu"] * 6, "cpu", None, expert_backend=backend)
    first = load_native_layer(model, 0, cpu.c, 6, cache, journal=journal)
    second = load_native_layer(model, 0, cpu.c, 6, cache, journal=journal)
    assert len(calls) == 1 and calls[0]["verify"]
    cached = shard_layer(second, ["cpu"] * 6, "cpu", None, expert_backend=backend)
    for a, b in zip(direct.tensor_experts.ranks, cached.tensor_experts.ranks):
        assert torch.equal(a.w["input_layernorm.weight"], b.w["input_layernorm.weight"])
        for phase in ("gate_up", "down"):
            old, new = getattr(a, phase), getattr(b, phase)
            assert new.backend is backend and (old.inputs, old.outputs) == (
                new.inputs,
                new.outputs,
            )
            for p, q in zip(old.parts, new.parts):
                for x, y in zip(p, q):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                    assert x.data_ptr() != y.data_ptr()
    with pytest.raises(ValueError, match="matching backend"):
        shard_layer(first, ["cpu"] * 6, "cpu", None)
    load_native_layer(model, 0, cpu.c, 2, cache, journal=journal)
    assert len(calls) == 2
    journal["shards"]["dense-000.safetensors"]["sha256"] = "different source"
    load_native_layer(model, 0, cpu.c, 6, cache, journal=journal)
    assert len(calls) == 3 and len(list(cache.glob("layer-*/receipt.json"))) == 3
    cpu.down.outputs -= 32
    journal["shards"]["dense-000.safetensors"]["sha256"] = "invalid-geometry"
    with pytest.raises(ValueError, match="layer geometry"):
        load_native_layer(model, 0, cpu.c, 6, cache, journal=journal)
    assert len(list(cache.glob("layer-*/receipt.json"))) == 3


def test_native_cache_byte_corruption_fails_closed_without_repacking(
    tmp_path, monkeypatch
):
    model, cache, journal, cpu, calls = fixture(tmp_path, monkeypatch)
    load_native_layer(model, 0, cpu.c, 6, cache, journal=journal)
    path = next(cache.glob("layer-*/weights.safetensors"))
    with path.open("r+b") as f:
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 1]))
    with pytest.raises(ValueError, match="integrity failure"):
        load_native_layer(model, 0, cpu.c, 6, cache, journal=journal)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="requires verification"):
        load_native_layer(model, 0, cpu.c, 6, cache, journal=journal, verify=False)


def test_native_cache_interrupted_write_never_publishes_partial_directory(
    tmp_path, monkeypatch
):
    model, cache, journal, cpu, calls = fixture(tmp_path, monkeypatch)
    import reap.muse_int4_cache as module

    original = module.save_file

    def fail(tensors, path):
        original(tensors, path)
        raise OSError("interrupted")

    monkeypatch.setattr(module, "save_file", fail)
    with pytest.raises(OSError, match="interrupted"):
        load_native_layer(model, 0, cpu.c, 6, cache, journal=journal)
    assert not [p for p in cache.glob("layer-*") if p.is_dir()]
    assert not list(cache.glob(".packing-*"))
    monkeypatch.setattr(module, "save_file", original)
    load_native_layer(model, 0, cpu.c, 6, cache, journal=journal)
    assert len(list(cache.glob("layer-*/receipt.json"))) == 1


def test_cached_native_bank_rejects_abi_and_owns_independent_staging():
    q = torch.zeros(2, 64, 128, dtype=torch.int8)
    s = torch.ones(2, 64, 2, dtype=torch.float16)
    bank = DirectA8Bank.from_native([(q, s, s)], 256, 32, "cpu", None)
    assert all(
        v.data_ptr() != original.data_ptr()
        for v, original in zip(bank.parts[0], (q, s, s))
    )
    with pytest.raises(ValueError, match="weight ABI"):
        DirectA8Bank.from_native([(q, s.float(), s)], 256, 32, "cpu", None)
    with pytest.raises(ValueError, match="split geometry"):
        DirectA8Bank.from_native([(q, s, s)], 512, 32, "cpu", None)
    with pytest.raises(ValueError, match="requires the experimental"):
        MuseRuntime("unused", experimental_a8_cache="unused")
