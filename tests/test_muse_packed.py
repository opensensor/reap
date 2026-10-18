import json

import pytest
import torch

from reap import muse_fast
from reap.muse_packed import load_cached_layer
from tests.test_muse_resident import runtime


def test_cache_reuse_integrity_and_source_identity(tmp_path, monkeypatch):
    model, cache = tmp_path / "model", tmp_path / "cache"
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
    layer = runtime().layers[0]
    calls = []

    def load(*args, **kwargs):
        calls.append(1)
        return layer

    monkeypatch.setattr(muse_fast, "load_fast_layer", load)
    first = load_cached_layer(model, 0, layer.c, "cpu", cache, journal=journal)
    second = load_cached_layer(model, 0, layer.c, "cpu", cache, journal=journal)
    assert len(calls) == 1
    for a, b in zip(first.gate_up.parts[0], second.gate_up.parts[0]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert first.gate_up.parts[0][0].data_ptr() != second.gate_up.parts[0][0].data_ptr()
    receipt_path = next(cache.glob("layer-*/receipt.json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["sha256"] = "corrupt"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="integrity"):
        load_cached_layer(model, 0, layer.c, "cpu", cache, journal=journal)
    assert len(calls) == 1
    # Changed export receipts must never reuse an old packed cache.
    journal["shards"]["dense-000.safetensors"]["sha256"] = "new-source"
    load_cached_layer(model, 0, layer.c, "cpu", cache, journal=journal)
    assert len(calls) == 2
