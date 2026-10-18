from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import torch

from reap.muse_fast import FastMuseLayer, TiledBank
from reap.muse_resident import Resident
from reap.muse_serve import create_app
from tests.test_muse_serve import Tokenizer


def runtime():
    q = torch.zeros(8, 128, 128, dtype=torch.int8)
    s = torch.ones(8, 128, 2, dtype=torch.float16)

    def bank():
        return TiledBank(
            [(q.clone(), s.clone(), torch.full_like(s, 8, dtype=torch.int8))], 256, 128
        )

    layer = FastMuseLayer(
        SimpleNamespace(),
        0,
        {"input_layernorm.weight": torch.zeros(128).half()},
        bank(),
        bank(),
    )

    class Runtime:
        optimized = True
        globals = {}
        max_context = 20
        layers = [layer]

        def reset(self):
            pass

        def forward(self, ids, all_logits):
            return layer.fast_rms(torch.arange(128).half(), 1e-5).float()

        def generate(self, ids, limit, eos):
            return [3, 2]

    return Runtime()


def test_switch_and_rollback_keep_resident_storage(monkeypatch):
    r = Resident(runtime())
    monkeypatch.setattr(r, "synchronize", lambda: None)
    identity = r.identity
    assert r.switch()["generation"] == 1
    # Valid Python with wrong numerical output must be rolled back automatically.
    source = """
def wrong(self, x, eps, weight=None):
    return x + 1

def replacements():
    return {"reap.muse_fast:FastMuseLayer.fast_rms": wrong}
"""
    with pytest.raises(AssertionError):
        r.switch(source=source)
    assert r.failed is None
    assert r.generation == 1
    assert r.storage_identity() == identity
    assert r.switch()["probe_passed"]
    with pytest.raises(ValueError, match="qualified first"):
        r.switch(native="missing")


def test_local_control_auth_and_failure_blocks_inference():
    r = Resident(runtime())
    token = "x" * 40
    app = create_app(r.runtime, Tokenizer(), eos=[2], resident=r, resident_token=token)
    local = TestClient(app, client=("127.0.0.1", 12345))
    remote = TestClient(app, client=("192.0.2.1", 12345))
    headers = {"Authorization": "Bearer " + token}
    assert local.get("/resident/status").status_code == 403
    assert local.get("/resident/status", headers=headers).status_code == 200
    assert (
        remote.get(
            "/resident/status", headers={**headers, "X-Forwarded-For": "127.0.0.1"}
        ).status_code
        == 403
    )
    r.failed = "native registration failed"
    assert local.get("/health").json()["status"] == "blocked"
    assert (
        local.post("/v1/completions", json={"prompt": "a", "max_tokens": 2}).status_code
        == 503
    )
    assert local.post("/resident/switch", json={}, headers=headers).status_code == 400


def test_native_preflight_and_partial_failure(monkeypatch):
    import reap.muse_resident as module

    r = Resident(runtime())
    manifest = {"name": "candidate", "namespace": "muse_w4_failure_test", "library": {}}

    def invalid(_):
        raise ValueError("bad hash")

    monkeypatch.setattr(module, "check_manifest", invalid)
    with pytest.raises(ValueError, match="bad hash"):
        r.load_native(manifest)
    assert r.failed is None and not r.loaded
    monkeypatch.setattr(module, "check_manifest", lambda _: "digest")

    def fails(*args):
        raise RuntimeError("partial native registration")

    monkeypatch.setattr(module, "NativeW4", fails)
    with pytest.raises(RuntimeError, match="partial native"):
        r.load_native(manifest)
    assert r.failed and not r.loaded
    with pytest.raises(RuntimeError, match="restart required"):
        r.switch()


def test_whole_layer_candidate_is_qualified_and_rolled_back(monkeypatch):
    model = runtime()
    monkeypatch.setattr(
        FastMuseLayer, "__call__", lambda self, x: self.fast_rms(x, 1e-5)
    )
    model.forward = lambda ids, all_logits: model.layers[0](torch.arange(128).half())
    resident = Resident(model)
    monkeypatch.setattr(resident, "synchronize", lambda: None)
    expected = model.forward([1], True)
    source = """
def wrong(self, x):
    return x + 1

def replacements():
    return {"reap.muse_fast:FastMuseLayer.__call__": wrong}
"""
    with pytest.raises(AssertionError):
        resident.switch(source=source)
    assert resident.failed is None and resident.generation == 0
    assert resident.storage_identity() == resident.identity
    torch.testing.assert_close(model.forward([1], True), expected, rtol=0, atol=0)
