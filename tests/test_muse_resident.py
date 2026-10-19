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


def test_oracle_control_auth_validation_and_request_lock(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    r = Resident(runtime())
    token = "x" * 40
    app = create_app(r.runtime, Tokenizer(), eos=[2], resident=r, resident_token=token)
    local = TestClient(app, client=("127.0.0.1", 12345))
    remote = TestClient(app, client=("192.0.2.1", 12345))
    headers = {"Authorization": "Bearer " + token}
    body = {"prompt": "code", "max_tokens": 8, "block_sizes": [1, 4]}
    entered, release = threading.Event(), threading.Event()
    calls = []

    def oracle(**options):
        calls.append(options)
        entered.set()
        assert release.wait(5)
        return {"completed": True}

    monkeypatch.setattr(r, "oracle", oracle)
    assert local.post("/resident/oracle", json=body).status_code == 403
    assert (
        remote.post("/resident/oracle", json=body, headers=headers).status_code == 403
    )
    for invalid in ({}, {"prompt": ""}, {**body, "source": "bad"}):
        assert (
            local.post("/resident/oracle", json=invalid, headers=headers).status_code
            == 400
        )
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(
            local.post, "/resident/oracle", json=body, headers=headers
        )
        try:
            assert entered.wait(5)
            assert (
                local.post(
                    "/v1/completions", json={"prompt": "a", "max_tokens": 2}
                ).status_code
                == 503
            )
            assert (
                local.post("/resident/oracle", json=body, headers=headers).status_code
                == 503
            )
        finally:
            release.set()
        assert pending.result().status_code == 200
    assert calls == [{"prompt_ids": [1, 4, 5], "max_tokens": 8, "block_sizes": [1, 4]}]
    assert (
        local.post("/v1/completions", json={"prompt": "a", "max_tokens": 2}).status_code
        == 200
    )


def test_oracle_failure_cleans_runtime_and_checks_readiness(monkeypatch):
    import reap.muse_speculative as module

    r = Resident(runtime())
    resets = []
    monkeypatch.setattr(r.runtime, "reset", lambda: resets.append(True))

    def fail(*args, **kwargs):
        raise ValueError("invalid benchmark")

    monkeypatch.setattr(module, "benchmark", fail)
    with pytest.raises(ValueError, match="invalid benchmark"):
        r.oracle(prompt_ids=[1])
    assert len(resets) == 2
    r.failed = "device fault"
    with pytest.raises(RuntimeError, match="restart required"):
        r.oracle(prompt_ids=[1])
    assert len(resets) == 2


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


def test_a8_python_swaps_preserve_backend_and_reject_w4_manifest(monkeypatch):
    import reap.muse_int4 as module

    model = runtime()
    manifest = {"name": "a8", "namespace": "muse_a8_test", "library": {}}
    model.experimental_a8_manifest = manifest
    backend = object()
    for layer in model.layers:
        layer.gate_up.backend = layer.down.backend = backend

    def validate(value):
        if value != manifest:
            raise ValueError("invalid direct A8 manifest")
        return "a8-digest"

    monkeypatch.setattr(module, "check_direct_manifest", validate)
    resident = Resident(model)
    monkeypatch.setattr(resident, "synchronize", lambda: None)
    assert resident.status()["expert_format"] == "W4A8"
    assert resident.switch()["active_native"] == "a8"
    assert model.layers[0].gate_up.backend is backend
    with pytest.raises(ValueError, match="invalid direct A8"):
        resident.load_native({"name": "w4"})
    assert resident.failed is None
    wrong = """
def wrong(self, x, eps, weight=None):
    return x + 1
def replacements():
    return {"reap.muse_fast:FastMuseLayer.fast_rms": wrong}
"""
    with pytest.raises(AssertionError):
        resident.switch(source=wrong)
    assert resident.failed is None and resident.active == "a8"
    assert model.layers[0].gate_up.backend is backend
    assert resident.storage_identity() == resident.identity
    alternate = object()
    resident.loaded["a8-next"] = "next-digest", alternate
    assert resident.switch(native="a8-next")["active_native"] == "a8-next"
    assert resident.switch()["active_native"] == "a8-next"
    assert model.layers[0].down.backend is alternate


def test_a8_resident_rejects_mixed_bank_backends(monkeypatch):
    import reap.muse_int4 as module

    model = runtime()
    model.experimental_a8_manifest = {}
    monkeypatch.setattr(module, "check_direct_manifest", lambda _: "digest")
    model.layers[0].gate_up.backend = object()
    with pytest.raises(ValueError, match="share their loaded backend"):
        Resident(model)


def test_a8_native_qualification_uses_a8_reference(monkeypatch):
    import reap.muse_int4 as module

    model = runtime()
    model.experimental_a8_manifest = {
        "name": "initial",
        "namespace": "muse_a8_initial_test",
        "library": {},
    }

    class Backend:
        def linear(self, bank, x, **routes):
            return x[:, : bank.outputs] * 2

    backend = Backend()
    bank = model.layers[0].gate_up
    bank.backend = model.layers[0].down.backend = backend
    bank.linear = lambda x, **routes: bank.backend.linear(bank, x, **routes)
    down = model.layers[0].down
    down.linear = lambda x, **routes: down.backend.linear(down, x, **routes)
    monkeypatch.setattr(module, "check_direct_manifest", lambda _: "digest")
    candidate = Backend()
    monkeypatch.setattr(module, "DirectA8", lambda *args: candidate)
    resident = Resident(model)
    monkeypatch.setattr(resident, "synchronize", lambda: None)
    manifest = {"name": "next", "namespace": "muse_a8_next_test", "library": {}}
    assert resident.load_native(manifest)["probe_passed"]
    assert bank.backend is backend  # Loading qualifies but does not activate.
    assert resident.loaded["next"][1] is candidate
