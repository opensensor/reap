"""Authenticated research route isolation; ordinary completion behavior stays intact."""

from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi.testclient import TestClient
import pytest

from reap.muse_resident import Resident
from reap.muse_serve import create_app
from tests.test_muse_resident import runtime
from tests.test_muse_serve import Tokenizer


TOKEN = "lookup-research-local-secret-1234567890"
HEADERS = {"Authorization": "Bearer " + TOKEN}
BODY = {"prompt": "code", "max_new_tokens": 8, "max_proposal_tokens": 3}


def clients(resident, tokenizer=None):
    app = create_app(
        resident.runtime,
        tokenizer or Tokenizer(),
        eos=[2],
        resident=resident,
        resident_token=TOKEN,
    )
    return TestClient(app, client=("127.0.0.1", 12345)), TestClient(
        app, client=("192.0.2.1", 12345)
    )


def test_lookup_auth_and_shared_request_lock(monkeypatch):
    resident = Resident(runtime())
    local, remote = clients(resident)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def lookup(**options):
        calls.append(options)
        entered.set()
        assert release.wait(5)
        return {"benchmark": {"tokens": [3, 2], "stats": {"verification_seconds": 0.1}}}

    monkeypatch.setattr(resident, "lookup", lookup)
    assert local.post("/resident/lookup", json=BODY).status_code == 403
    assert (
        local.post(
            "/resident/lookup", json=BODY, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 403
    )
    assert (
        remote.post(
            "/resident/lookup",
            json=BODY,
            headers={**HEADERS, "X-Forwarded-For": "127.0.0.1"},
        ).status_code
        == 403
    )
    assert calls == []
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(
            local.post, "/resident/lookup", json=BODY, headers=HEADERS
        )
        try:
            assert entered.wait(5)
            assert (
                local.post(
                    "/v1/completions", json={"prompt": "a", "max_tokens": 2}
                ).status_code
                == 503
            )
            for action, body in (
                ("lookup", BODY),
                ("oracle", {"prompt": "code", "max_tokens": 8}),
                ("switch", {}),
            ):
                assert (
                    local.post(
                        f"/resident/{action}", json=body, headers=HEADERS
                    ).status_code
                    == 503
                )
        finally:
            release.set()
        result = pending.result()
    assert result.status_code == 200
    assert result.json()["benchmark"] == {
        "tokens": [3, 2],
        "stats": {"verification_seconds": 0.1},
    }
    assert calls == [
        {
            "prompt_ids": [1, 4, 5],
            "eos": [2],
            "max_new_tokens": 8,
            "max_proposal_tokens": 3,
            "min_ngram": 2,
            "max_ngram": 4,
            "max_index_entries": 65536,
        }
    ]
    assert (
        local.post("/v1/completions", json={"prompt": "a", "max_tokens": 2}).status_code
        == 200
    )


def test_live_completion_blocks_lookup_before_runtime_reset(monkeypatch):
    resident = Resident(runtime())
    local, _ = clients(resident)
    entered, release = threading.Event(), threading.Event()
    resets, lookups = [], []
    monkeypatch.setattr(resident.runtime, "reset", lambda: resets.append(True))
    monkeypatch.setattr(resident, "lookup", lambda **options: lookups.append(options))

    def complete(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return [3, 2]

    monkeypatch.setattr(resident.runtime, "generate", complete)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(
            local.post, "/v1/completions", json={"prompt": "a", "max_tokens": 2}
        )
        try:
            assert entered.wait(5)
            before = len(resets)
            assert (
                local.post("/resident/lookup", json=BODY, headers=HEADERS).status_code
                == 503
            )
            assert len(resets) == before and lookups == []
        finally:
            release.set()
        assert pending.result().status_code == 200


@pytest.mark.parametrize(
    "updates",
    [
        {"prompt": ""},
        {"prompt": None},
        {"prompt": 123},
        {"prompt": "x" * 1_048_577},
        {"source": "not accepted"},
        {"temperature": 0},
        {"eos": [3]},
        {"max_new_tokens": 0},
        {"max_new_tokens": 129},
        {"max_new_tokens": True},
        {"max_new_tokens": "8"},
        {"max_new_tokens": 8.0},
        {"max_proposal_tokens": -1},
        {"max_proposal_tokens": 8},
        {"max_proposal_tokens": False},
        {"min_ngram": 0},
        {"max_ngram": 17},
        {"min_ngram": 4, "max_ngram": 3},
        {"max_index_entries": 0},
        {"max_index_entries": 65537},
    ],
)
def test_invalid_lookup_options_rejected_before_tokenization(monkeypatch, updates):
    resident = Resident(runtime())
    tokenizer = Tokenizer()
    encoded, calls = [], []
    monkeypatch.setattr(
        tokenizer, "encode", lambda prompt: encoded.append(prompt) or [1, 4, 5]
    )
    monkeypatch.setattr(resident, "lookup", lambda **options: calls.append(options))
    local, _ = clients(resident, tokenizer)
    assert (
        local.post(
            "/resident/lookup", json={**BODY, **updates}, headers=HEADERS
        ).status_code
        == 400
    )
    assert calls == [] and encoded == []


def test_lookup_context_bound_and_missing_prompt(monkeypatch):
    resident = Resident(runtime())
    calls = []
    monkeypatch.setattr(resident, "lookup", lambda **options: calls.append(options))
    local, _ = clients(resident)
    assert local.post("/resident/lookup", json={}, headers=HEADERS).status_code == 400
    assert (
        local.post(
            "/resident/lookup", json={**BODY, "max_new_tokens": 18}, headers=HEADERS
        ).status_code
        == 400
    )
    assert calls == []


def test_lookup_failure_releases_lock_for_ordinary_requests(monkeypatch):
    resident = Resident(runtime())
    local, _ = clients(resident)
    resets = []
    monkeypatch.setattr(resident.runtime, "reset", lambda: resets.append(True))

    def fail(**options):
        raise ValueError("injected lookup failure")

    monkeypatch.setattr(resident, "lookup", fail)
    result = local.post("/resident/lookup", json=BODY, headers=HEADERS)
    assert (
        result.status_code == 400
        and "injected lookup failure" in result.json()["detail"]
    )
    assert resets == [True]
    assert (
        local.post("/v1/completions", json={"prompt": "a", "max_tokens": 2}).status_code
        == 200
    )


@pytest.mark.parametrize("fails", [False, True])
def test_resident_lookup_resets_before_and_after_and_retains_identity(
    monkeypatch, fails
):
    import reap.muse_lookup as module

    resident = Resident(runtime())
    resets, synchronized = [], []
    monkeypatch.setattr(resident.runtime, "reset", lambda: resets.append(True))
    monkeypatch.setattr(resident, "synchronize", lambda: synchronized.append(True))

    def generate(instance, prompt_ids, **options):
        assert instance is resident.runtime and resets == [True]
        assert prompt_ids == [1] and options["max_new_tokens"] == 8
        if fails:
            raise ValueError("injected generation failure")
        return {"tokens": [3, 2], "stats": {"draft_seconds": 0.2}}

    monkeypatch.setattr(module, "generate", generate)
    if fails:
        with pytest.raises(ValueError, match="injected generation failure"):
            resident.lookup(prompt_ids=[1], max_new_tokens=8)
        assert synchronized == []
    else:
        result = resident.lookup(prompt_ids=[1], max_new_tokens=8)
        assert result["benchmark"]["tokens"] == [3, 2]
        assert result["weight_storage_sha256"] == resident.identity
        assert synchronized == [True]
    assert resets == [True, True]
    resident.failed = "previous device failure"
    with pytest.raises(RuntimeError, match="restart required"):
        resident.lookup(prompt_ids=[1], max_new_tokens=8)
    assert resets == [True, True]


@pytest.mark.parametrize("fails", [False, True])
def test_storage_change_blocks_resident_even_when_generation_failed(monkeypatch, fails):
    import reap.muse_lookup as module

    resident = Resident(runtime())
    resets = []
    monkeypatch.setattr(resident.runtime, "reset", lambda: resets.append(True))
    monkeypatch.setattr(resident, "synchronize", lambda: None)

    def generate(*args, **kwargs):
        monkeypatch.setattr(resident, "storage_identity", lambda: "changed")
        if fails:
            raise ValueError("original generation error")
        return {"tokens": [3]}

    monkeypatch.setattr(module, "generate", generate)
    with pytest.raises(RuntimeError, match="changed resident weight storage"):
        resident.lookup(prompt_ids=[1], max_new_tokens=8)
    assert resets == [True, True]
    assert resident.failed == "lookup benchmark changed resident weight storage"
    local, _ = clients(resident)
    assert local.get("/health").json()["status"] == "blocked"
    assert (
        local.post("/v1/completions", json={"prompt": "a", "max_tokens": 2}).status_code
        == 503
    )


@pytest.mark.parametrize(
    "options",
    [
        {"max_new_tokens": 129},
        {"max_new_tokens": True},
        {"max_proposal_tokens": 8},
        {"max_index_entries": 65537},
        {"min_ngram": 5, "max_ngram": 4},
        {"prompt_ids": [True]},
    ],
)
def test_direct_resident_lookup_rejects_unbounded_options_before_reset(
    monkeypatch, options
):
    resident = Resident(runtime())
    resets = []
    monkeypatch.setattr(resident.runtime, "reset", lambda: resets.append(True))
    with pytest.raises(ValueError, match="unbounded resident lookup"):
        resident.lookup(**{"prompt_ids": [1], "max_new_tokens": 8, **options})
    assert resets == []
