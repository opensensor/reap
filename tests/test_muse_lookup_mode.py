"""Live decode-policy changes are authenticated, serialized and preserve KV."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from reap.muse_resident import Resident
from tests.test_muse_lookup_control import HEADERS, clients
from tests.test_muse_resident import runtime


def supported():
    value = runtime()
    value.bounded_attention = True
    value.experimental_lookup_tokens = 0
    value.execution = "greedy; experimental W4A8 experts"
    value.prefix_snapshot = object()
    value.cache = [object()]
    return Resident(value)


def test_switch_modes_preserves_prefix_weights_and_reports_current_policy(monkeypatch):
    resident = supported()
    local, _ = clients(resident)
    prefix, cache = resident.runtime.prefix_snapshot, resident.runtime.cache
    identity, generation = resident.identity, resident.generation
    monkeypatch.setattr(
        resident.runtime, "reset", lambda: pytest.fail("mode change reset KV")
    )
    for value in (7, 3, 3, 0):
        result = local.post(
            "/resident/lookup-mode",
            json={"max_proposal_tokens": value},
            headers=HEADERS,
        )
        assert result.status_code == 200
        assert result.json()["experimental_lookup_tokens"] == value
        assert local.get("/health").json()["experimental_lookup_tokens"] == value
        assert (
            local.get("/resident/status", headers=HEADERS).json()[
                "experimental_lookup_tokens"
            ]
            == value
        )
        assert (
            resident.runtime.prefix_snapshot is prefix
            and resident.runtime.cache is cache
        )
        assert (
            resident.storage_identity() == identity
            and resident.generation == generation
        )
        assert resident.runtime.execution.count("experimental prompt lookup") == bool(
            value
        )
        assert "experimental W4A8 experts" in resident.runtime.execution
    assert resident.runtime.execution == "greedy; experimental W4A8 experts"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"max_proposal_tokens": 3, "source": "x"},
        *(
            {"max_proposal_tokens": value}
            for value in (None, True, False, -1, 8, 3.0, "3")
        ),
    ],
)
def test_invalid_payload_does_not_mutate_or_reset(monkeypatch, body):
    resident = supported()
    local, _ = clients(resident)
    monkeypatch.setattr(
        resident.runtime, "reset", lambda: pytest.fail("invalid mode reset KV")
    )
    result = local.post("/resident/lookup-mode", json=body, headers=HEADERS)
    assert result.status_code == 400
    assert resident.runtime.experimental_lookup_tokens == 0


def test_auth_required_before_mode_change():
    resident = supported()
    local, remote = clients(resident)
    body = {"max_proposal_tokens": 7}
    assert local.post("/resident/lookup-mode", json=body).status_code == 403
    assert (
        local.post(
            "/resident/lookup-mode",
            json=body,
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 403
    )
    assert (
        remote.post(
            "/resident/lookup-mode",
            json=body,
            headers={**HEADERS, "X-Forwarded-For": "127.0.0.1"},
        ).status_code
        == 403
    )
    assert resident.runtime.experimental_lookup_tokens == 0


@pytest.mark.parametrize(
    "unsupported", ["failed", "bounded_attention", "optimized", "missing"]
)
def test_unready_or_unsupported_rejected(unsupported):
    resident = supported()
    if unsupported == "failed":
        resident.failed = "previous qualification failure"
    elif unsupported == "missing":
        del resident.runtime.experimental_lookup_tokens
    else:
        setattr(resident.runtime, unsupported, False)
    local, _ = clients(resident)
    assert (
        local.post(
            "/resident/lookup-mode", json={"max_proposal_tokens": 7}, headers=HEADERS
        ).status_code
        == 400
    )
    assert getattr(resident.runtime, "experimental_lookup_tokens", 0) == 0


def test_running_generation_prevents_policy_change(monkeypatch):
    resident = supported()
    local, _ = clients(resident)
    entered, release = Event(), Event()

    def generate(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return [3, 2]

    monkeypatch.setattr(resident.runtime, "generate", generate)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            local.post, "/v1/completions", json={"prompt": "a", "max_tokens": 2}
        )
        try:
            assert entered.wait(5)
            result = local.post(
                "/resident/lookup-mode",
                json={"max_proposal_tokens": 7},
                headers=HEADERS,
            )
            assert result.status_code == 503
            assert resident.runtime.experimental_lookup_tokens == 0
        finally:
            release.set()
        assert future.result().status_code == 200
    assert (
        local.post(
            "/resident/lookup-mode", json={"max_proposal_tokens": 7}, headers=HEADERS
        ).status_code
        == 200
    )
