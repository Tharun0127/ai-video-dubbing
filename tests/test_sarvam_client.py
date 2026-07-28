"""Tests for the Sarvam client: caching, retries, backoff, errors, and cost recording.

The API is mocked throughout using a fake requests.Session that returns the real
response captured in tests/fixtures/. No test may open a socket -- the network
guard fixture in test_pipeline.py covers the pipeline path, and here the fake
session simply never connects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from src.cache import CacheMissError, DiskCache
from src.config import Config
from src.metrics import MetricsCollector
from src.sarvam_client import SarvamAPIError, SarvamClient


class FakeResponse:
    """Minimal stand-in for requests.Response covering what the client reads."""

    def __init__(self, status_code: int, payload: Any = None, headers: dict[str, str] | None = None):
        """Build a fake response with a status, JSON payload, and optional headers."""
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.content = json.dumps(self._payload).encode("utf-8")

    def json(self) -> Any:
        """Return the payload, mimicking requests' JSON decoding."""
        return self._payload

    @property
    def text(self) -> str:
        """Return the payload as text."""
        return json.dumps(self._payload)


class FakeSession:
    """A requests.Session replacement that replays a scripted list of responses."""

    def __init__(self, responses: list[FakeResponse | Exception]):
        """Queue the responses (or exceptions) this session will produce, in order."""
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        """Record the call and return (or raise) the next scripted item."""
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._responses:
            raise AssertionError(f"unexpected extra request: {method} {url}")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        """No-op close so the client's context manager works."""


def build_client(
    config: Config,
    responses: list[FakeResponse | Exception],
    metrics: MetricsCollector | None = None,
    cache: DiskCache | None = None,
) -> tuple[SarvamClient, FakeSession, list[float]]:
    """Wire a client to a fake session and a sleep spy, so backoff is fast and observable."""
    slept: list[float] = []
    session = FakeSession(responses)
    client = SarvamClient(
        config,
        cache=cache if cache is not None else DiskCache(config.cache_dir),
        metrics=metrics or MetricsCollector(),
        session=session,  # type: ignore[arg-type]
        sleep=slept.append,
    )
    return client, session, slept


# --- happy path + caching --------------------------------------------------------------

def test_first_call_hits_network_and_second_is_served_from_cache(
    config: Config, tiny_wav: Path, asr_response: dict[str, Any]
) -> None:
    """The core M1 guarantee: identical call twice = one network call, one cache hit, zero extra cost."""
    metrics = MetricsCollector()
    cache = DiskCache(config.cache_dir)

    client, session, _ = build_client(config, [FakeResponse(200, asr_response)], metrics, cache)
    first = client.speech_to_text(tiny_wav, language_code="en-IN", audio_duration_s=0.5)

    # A second client shares the cache but its session would raise on any request.
    client2, session2, _ = build_client(config, [], metrics, cache)
    second = client2.speech_to_text(tiny_wav, language_code="en-IN", audio_duration_s=0.5)

    assert first == second == asr_response
    assert len(session.calls) == 1
    assert len(session2.calls) == 0
    assert metrics.cache_hits == 1
    assert metrics.cache_misses == 1
    assert metrics.api_calls[1].cost_inr == 0.0
    assert metrics.total_cost_inr == pytest.approx(metrics.api_calls[0].cost_inr)


def test_changing_a_parameter_forces_a_new_network_call(
    config: Config, tiny_wav: Path, asr_response: dict[str, Any]
) -> None:
    """A different mode is a different cache key and must not reuse the cached result."""
    cache = DiskCache(config.cache_dir)
    client, session, _ = build_client(
        config, [FakeResponse(200, asr_response), FakeResponse(200, asr_response)], cache=cache
    )
    client.speech_to_text(tiny_wav, mode="transcribe", audio_duration_s=0.5)
    client.speech_to_text(tiny_wav, mode="verbatim", audio_duration_s=0.5)
    assert len(session.calls) == 2


def test_request_carries_auth_header_and_documented_form_fields(
    config: Config, tiny_wav: Path, asr_response: dict[str, Any]
) -> None:
    """The client sends api-subscription-key plus exactly the documented multipart fields."""
    client, session, _ = build_client(config, [FakeResponse(200, asr_response)])
    client.speech_to_text(tiny_wav, mode="transcribe", language_code="en-IN", audio_duration_s=0.5)

    assert session.headers["api-subscription-key"] == config.api_key
    call = session.calls[0]
    assert call["url"] == "https://api.sarvam.ai/speech-to-text"
    assert call["data"] == {"model": "saaras:v3", "mode": "transcribe", "language_code": "en-IN"}
    assert "file" in call["files"]


def test_cost_is_recorded_per_second_of_audio(
    config: Config, tiny_wav: Path, asr_response: dict[str, Any]
) -> None:
    """A 10s clip is billed at ceil(10) seconds x Rs30/hour."""
    metrics = MetricsCollector()
    client, _, _ = build_client(config, [FakeResponse(200, asr_response)], metrics)
    client.speech_to_text(tiny_wav, audio_duration_s=10.0)

    call = metrics.api_calls[0]
    assert call.cost_inr == pytest.approx(30.0 * 10 / 3600)
    assert call.billable_units == 10.0
    assert call.billable_unit_name == "audio_seconds"
    assert call.request_id == asr_response["request_id"]


# --- dry run ---------------------------------------------------------------------------

def test_dry_run_raises_on_cache_miss_instead_of_calling_the_api(
    config: Config, tiny_wav: Path
) -> None:
    """--dry-run must fail loudly, never silently spend credits."""
    dry = Config(**{**config.__dict__, "dry_run": True})
    client, session, _ = build_client(dry, [])
    with pytest.raises(CacheMissError, match="not cached"):
        client.speech_to_text(tiny_wav, audio_duration_s=0.5)
    assert session.calls == []


def test_dry_run_succeeds_when_the_entry_is_cached(
    config: Config, tiny_wav: Path, asr_response: dict[str, Any]
) -> None:
    """A populated cache lets --dry-run complete with zero network calls."""
    cache = DiskCache(config.cache_dir)
    warm, _, _ = build_client(config, [FakeResponse(200, asr_response)], cache=cache)
    warm.speech_to_text(tiny_wav, audio_duration_s=0.5)

    dry = Config(**{**config.__dict__, "dry_run": True})
    client, session, _ = build_client(dry, [], cache=DiskCache(config.cache_dir))
    assert client.speech_to_text(tiny_wav, audio_duration_s=0.5) == asr_response
    assert session.calls == []


# --- retries and backoff ---------------------------------------------------------------

def test_retries_on_429_then_succeeds(
    config: Config, tiny_wav: Path, asr_response: dict[str, Any]
) -> None:
    """A rate-limited call backs off and retries rather than failing the run."""
    metrics = MetricsCollector()
    client, session, slept = build_client(
        config,
        [FakeResponse(429, {"error": {"message": "rate limited"}}), FakeResponse(200, asr_response)],
        metrics,
    )
    assert client.speech_to_text(tiny_wav, audio_duration_s=0.5) == asr_response
    assert len(session.calls) == 2
    assert len(slept) == 1
    assert metrics.api_calls[0].attempts == 2


def test_retry_after_header_is_honoured(
    config: Config, tiny_wav: Path, asr_response: dict[str, Any]
) -> None:
    """When the server says how long to wait, the client waits exactly that long."""
    client, _, slept = build_client(
        config,
        [FakeResponse(429, {}, headers={"Retry-After": "2.5"}), FakeResponse(200, asr_response)],
    )
    client.speech_to_text(tiny_wav, audio_duration_s=0.5)
    assert slept == [2.5]


def test_retries_on_503_service_overloaded(
    config: Config, tiny_wav: Path, asr_response: dict[str, Any]
) -> None:
    """Docs say to retry 503 the same way as 429."""
    client, session, _ = build_client(
        config, [FakeResponse(503, {}), FakeResponse(200, asr_response)]
    )
    client.speech_to_text(tiny_wav, audio_duration_s=0.5)
    assert len(session.calls) == 2


def test_network_errors_are_retried(
    config: Config, tiny_wav: Path, asr_response: dict[str, Any]
) -> None:
    """A dropped connection is retried, not surfaced as a pipeline failure."""
    client, session, _ = build_client(
        config, [requests.ConnectionError("reset"), FakeResponse(200, asr_response)]
    )
    client.speech_to_text(tiny_wav, audio_duration_s=0.5)
    assert len(session.calls) == 2


def test_gives_up_after_max_retries(config: Config, tiny_wav: Path) -> None:
    """max_retries=3 means 4 attempts total, then a loud failure."""
    client, session, _ = build_client(config, [FakeResponse(503, {}) for _ in range(4)])
    with pytest.raises(SarvamAPIError) as exc:
        client.speech_to_text(tiny_wav, audio_duration_s=0.5)
    assert len(session.calls) == config.max_retries + 1
    assert exc.value.status_code == 503


def test_backoff_delays_stay_within_the_configured_ceiling(config: Config, tiny_wav: Path) -> None:
    """Full-jitter backoff never exceeds backoff_max_s."""
    client, _, slept = build_client(config, [FakeResponse(503, {}) for _ in range(4)])
    with pytest.raises(SarvamAPIError):
        client.speech_to_text(tiny_wav, audio_duration_s=0.5)
    assert all(0.0 <= d <= config.backoff_max_s for d in slept)


def test_non_retryable_4xx_fails_immediately(config: Config, tiny_wav: Path) -> None:
    """A 422 is a client mistake; retrying it just wastes time."""
    body = {"error": {"message": "audio too long", "code": "unprocessable_entity_error",
                      "request_id": "req-422"}}
    client, session, _ = build_client(config, [FakeResponse(422, body)])
    with pytest.raises(SarvamAPIError) as exc:
        client.speech_to_text(tiny_wav, audio_duration_s=0.5)
    assert len(session.calls) == 1
    assert exc.value.status_code == 422
    assert exc.value.request_id == "req-422"


def test_403_error_message_names_the_actual_cause(config: Config, tiny_wav: Path) -> None:
    """Sarvam returns 403 for a bad key; the error must say so instead of 'forbidden'."""
    client, _, _ = build_client(config, [FakeResponse(403, {"error": {"message": "bad key"}})])
    with pytest.raises(SarvamAPIError, match="403"):
        client.speech_to_text(tiny_wav, audio_duration_s=0.5)


def test_failed_calls_are_not_cached(config: Config, tiny_wav: Path) -> None:
    """An error must never poison the cache and be replayed forever."""
    cache = DiskCache(config.cache_dir)
    client, _, _ = build_client(config, [FakeResponse(422, {})], cache=cache)
    with pytest.raises(SarvamAPIError):
        client.speech_to_text(tiny_wav, audio_duration_s=0.5)
    assert cache.stats.writes == 0


# --- input validation ------------------------------------------------------------------

def test_audio_over_30s_is_rejected_before_any_network_call(config: Config, tiny_wav: Path) -> None:
    """The sync endpoint caps at 30s; catch it locally rather than paying for a 422."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="30s"):
        client.speech_to_text(tiny_wav, audio_duration_s=31.0)
    assert session.calls == []


def test_unknown_mode_is_rejected(config: Config, tiny_wav: Path) -> None:
    """Only the five documented saaras:v3 modes are accepted."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="mode"):
        client.speech_to_text(tiny_wav, mode="summarise", audio_duration_s=0.5)
    assert session.calls == []


def test_unknown_language_code_is_rejected(config: Config, tiny_wav: Path) -> None:
    """A typo in a language code should fail locally, not after a paid round-trip."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="language_code"):
        client.speech_to_text(tiny_wav, language_code="en-US", audio_duration_s=0.5)
    assert session.calls == []


def test_missing_file_is_rejected(config: Config, tmp_path: Path) -> None:
    """A missing input file fails with a path in the message."""
    client, _, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="not found"):
        client.speech_to_text(tmp_path / "nope.wav", audio_duration_s=0.5)
