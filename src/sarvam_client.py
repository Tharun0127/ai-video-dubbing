"""
Sarvam API client: the single HTTP gateway for the whole pipeline.

Every Sarvam call in this repo goes through this module. It owns, in one place:
auth, timeouts, retries with exponential backoff and full jitter, 429/503
handling, the disk cache, per-call latency measurement, and cost accounting.
No other module is permitted to make a raw HTTP request.

Cache semantics: a result is looked up before any socket is opened. On a hit the
call costs zero rupees and issues zero network traffic. Under --dry-run a miss is
a hard failure (CacheMissError) rather than a silent network call.

Confirmed against docs.sarvam.ai on 2026-07-28 (see docs/api-notes.md):
  POST https://api.sarvam.ai/speech-to-text   multipart/form-data
  header: api-subscription-key            (auth failure returns 403, not 401)
  fields: file, model, mode, language_code, input_audio_codec
  limits: 30 s of audio per request; 16 kHz mono WAV recommended

Input:  a Config, a DiskCache, and a MetricsCollector.
Output: parsed API response dicts, with every call recorded in metrics.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

import requests

from .cache import CacheMissError, DiskCache, cache_key, hash_file
from .config import (
    ASR_LANGUAGES,
    ASR_MAX_AUDIO_S,
    ASR_MODES,
    AUTH_HEADER,
    Config,
)
from .metrics import ApiCall, MetricsCollector, cost_stt_inr

logger = logging.getLogger(__name__)

#: Status codes worth retrying: rate limit, and transient backend failures.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class SarvamAPIError(RuntimeError):
    """A Sarvam API call failed. Carries the status, body, and request id for post-mortems."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 body: str | None = None, request_id: str | None = None) -> None:
        """Build an error that prints the full API context, not just a status number."""
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.request_id = request_id

    def __str__(self) -> str:  # pragma: no cover - formatting only
        """Render status, request id, and response body alongside the message."""
        parts = [super().__str__()]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        if self.body:
            parts.append(f"body={self.body[:800]}")
        return " | ".join(parts)


class SarvamClient:
    """The one HTTP client: auth, retries, backoff, caching, cost accounting, latency."""

    def __init__(
        self,
        config: Config,
        cache: DiskCache | None = None,
        metrics: MetricsCollector | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Wire the client to its config, cache, and metrics; `sleep` is injectable for tests."""
        self.config = config
        self.cache = cache if cache is not None else DiskCache(config.cache_dir)
        self.metrics = metrics if metrics is not None else MetricsCollector()
        self._sleep = sleep
        self._session = session or requests.Session()
        self._session.headers.update({
            AUTH_HEADER: config.api_key,
            "User-Agent": "sarvam-video-dubbing/0.1 (+portfolio)",
        })

    # ----------------------------------------------------------------------------------
    # Public API surface
    # ----------------------------------------------------------------------------------

    def speech_to_text(
        self,
        audio_path: str | Path,
        *,
        model: str | None = None,
        mode: str = "transcribe",
        language_code: str | None = None,
        input_audio_codec: str | None = None,
        audio_duration_s: float | None = None,
        stage: str = "asr",
    ) -> dict[str, Any]:
        """Transcribe (or translate) one <=30s audio file via POST /speech-to-text."""
        path = Path(audio_path)
        if not path.exists():
            raise SarvamAPIError(f"audio file not found: {path}")
        if mode not in ASR_MODES:
            raise SarvamAPIError(f"mode {mode!r} is not one of {sorted(ASR_MODES)}")
        if language_code is not None and language_code not in ASR_LANGUAGES:
            raise SarvamAPIError(f"language_code {language_code!r} is not supported by Sarvam")

        if audio_duration_s is None:
            from .audio import probe_duration_s  # imported lazily: ffprobe only needed here

            audio_duration_s = probe_duration_s(path)

        if audio_duration_s > ASR_MAX_AUDIO_S:
            raise SarvamAPIError(
                f"audio is {audio_duration_s:.2f}s but the sync /speech-to-text endpoint caps at "
                f"{ASR_MAX_AUDIO_S:.0f}s (HTTP 422). Chunk it first (M2) or use the Batch API."
            )

        resolved_model = model or self.config.asr_model
        params: dict[str, Any] = {
            "model": resolved_model,
            "mode": mode,
            "language_code": language_code,
            "input_audio_codec": input_audio_codec,
        }

        def do_request() -> tuple[dict[str, Any], int, bytes | None]:
            """Perform the multipart POST; returns (payload, status, binary=None)."""
            form = {k: v for k, v in params.items() if v is not None and k != "file"}
            with open(path, "rb") as handle:
                files = {"file": (path.name, handle, "audio/wav")}
                response = self._request("POST", "/speech-to-text", data=form, files=files)
            return response.json(), response.status_code, None

        payload, was_cached = self._cached_call(
            stage=stage,
            endpoint="/speech-to-text",
            model=resolved_model,
            params=params,
            input_hash=hash_file(path),
            billable_units=audio_duration_s,
            billable_unit_name="audio_seconds",
            list_price_inr=cost_stt_inr(audio_duration_s),
            do_request=do_request,
        )
        logger.info(
            "ASR %s (%s, %.2fs audio) -> %s",
            path.name, "CACHE HIT" if was_cached else "network call", audio_duration_s,
            f"{len(payload.get('transcript', ''))} chars",
        )
        return payload

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> SarvamClient:
        """Support use as a context manager so the session is always closed."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the session on context exit."""
        self.close()

    # ----------------------------------------------------------------------------------
    # Cache + cost wrapper
    # ----------------------------------------------------------------------------------

    def _cached_call(
        self,
        *,
        stage: str,
        endpoint: str,
        model: str,
        params: dict[str, Any],
        input_hash: str,
        billable_units: float,
        billable_unit_name: str,
        list_price_inr: float,
        do_request: Callable[[], tuple[dict[str, Any], int, bytes | None]],
    ) -> tuple[dict[str, Any], bool]:
        """Serve a call from cache if possible, else hit the network; record cost either way."""
        key = cache_key(stage, model, params, input_hash)
        started = time.perf_counter()

        cached = self.cache.get(stage, key)
        if cached is not None:
            payload = cached["payload"]
            self.metrics.record_api_call(ApiCall(
                stage=stage,
                endpoint=endpoint,
                model=model,
                cache_hit=True,
                latency_s=time.perf_counter() - started,
                cost_inr=0.0,
                cost_if_billed_inr=list_price_inr,
                billable_units=billable_units,
                billable_unit_name=billable_unit_name,
                http_status=None,
                attempts=0,
                request_id=payload.get("request_id") if isinstance(payload, dict) else None,
            ))
            return payload, True

        if self.config.dry_run:
            raise CacheMissError(
                f"--dry-run requested but {stage} call is not cached "
                f"(endpoint={endpoint} model={model} key={key[:16]}). "
                f"Run once without --dry-run to populate the cache."
            )

        payload, status, binary = do_request()
        latency = time.perf_counter() - started

        self.cache.put(
            stage, key, model=model, params=params,
            input_hash=input_hash, payload=payload, binary=binary,
        )
        self.metrics.record_api_call(ApiCall(
            stage=stage,
            endpoint=endpoint,
            model=model,
            cache_hit=False,
            latency_s=latency,
            cost_inr=list_price_inr,
            cost_if_billed_inr=list_price_inr,
            billable_units=billable_units,
            billable_unit_name=billable_unit_name,
            http_status=status,
            attempts=self._last_attempts,
            request_id=payload.get("request_id") if isinstance(payload, dict) else None,
        ))
        return payload, False

    # ----------------------------------------------------------------------------------
    # HTTP with retries
    # ----------------------------------------------------------------------------------

    _last_attempts: int = 0

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        """Issue one HTTP request with exponential backoff on 429/5xx and network errors."""
        url = f"{self.config.base_url}{endpoint}"
        max_attempts = self.config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            self._last_attempts = attempt
            logger.debug("HTTP %s %s attempt %d/%d", method, url, attempt, max_attempts)
            try:
                response = self._session.request(
                    method, url, timeout=self.config.request_timeout_s, **kwargs
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= max_attempts:
                    raise SarvamAPIError(
                        f"{method} {endpoint} failed after {attempt} attempt(s): {exc}"
                    ) from exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "%s %s network error on attempt %d/%d (%s); retrying in %.2fs",
                    method, endpoint, attempt, max_attempts, exc, delay,
                )
                self._sleep(delay)
                continue

            logger.debug("HTTP %s -> %d (%d bytes)", endpoint, response.status_code,
                         len(response.content or b""))

            if response.status_code < 400:
                return response

            body = self._safe_body(response)
            request_id = self._extract_request_id(response)

            if response.status_code in RETRYABLE_STATUS and attempt < max_attempts:
                delay = self._retry_after(response) or self._backoff_delay(attempt)
                logger.warning(
                    "%s %s returned %d on attempt %d/%d; retrying in %.2fs (body=%s)",
                    method, endpoint, response.status_code, attempt, max_attempts,
                    delay, body[:200],
                )
                self._sleep(delay)
                continue

            hint = ""
            if response.status_code == 403:
                hint = (" Sarvam returns 403 (not 401) for a bad or missing api-subscription-key; "
                        f"check SARVAM_API_KEY (using {self.config.key_fingerprint}).")
            elif response.status_code == 422:
                hint = " Check the audio format and that the clip is under 30 seconds."
            elif response.status_code == 429:
                hint = " Rate limited after all retries; reduce concurrency or wait."

            raise SarvamAPIError(
                f"{method} {endpoint} failed after {attempt} attempt(s).{hint}",
                status_code=response.status_code, body=body, request_id=request_id,
            )

        raise SarvamAPIError(  # pragma: no cover - loop always returns or raises
            f"{method} {endpoint} exhausted retries: {last_error}"
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter, capped at backoff_max_s."""
        ceiling = min(self.config.backoff_base_s * (2 ** (attempt - 1)), self.config.backoff_max_s)
        return random.uniform(0.0, ceiling)

    @staticmethod
    def _retry_after(response: requests.Response) -> float | None:
        """Honour a Retry-After header when the server sends one."""
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    @staticmethod
    def _safe_body(response: requests.Response) -> str:
        """Return the response body as text, never raising while building an error message."""
        try:
            return response.text or ""
        except Exception:  # pragma: no cover - defensive
            return "<unreadable body>"

    @staticmethod
    def _extract_request_id(response: requests.Response) -> str | None:
        """Pull Sarvam's request_id out of an error body when present."""
        try:
            parsed = response.json()
        except (ValueError, json.JSONDecodeError):
            return None
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                return error.get("request_id")
            return parsed.get("request_id")
        return None
