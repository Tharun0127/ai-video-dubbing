"""
Sarvam API client: the single HTTP gateway for the whole pipeline.

Every Sarvam call in this repo goes through this module. It owns, in one place:
auth, timeouts, retries with exponential backoff and full jitter, 429/503
handling, the disk cache, per-call latency measurement, and cost accounting.
No other module is permitted to make a raw HTTP request.

Cache semantics: a result is looked up before any socket is opened. On a hit the
call costs zero rupees and issues zero network traffic. Under --dry-run a miss is
a hard failure (CacheMissError) rather than a silent network call.

Confirmed against docs.sarvam.ai and real calls (see docs/api-notes.md):
  POST https://api.sarvam.ai/speech-to-text   multipart/form-data   (2026-07-28)
  header: api-subscription-key            (auth failure returns 403, not 401)
  fields: file, model, mode, language_code, input_audio_codec
  limits: 30 s of audio per request; 16 kHz mono WAV recommended

  POST https://api.sarvam.ai/translate       application/json       (2026-07-29)
  fields: input, source_language_code, target_language_code, model, mode,
          speaker_gender, output_script, numerals_format
  returns: {request_id, translated_text, source_language_code}
  limits: 1000 chars for mayura:v1, 2000 for sarvam-translate:v1 (HTTP 422 beyond);
          style modes are honoured only by mayura:v1

Input:  a Config, a DiskCache, and a MetricsCollector.
Output: parsed API response dicts, with every call recorded in metrics.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

import requests

from .cache import CacheMissError, DiskCache, cache_key, hash_file, hash_text
from .config import (
    ASR_LANGUAGES,
    ASR_MAX_AUDIO_S,
    ASR_MODES,
    AUTH_HEADER,
    TRANSLATE_MODEL_LANGUAGES,
    TRANSLATE_MODEL_MAX_CHARS,
    TRANSLATE_MODEL_MODES,
    TRANSLATE_MODELS,
    TRANSLATE_NUMERALS_FORMATS,
    TRANSLATE_OUTPUT_SCRIPTS,
    TRANSLATE_SPEAKER_GENDERS,
    TTS_API_PACE_MAX,
    TTS_API_PACE_MIN,
    TTS_LANGUAGES,
    TTS_MAX_CHARS,
    TTS_MODELS,
    TTS_OUTPUT_CODEC,
    TTS_SAMPLE_RATES,
    TTS_SPEAKERS_V3,
    Config,
)
from .metrics import (
    ApiCall,
    MetricsCollector,
    cost_stt_inr,
    cost_translate_inr,
    cost_tts_inr,
)

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
        with_timestamps: bool | None = None,
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
            "with_timestamps": with_timestamps,
        }

        def do_request() -> tuple[dict[str, Any], int, bytes | None]:
            """Perform the multipart POST; returns (payload, status, binary=None)."""
            # Booleans must go over the wire as the lowercase JSON spelling; requests
            # would otherwise send Python's "True", which the API rejects.
            form = {
                k: ("true" if v is True else "false" if v is False else v)
                for k, v in params.items() if v is not None and k != "file"
            }
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

    def translate(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
        model: str | None = None,
        mode: str | None = None,
        speaker_gender: str | None = None,
        output_script: str | None = None,
        numerals_format: str | None = None,
        stage: str = "translate",
    ) -> dict[str, Any]:
        """Translate one text via POST /translate; returns the parsed response dict."""
        resolved_model = model or self.config.translate_model
        if resolved_model not in TRANSLATE_MODELS:
            raise SarvamAPIError(
                f"model {resolved_model!r} is not one of {sorted(TRANSLATE_MODELS)}"
            )
        if not text.strip():
            raise SarvamAPIError("refusing to translate empty text; it would bill for nothing")

        max_chars = TRANSLATE_MODEL_MAX_CHARS[resolved_model]
        if len(text) > max_chars:
            raise SarvamAPIError(
                f"input is {len(text)} chars but {resolved_model} caps at {max_chars} "
                f"(HTTP 422 beyond it). Batch fewer segments per request."
            )

        allowed_modes = TRANSLATE_MODEL_MODES[resolved_model]
        if mode is not None:
            if mode not in allowed_modes:
                # The API accepts an unsupported mode and silently ignores it, so this
                # check is the only thing standing between a "colloquial" run and a
                # formal one that merely claims to be colloquial.
                raise SarvamAPIError(
                    f"mode {mode!r} is not supported by {resolved_model}; it honours "
                    f"{sorted(allowed_modes)}"
                )
        languages = TRANSLATE_MODEL_LANGUAGES[resolved_model]
        for code, label in ((source_language_code, "source"), (target_language_code, "target")):
            if code not in languages:
                raise SarvamAPIError(
                    f"{label}_language_code {code!r} is not supported by {resolved_model}"
                )
        if speaker_gender is not None and speaker_gender not in TRANSLATE_SPEAKER_GENDERS:
            raise SarvamAPIError(
                f"speaker_gender {speaker_gender!r} is not one of {sorted(TRANSLATE_SPEAKER_GENDERS)}"
            )
        if output_script is not None and output_script not in TRANSLATE_OUTPUT_SCRIPTS:
            raise SarvamAPIError(
                f"output_script {output_script!r} is not one of {sorted(TRANSLATE_OUTPUT_SCRIPTS)}"
            )
        if numerals_format is not None and numerals_format not in TRANSLATE_NUMERALS_FORMATS:
            raise SarvamAPIError(
                f"numerals_format {numerals_format!r} is not one of "
                f"{sorted(TRANSLATE_NUMERALS_FORMATS)}"
            )

        params: dict[str, Any] = {
            "model": resolved_model,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
            "mode": mode,
            "speaker_gender": speaker_gender,
            "output_script": output_script,
            "numerals_format": numerals_format,
        }

        def do_request() -> tuple[dict[str, Any], int, bytes | None]:
            """Perform the JSON POST; returns (payload, status, binary=None)."""
            body = {k: v for k, v in params.items() if v is not None}
            body["input"] = text
            response = self._request("POST", "/translate", json=body)
            return response.json(), response.status_code, None

        payload, was_cached = self._cached_call(
            stage=stage,
            endpoint="/translate",
            model=resolved_model,
            params=params,
            input_hash=hash_text(text),
            billable_units=float(len(text)),
            billable_unit_name="input_characters",
            list_price_inr=cost_translate_inr(len(text)),
            do_request=do_request,
        )
        logger.info(
            "translate %s (%s, %d chars in -> %d chars out)",
            f"{source_language_code}->{target_language_code} mode={mode or 'default'}",
            "CACHE HIT" if was_cached else "network call",
            len(text), len(payload.get("translated_text", "")),
        )
        return payload

    def text_to_speech(
        self,
        text: str,
        *,
        target_language_code: str,
        model: str | None = None,
        speaker: str | None = None,
        pace: float = 1.0,
        speech_sample_rate: int | None = None,
        output_audio_codec: str | None = None,
        stage: str = "tts",
    ) -> dict[str, Any]:
        """Synthesise one text via POST /text-to-speech; returns the parsed response dict.

        `pace` is part of the cache key, so each attempt of the duration-fit loop is a
        distinct entry and re-running the whole loop costs nothing.
        """
        resolved_model = model or self.config.tts_model
        if resolved_model not in TTS_MODELS:
            raise SarvamAPIError(f"model {resolved_model!r} is not one of {sorted(TTS_MODELS)}")
        if not text.strip():
            raise SarvamAPIError("refusing to synthesise empty text; it would bill for nothing")
        if len(text) > TTS_MAX_CHARS:
            raise SarvamAPIError(
                f"input is {len(text)} chars but {resolved_model} caps at {TTS_MAX_CHARS} "
                f"(HTTP 422 beyond it). Split the segment first."
            )
        if target_language_code not in TTS_LANGUAGES:
            raise SarvamAPIError(
                f"target_language_code {target_language_code!r} is not supported by "
                f"{resolved_model}; it speaks {sorted(TTS_LANGUAGES)}"
            )

        resolved_speaker = speaker or self.config.tts_speaker
        if resolved_model == "bulbul:v3" and resolved_speaker not in TTS_SPEAKERS_V3:
            raise SarvamAPIError(
                f"speaker {resolved_speaker!r} is not a bulbul:v3 voice (lowercase, "
                f"model-specific): {sorted(TTS_SPEAKERS_V3)}"
            )
        if not TTS_API_PACE_MIN <= pace <= TTS_API_PACE_MAX:
            raise SarvamAPIError(
                f"pace {pace} is outside the API's documented range "
                f"[{TTS_API_PACE_MIN}, {TTS_API_PACE_MAX}]"
            )

        rate = speech_sample_rate if speech_sample_rate is not None else self.config.tts_sample_rate
        if rate not in TTS_SAMPLE_RATES:
            raise SarvamAPIError(
                f"speech_sample_rate {rate} is not one of {sorted(TTS_SAMPLE_RATES)}"
            )

        # Rounded so that float noise in the fit loop's pace update cannot produce two
        # cache keys for what is audibly the same request.
        quantised_pace = round(float(pace), 4)

        params: dict[str, Any] = {
            "model": resolved_model,
            "target_language_code": target_language_code,
            "speaker": resolved_speaker,
            "pace": quantised_pace,
            "speech_sample_rate": rate,
            "output_audio_codec": output_audio_codec or TTS_OUTPUT_CODEC,
        }

        def do_request() -> tuple[dict[str, Any], int, bytes | None]:
            """Perform the JSON POST; returns (payload, status, binary=None)."""
            body = {k: v for k, v in params.items() if v is not None}
            body["text"] = text
            response = self._request("POST", "/text-to-speech", json=body)
            return response.json(), response.status_code, None

        payload, was_cached = self._cached_call(
            stage=stage,
            endpoint="/text-to-speech",
            model=resolved_model,
            params=params,
            input_hash=hash_text(text),
            billable_units=float(len(text)),
            billable_unit_name="input_characters",
            list_price_inr=cost_tts_inr(len(text)),
            do_request=do_request,
        )
        logger.info(
            "tts %s pace=%.4f (%s, %d chars)",
            target_language_code, quantised_pace,
            "CACHE HIT" if was_cached else "network call", len(text),
        )
        return payload

    @staticmethod
    def audio_bytes(payload: dict[str, Any]) -> bytes:
        """Decode the first entry of a /text-to-speech response's base64 `audios` array."""
        audios = payload.get("audios")
        if not isinstance(audios, list) or not audios:
            raise SarvamAPIError(
                f"/text-to-speech response has no audio: keys={sorted(payload)}"
            )
        try:
            return base64.b64decode(audios[0], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SarvamAPIError(f"audios[0] is not valid base64: {exc}") from exc

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
