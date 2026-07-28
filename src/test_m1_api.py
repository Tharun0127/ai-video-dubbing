"""
M1 live acceptance test: one real Saaras v3 ASR call, then the same call from cache.

This is the executable proof behind the M1 claim "a second run over the same input
with identical parameters costs Rs 0 and issues 0 network calls". It makes exactly
one billable call and reports only measured numbers:

  RUN 1  cold cache -> real HTTP POST /speech-to-text, real latency, real rupees
  RUN 2  warm cache -> served from disk with an armed socket guard that raises on
                       ANY outbound connection attempt, so "zero network" is proven
                       rather than asserted

Usage:
    python -m src.test_m1_api                       # evicts the entry, spends ~Rs 0.08
    python -m src.test_m1_api --reuse-cache         # re-prove from warm cache, Rs 0
    python -m src.test_m1_api --input path/to.wav

Input:  an audio file <= 30 s (default samples/test_audio.wav) and SARVAM_API_KEY in .env.
Output: a printed measured-results table plus output/m1_api_test.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from .audio import probe_duration_s
from .cache import DiskCache, cache_key, hash_file
from .config import Config, load_config, setup_logging
from .metrics import MetricsCollector, cost_stt_inr
from .sarvam_client import SarvamClient

logger = logging.getLogger(__name__)

#: Keep pytest from ever collecting this module as a unit test -- it spends real money.
__test__ = False

DEFAULT_INPUT = Path("samples/test_audio.wav")
REPORT_PATH = Path("output/m1_api_test.json")


class NetworkAccessError(AssertionError):
    """Raised when code attempts a network connection while the socket guard is armed."""


@contextmanager
def no_network_allowed() -> Generator[None]:
    """Block every outbound socket connection, so a 'cache hit' claim can be proven."""
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def blocked_connect(self: socket.socket, address: Any) -> None:
        raise NetworkAccessError(f"network access attempted to {address!r} while cache-only")

    def blocked_create(address: Any, *args: Any, **kwargs: Any) -> None:
        raise NetworkAccessError(f"network access attempted to {address!r} while cache-only")

    socket.socket.connect = blocked_connect  # type: ignore[method-assign]
    socket.create_connection = blocked_create  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.create_connection = real_create  # type: ignore[assignment]


@dataclass
class RunResult:
    """Everything measured about one ASR call, whether it hit the network or the cache."""

    label: str
    cache_status: str
    latency_ms: float
    cost_inr: float
    cost_if_billed_inr: float
    network_calls: int
    cache_hits: int
    cache_writes: int
    http_status: int | None
    http_attempts: int
    request_id: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the run for output/m1_api_test.json (payload kept separately)."""
        return {
            "run": self.label,
            "cache_status": self.cache_status,
            "latency_ms": round(self.latency_ms, 3),
            "cost_inr": round(self.cost_inr, 6),
            "cost_if_billed_inr": round(self.cost_if_billed_inr, 6),
            "network_calls": self.network_calls,
            "cache_hits": self.cache_hits,
            "cache_writes": self.cache_writes,
            "http_status": self.http_status,
            "http_attempts": self.http_attempts,
            "request_id": self.request_id,
        }


def _asr_once(
    label: str,
    config: Config,
    audio_path: Path,
    *,
    mode: str,
    language_code: str,
    duration_s: float,
    block_network: bool,
) -> tuple[RunResult, MetricsCollector]:
    """Make one ASR call through a fresh client/cache/metrics trio and measure the outcome."""
    cache = DiskCache(config.cache_dir)
    metrics = MetricsCollector()

    # Each run gets its own guard scope so run 1 can use the network and run 2 cannot.
    guard = no_network_allowed() if block_network else _no_guard()
    with guard:
        with SarvamClient(config, cache=cache, metrics=metrics) as client:
            with metrics.time_stage("asr"):
                payload = client.speech_to_text(
                    audio_path,
                    mode=mode,
                    language_code=language_code,
                    audio_duration_s=duration_s,
                )

    call = metrics.api_calls[-1]
    result = RunResult(
        label=label,
        cache_status="HIT" if call.cache_hit else "MISS",
        latency_ms=call.latency_s * 1000.0,
        cost_inr=call.cost_inr,
        cost_if_billed_inr=call.cost_if_billed_inr,
        network_calls=metrics.cache_misses,
        cache_hits=metrics.cache_hits,
        cache_writes=cache.stats.writes,
        http_status=call.http_status,
        http_attempts=call.attempts if not call.cache_hit else 0,
        request_id=call.request_id,
        payload=payload,
    )
    logger.info(
        "%s -> %s | %.3f ms | Rs %.6f | %d network call(s)",
        label, result.cache_status, result.latency_ms, result.cost_inr, result.network_calls,
    )
    return result, metrics


@contextmanager
def _no_guard() -> Generator[None]:
    """No-op context manager, so the guarded and unguarded paths share one code path."""
    yield


def run_m1_api_test(
    audio_path: str | Path = DEFAULT_INPUT,
    *,
    mode: str = "transcribe",
    language_code: str = "en-IN",
    cache_dir: str | Path = "cache",
    reuse_cache: bool = False,
    report_path: str | Path = REPORT_PATH,
) -> dict[str, Any]:
    """Call Saaras v3 twice on one clip and prove the second call is a free cache hit."""
    audio = Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"test audio not found: {audio}")

    duration_s = probe_duration_s(audio)
    list_price = cost_stt_inr(duration_s)

    base_kwargs: dict[str, Any] = {
        "input_path": audio,
        "source_lang": language_code,
        "cache_dir": Path(cache_dir),
    }
    config = load_config(**base_kwargs)

    # Recompute the key the client will derive, so we can show it and evict it on demand.
    params = {
        "model": config.asr_model,
        "mode": mode,
        "language_code": language_code,
        "input_audio_codec": None,
    }
    key = cache_key("asr", config.asr_model, params, hash_file(audio))
    entry = Path(cache_dir) / "asr" / f"{key}.json"

    print(f"\ninput            : {audio} ({duration_s:.3f}s measured, {audio.stat().st_size:,} bytes)")
    print(f"model / mode     : {config.asr_model} / {mode}   language_code={language_code}")
    print(f"list price (STT) : Rs {list_price:.6f}  (Rs 30/hour, billed per second, rounded up)")
    print(f"cache key        : {key}")
    print(f"cache entry      : {entry}")

    cold_call_expected = True
    if reuse_cache:
        cold_call_expected = not entry.exists()
        print("--reuse-cache    : keeping any existing entry, so this re-proof costs Rs 0\n")
    elif entry.exists():
        entry.unlink()
        print("evicted the existing entry so RUN 1 is a genuine cold call\n")
    else:
        print("no existing entry; RUN 1 is already cold\n")

    print("-" * 72)
    print("RUN 1 -- cold cache: expect a real network call to /speech-to-text")
    print("-" * 72)
    run1, metrics1 = _asr_once(
        "RUN 1 (cold cache)", load_config(**base_kwargs), audio,
        mode=mode, language_code=language_code, duration_s=duration_s, block_network=False,
    )

    print("\n" + "-" * 72)
    print("RUN 2 -- warm cache, socket guard armed: expect a HIT and zero network traffic")
    print("-" * 72)
    run2, metrics2 = _asr_once(
        "RUN 2 (warm cache)", load_config(**base_kwargs), audio,
        mode=mode, language_code=language_code, duration_s=duration_s, block_network=True,
    )

    identical = run1.payload == run2.payload

    print("\n" + "=" * 72)
    print("MEASURED RESULTS")
    print("=" * 72)
    header = f"{'run':<20}{'cache':<8}{'latency_ms':>13}{'cost_INR':>12}{'net calls':>11}"
    print(header)
    print("-" * len(header))
    for row in (run1, run2):
        print(f"{row.label:<20}{row.cache_status:<8}"
              f"{row.latency_ms:>13.3f}{row.cost_inr:>12.6f}{row.network_calls:>11}")
    print("-" * len(header))

    speedup = run1.latency_ms / run2.latency_ms if run2.latency_ms > 0 else None
    if speedup:
        print(f"cache speedup    : {speedup:,.0f}x")
    print(f"cost avoided     : Rs {run2.cost_if_billed_inr:.6f} on run 2")
    print(f"total spent      : Rs {run1.cost_inr + run2.cost_inr:.6f}")

    print(f"\ntranscript       : {run1.payload.get('transcript', '')!r}")
    print(f"detected language: {run1.payload.get('language_code')!r}")
    print(f"response fields  : {sorted(run1.payload.keys())}")
    print(f"payloads match   : {identical}")

    checks = {
        # Under --reuse-cache run 1 is legitimately a hit, so only demand a cold call
        # when the entry was actually evicted; otherwise a free re-proof reports FAIL.
        "run1_was_network_call": run1.cache_status == "MISS" if cold_call_expected else True,
        "run1_http_200": run1.http_status == 200 if cold_call_expected else True,
        "run1_transcript_non_empty": bool(str(run1.payload.get("transcript", "")).strip()),
        "run1_response_cached_to_disk": entry.exists(),
        "run2_was_cache_hit": run2.cache_status == "HIT",
        "run2_zero_network_calls": run2.network_calls == 0,
        "run2_zero_cost": run2.cost_inr == 0.0,
        "run2_faster_than_network": run2.latency_ms < run1.latency_ms or not cold_call_expected,
        "payloads_identical": identical,
    }
    print("\nACCEPTANCE CHECKS")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    metrics1.log_summary()

    report = {
        "input": {
            "path": str(audio),
            "duration_s": round(duration_s, 6),
            "size_bytes": audio.stat().st_size,
            "sha256": hash_file(audio),
        },
        "model": config.asr_model,
        "mode": mode,
        "language_code": language_code,
        "list_price_inr": round(list_price, 6),
        "cache_key": key,
        "cache_entry_path": str(entry),
        "runs": [run1.to_dict(), run2.to_dict()],
        "cache_speedup_x": round(speedup, 2) if speedup else None,
        "transcript": run1.payload.get("transcript"),
        "detected_language_code": run1.payload.get("language_code"),
        "payloads_identical": identical,
        "checks": checks,
        "run1_metrics": metrics1.to_dict(),
        "run2_metrics": metrics2.to_dict(),
    }
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")

    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: returns 0 only when every acceptance check passes."""
    parser = argparse.ArgumentParser(
        prog="python -m src.test_m1_api",
        description="M1 live test: one real Saaras v3 call, then the same call from cache.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mode", default="transcribe")
    parser.add_argument("--language-code", default="en-IN")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Keep any existing entry; re-proves the cache path for Rs 0.")
    parser.add_argument("--verbose", action="store_true",
                        help="DEBUG logging, including API request/response detail.")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    try:
        report = run_m1_api_test(
            args.input,
            mode=args.mode,
            language_code=args.language_code,
            cache_dir=args.cache_dir,
            reuse_cache=args.reuse_cache,
        )
    except Exception:
        logger.exception(
            "M1 API test failed. Last successful checkpoint: cache dir %s "
            "(inspect it to see which entries exist before re-running).", args.cache_dir,
        )
        return 1

    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
