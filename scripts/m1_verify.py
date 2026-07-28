"""
Milestone 1 acceptance harness: prove the client, cache, and metrics actually work.

Runs three checks against a real 10-second clip and reports only measured numbers:

  RUN 1  identical call, cold cache  -> must hit the network, cost real rupees
  RUN 2  identical call, warm cache  -> must be a cache hit with ZERO network traffic,
                                        enforced by a socket guard that raises on any
                                        outbound connection attempt
  RUN 3  identical call under --dry-run against a key that was never cached
                                     -> must fail loudly with CacheMissError

Usage:
    python -m scripts.m1_verify --input samples/jfk_10s_16k_mono.wav
    python -m scripts.m1_verify --input samples/jfk_10s_16k_mono.wav --reuse-cache
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio import probe_duration_s  # noqa: E402
from src.cache import CacheMissError, DiskCache, cache_key, hash_file  # noqa: E402
from src.config import load_config, setup_logging  # noqa: E402
from src.metrics import MetricsCollector, cost_stt_inr  # noqa: E402
from src.sarvam_client import SarvamClient  # noqa: E402


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


def summarise(label: str, metrics: MetricsCollector, cache: DiskCache) -> dict[str, Any]:
    """Collect the measured outcome of a single run into a plain dict."""
    call = metrics.api_calls[-1]
    return {
        "run": label,
        "cache_status": "HIT" if call.cache_hit else "MISS (network call)",
        "latency_s": round(call.latency_s, 4),
        "cost_inr": round(call.cost_inr, 6),
        "cost_if_billed_inr": round(call.cost_if_billed_inr, 6),
        "billable": f"{call.billable_units:g} {call.billable_unit_name}",
        "http_status": call.http_status,
        "http_attempts": call.attempts,
        "request_id": call.request_id,
        "network_calls_this_run": metrics.cache_misses,
        "cache_hits_this_run": metrics.cache_hits,
        "cache_writes_this_run": cache.stats.writes,
    }


def main(argv: list[str] | None = None) -> int:
    """Execute the three M1 checks and print the measured results table."""
    parser = argparse.ArgumentParser(description="Milestone 1 acceptance harness.")
    parser.add_argument("--input", type=Path, default=Path("samples/jfk_10s_16k_mono.wav"))
    parser.add_argument("--source-lang", default="en-IN")
    parser.add_argument("--target-lang", default="hi-IN")
    parser.add_argument("--mode", default="transcribe")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Do not evict the entry first; use to re-prove a warm cache for free.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    if not args.input.exists():
        print(f"ERROR: input clip not found: {args.input}", file=sys.stderr)
        return 1

    duration_s = probe_duration_s(args.input)
    print(f"\ninput            : {args.input} ({duration_s:.3f}s measured)")
    print(f"list price (STT) : Rs {cost_stt_inr(duration_s):.6f} at Rs30/hour\n")

    base_kwargs = dict(
        input_path=args.input,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        cache_dir=args.cache_dir,
    )

    # The exact key both runs must agree on.
    probe_cfg = load_config(**base_kwargs)  # type: ignore[arg-type]
    params = {
        "model": probe_cfg.asr_model,
        "mode": args.mode,
        "language_code": args.source_lang,
        "input_audio_codec": None,
    }
    key = cache_key("asr", probe_cfg.asr_model, params, hash_file(args.input))
    entry = args.cache_dir / "asr" / f"{key}.json"
    print(f"cache key        : {key}")
    print(f"cache entry path : {entry}\n")

    cold_call_expected = True
    if args.reuse_cache:
        cold_call_expected = not entry.exists()
        print("--reuse-cache: keeping any existing entry, so this re-proof costs Rs 0\n")
    elif entry.exists():
        entry.unlink()
        print("evicted existing cache entry so RUN 1 is a genuine cold call\n")

    results: list[dict[str, Any]] = []

    # ---- RUN 1: cold cache, real network call ----------------------------------------
    print("-" * 72)
    print("RUN 1 -- cold cache: expect a real network call")
    print("-" * 72)
    cfg1 = load_config(**base_kwargs)  # type: ignore[arg-type]
    metrics1, cache1 = MetricsCollector(), DiskCache(args.cache_dir)
    with SarvamClient(cfg1, cache=cache1, metrics=metrics1) as client:
        with metrics1.time_stage("asr"):
            payload1 = client.speech_to_text(
                args.input, mode=args.mode, language_code=args.source_lang,
                audio_duration_s=duration_s,
            )
    results.append(summarise("1 (cold cache)", metrics1, cache1))
    print(f"\ntranscript: {payload1.get('transcript', '')!r}")
    print(f"language_code: {payload1.get('language_code')!r}")
    print(f"response keys: {sorted(payload1.keys())}\n")

    # ---- RUN 2: warm cache, network physically blocked --------------------------------
    print("-" * 72)
    print("RUN 2 -- warm cache with socket guard armed: expect a cache hit, zero network")
    print("-" * 72)
    cfg2 = load_config(**base_kwargs)  # type: ignore[arg-type]
    metrics2, cache2 = MetricsCollector(), DiskCache(args.cache_dir)
    with no_network_allowed():
        with SarvamClient(cfg2, cache=cache2, metrics=metrics2) as client:
            with metrics2.time_stage("asr"):
                payload2 = client.speech_to_text(
                    args.input, mode=args.mode, language_code=args.source_lang,
                    audio_duration_s=duration_s,
                )
    results.append(summarise("2 (warm cache)", metrics2, cache2))

    identical = payload1 == payload2
    print(f"\npayloads identical across runs: {identical}")

    # ---- RUN 3: --dry-run against an uncached key must fail loudly --------------------
    print("\n" + "-" * 72)
    print("RUN 3 -- --dry-run on a parameter combination that was never cached")
    print("-" * 72)
    cfg3 = load_config(dry_run=True, **base_kwargs)  # type: ignore[arg-type]
    metrics3, cache3 = MetricsCollector(), DiskCache(args.cache_dir)
    dry_run_failed_loudly = False
    with no_network_allowed():
        with SarvamClient(cfg3, cache=cache3, metrics=metrics3) as client:
            try:
                client.speech_to_text(
                    args.input, mode="verbatim", language_code=args.source_lang,
                    audio_duration_s=duration_s,
                )
            except CacheMissError as exc:
                dry_run_failed_loudly = True
                print(f"CacheMissError raised as required:\n  {exc}\n")
    if not dry_run_failed_loudly:
        print("FAIL: --dry-run did not raise CacheMissError on an uncached key")

    # ---- Report -----------------------------------------------------------------------
    print("=" * 72)
    print("MEASURED RESULTS")
    print("=" * 72)
    header = f"{'run':<16}{'cache':<22}{'latency_s':>11}{'cost_INR':>11}{'net calls':>11}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(f"{row['run']:<16}{row['cache_status']:<22}"
              f"{row['latency_s']:>11.4f}{row['cost_inr']:>11.6f}"
              f"{row['network_calls_this_run']:>11}")
    print("-" * len(header))

    speedup = (results[0]["latency_s"] / results[1]["latency_s"]) if results[1]["latency_s"] else None
    if speedup:
        print(f"cache speedup    : {speedup:,.0f}x faster than the network call")
    print(f"cost of run 2    : Rs {results[1]['cost_inr']:.6f} "
          f"(avoided Rs {results[1]['cost_if_billed_inr']:.6f})")

    checks = {
        # Under --reuse-cache run 1 is legitimately a hit; only demand a cold call when
        # the entry was actually evicted, so a free re-proof cannot report a false FAIL.
        "run1_was_network_call": (
            results[0]["cache_status"].startswith("MISS") if cold_call_expected else True
        ),
        "run2_was_cache_hit": results[1]["cache_status"] == "HIT",
        "run2_zero_network_calls": results[1]["network_calls_this_run"] == 0,
        "run2_cost_zero": results[1]["cost_inr"] == 0.0,
        "payloads_identical": identical,
        "dry_run_fails_loudly_on_miss": dry_run_failed_loudly,
        "transcript_non_empty": bool(payload1.get("transcript", "").strip()),
    }
    print("\nACCEPTANCE CHECKS")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    report = {
        "input": str(args.input),
        "input_duration_s": duration_s,
        "cache_key": key,
        "transcript": payload1.get("transcript"),
        "detected_language_code": payload1.get("language_code"),
        "response_fields_observed": sorted(payload1.keys()),
        "runs": results,
        "checks": checks,
        "run1_metrics": metrics1.to_dict(),
        "run2_metrics": metrics2.to_dict(),
    }
    # A warm re-proof must not overwrite the cold-call record it is meant to corroborate.
    out = Path("output/m1_verification.json" if cold_call_expected
               else "output/m1_verification_warm.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
