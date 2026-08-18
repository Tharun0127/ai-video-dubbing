"""
Metrics stage: measured-only instrumentation for the dubbing pipeline.

Every number this project reports comes from here, and every number here comes
from an actual run: stage wall-clock latencies, per-call API cost derived from
Sarvam's published price list, cache hit/miss counts, and duration-fit drift
statistics. Fields with no observations serialise as the string "not yet
measured" rather than 0, so an unmeasured value can never masquerade as a result.

Input:  timing events, ApiCall records, and per-segment fit results.
Output: output/metrics.json in the schema defined by SPEC.md.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generator

from contextlib import contextmanager

from .config import (
    PRICE_STT_INR_PER_HOUR,
    PRICE_TRANSLATE_INR_PER_10K_CHARS,
    PRICE_TTS_BULBUL_V3_INR_PER_10K_CHARS,
    STAGE_ORDER,
)

logger = logging.getLogger(__name__)

#: Sentinel written into metrics.json wherever nothing has actually been measured.
NOT_MEASURED = "not yet measured"


# --------------------------------------------------------------------------------------
# Cost model -- prices from docs.sarvam.ai/api/getting-started/pricing (2026-07-28)
# --------------------------------------------------------------------------------------

def cost_stt_inr(audio_seconds: float, diarized: bool = False) -> float:
    """Cost of one speech-to-text call: charged per second of audio, rounded up per request."""
    from .config import PRICE_STT_DIARIZED_INR_PER_HOUR

    rate = PRICE_STT_DIARIZED_INR_PER_HOUR if diarized else PRICE_STT_INR_PER_HOUR
    billable_seconds = math.ceil(max(0.0, audio_seconds))
    return billable_seconds * rate / 3600.0


def cost_translate_inr(characters: int) -> float:
    """Cost of one translation call: charged per character, rounded up per request."""
    return math.ceil(max(0, characters)) * PRICE_TRANSLATE_INR_PER_10K_CHARS / 10_000.0


def cost_tts_inr(characters: int) -> float:
    """Cost of one Bulbul v3 TTS call: charged per character, rounded up per request."""
    return math.ceil(max(0, characters)) * PRICE_TTS_BULBUL_V3_INR_PER_10K_CHARS / 10_000.0


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------

@dataclass
class ApiCall:
    """One attempt to obtain an API result, whether it hit the network or the cache."""

    stage: str
    endpoint: str
    model: str
    cache_hit: bool
    latency_s: float
    #: Rupees actually charged: always 0.0 for a cache hit.
    cost_inr: float
    #: Rupees this call would have cost at list price; lets us report cache savings.
    cost_if_billed_inr: float
    #: What was billed: seconds of audio for STT, characters for translate/TTS.
    billable_units: float
    billable_unit_name: str
    http_status: int | None = None
    attempts: int = 1
    request_id: str | None = None


@dataclass
class SegmentFit:
    """Duration-fit outcome for a single TTS segment.

    Carries both the fitted result and the pace=1.0 baseline, because the headline number
    this project produces is the *paired* before/after drift over the same segments. The
    baseline is attempt 1 of the fit loop itself, never a second synthesis run, so the
    comparison costs nothing extra and is measured on identical text.
    """

    segment_id: int
    target_duration_s: float
    achieved_duration_s: float
    #: Duration measured at pace=1.0, i.e. attempt 1. This is the unfitted baseline.
    baseline_duration_s: float
    final_pace: float
    attempts: int
    clamped: bool
    converged: bool

    @property
    def abs_drift_pct(self) -> float:
        """Absolute drift of achieved vs target duration, as a percentage of target."""
        return _abs_drift_pct(self.achieved_duration_s, self.target_duration_s)

    @property
    def baseline_abs_drift_pct(self) -> float:
        """Absolute drift the unfitted pace=1.0 audio would have had, same segment."""
        return _abs_drift_pct(self.baseline_duration_s, self.target_duration_s)

    @property
    def signed_drift_pct(self) -> float:
        """Signed drift: positive means the dub overruns its window."""
        if self.target_duration_s <= 0:
            return 0.0
        return (self.achieved_duration_s - self.target_duration_s) / self.target_duration_s * 100.0

    @property
    def overrun_pct(self) -> float:
        """Drift in the only direction that breaks sync: how far the dub overruns.

        An underrun is absorbed by padding the gap with silence (SPEC.md), so it costs
        nothing in sync terms. An overrun has to spill into the next segment's window or
        be accepted as drift. Reporting both this and the symmetric absolute drift keeps
        the distinction visible instead of averaging the two failure modes together.
        """
        if self.target_duration_s <= 0:
            return 0.0
        return max(0.0, self.achieved_duration_s - self.target_duration_s) / self.target_duration_s * 100.0

    @property
    def baseline_overrun_pct(self) -> float:
        """Overrun the unfitted pace=1.0 audio would have had, same segment."""
        if self.target_duration_s <= 0:
            return 0.0
        return max(0.0, self.baseline_duration_s - self.target_duration_s) / self.target_duration_s * 100.0


def _abs_drift_pct(achieved: float, target: float) -> float:
    """Absolute difference between achieved and target duration, as a percentage of target."""
    if target <= 0:
        return 0.0
    return abs(achieved - target) / target * 100.0


# --------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------

def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of a non-empty list; pct is 0-100."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


def _drift_block(drifts: list[float]) -> dict[str, Any]:
    """Summarise one column of absolute drift percentages."""
    return {
        "mean_abs_drift_pct": round(_mean(drifts), 4),
        "p50_abs_drift_pct": round(percentile(drifts, 50), 4),
        "p95_abs_drift_pct": round(percentile(drifts, 95), 4),
        "max_abs_drift_pct": round(max(drifts), 4),
        "segments_over_5pct": sum(1 for d in drifts if d > 5.0),
    }


# --------------------------------------------------------------------------------------
# Collector
# --------------------------------------------------------------------------------------

class MetricsCollector:
    """Accumulates every measurement taken during one pipeline run."""

    def __init__(self) -> None:
        """Start the wall clock and initialise empty measurement stores."""
        self._run_started = time.perf_counter()
        self.stage_latency_s: dict[str, float] = {}
        self.api_calls: list[ApiCall] = []
        self.segment_fits: list[SegmentFit] = []
        self.input_info: dict[str, Any] = {}
        self.segment_count: int | None = None
        self.segment_mean_duration_s: float | None = None
        self.duration_fit_enabled: bool | None = None
        self.qc: dict[str, Any] = {}

    # --- timing -----------------------------------------------------------------------

    @contextmanager
    def time_stage(self, stage: str) -> Generator[None]:
        """Context manager recording the wall-clock duration of one pipeline stage."""
        started = time.perf_counter()
        logger.info("stage %-9s START", stage)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.stage_latency_s[stage] = self.stage_latency_s.get(stage, 0.0) + elapsed
            logger.info("stage %-9s DONE in %.3fs", stage, elapsed)

    @property
    def total_wall_clock_s(self) -> float:
        """Seconds elapsed since this collector was constructed."""
        return time.perf_counter() - self._run_started

    # --- recording --------------------------------------------------------------------

    def record_api_call(self, call: ApiCall) -> None:
        """Record one API call (network or cache) for cost and latency accounting."""
        self.api_calls.append(call)
        logger.debug(
            "api %s %s cache_hit=%s latency=%.3fs cost=Rs%.4f",
            call.stage, call.endpoint, call.cache_hit, call.latency_s, call.cost_inr,
        )

    def record_segment_fit(self, fit: SegmentFit) -> None:
        """Record the duration-fit outcome of one TTS segment."""
        self.segment_fits.append(fit)

    def set_input_info(self, path: str, duration_s: float | None, source_lang: str, target_lang: str) -> None:
        """Record what was fed into this run."""
        self.input_info = {
            "path": path,
            "duration_s": duration_s if duration_s is not None else NOT_MEASURED,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

    # --- derived views ----------------------------------------------------------------

    @property
    def network_calls(self) -> list[ApiCall]:
        """API calls that actually hit the network this run."""
        return [c for c in self.api_calls if not c.cache_hit]

    @property
    def cache_hits(self) -> int:
        """Number of API results served from disk cache this run."""
        return sum(1 for c in self.api_calls if c.cache_hit)

    @property
    def cache_misses(self) -> int:
        """Number of API results that required a network call this run."""
        return len(self.network_calls)

    @property
    def total_cost_inr(self) -> float:
        """Rupees actually spent this run; cache hits contribute zero."""
        return sum(c.cost_inr for c in self.api_calls if not c.cache_hit)

    @property
    def cost_avoided_inr(self) -> float:
        """Rupees the cache saved this run, at list price."""
        return sum(c.cost_if_billed_inr for c in self.api_calls if c.cache_hit)

    def calls_by_stage(self) -> dict[str, int]:
        """Count of network calls per stage."""
        counts: dict[str, int] = {}
        for call in self.network_calls:
            counts[call.stage] = counts.get(call.stage, 0) + 1
        return counts

    def duration_fit_stats(self) -> dict[str, Any]:
        """Paired before/after drift statistics over recorded segment fits.

        "Before" is each segment's own pace=1.0 attempt and "after" is its fitted result,
        so the two columns cover exactly the same segments and the same text -- an
        unpaired comparison across different runs would confound the fit loop with
        whatever else changed.
        """
        enabled = self.duration_fit_enabled
        empty = {
            "mean_abs_drift_pct": NOT_MEASURED,
            "p50_abs_drift_pct": NOT_MEASURED,
            "p95_abs_drift_pct": NOT_MEASURED,
            "max_abs_drift_pct": NOT_MEASURED,
            "segments_over_5pct": NOT_MEASURED,
        }
        if not self.segment_fits:
            return {
                "enabled": enabled if enabled is not None else NOT_MEASURED,
                "n_segments": NOT_MEASURED,
                "without_fit": dict(empty),
                "with_fit": dict(empty),
                "improvement": NOT_MEASURED,
                **empty,
                "mean_attempts": NOT_MEASURED,
                "mean_final_pace": NOT_MEASURED,
                "clamped_segments": NOT_MEASURED,
                "clamped_pct": NOT_MEASURED,
                "converged_segments": NOT_MEASURED,
            }

        fits = self.segment_fits
        count = len(fits)
        before = [f.baseline_abs_drift_pct for f in fits]
        after = [f.abs_drift_pct for f in fits]

        stats = {
            "enabled": enabled if enabled is not None else NOT_MEASURED,
            "n_segments": count,
            # A four-segment sample cannot support a meaningful p95; it is reported
            # because SPEC.md asks for it, and flagged so nobody quotes it as robust.
            "small_sample": count < 20,
            "without_fit": _drift_block(before),
            "with_fit": _drift_block(after),
            "overrun_only": {
                "note": (
                    "Underruns are absorbed by silence padding in M5 and cost nothing in "
                    "sync terms; only overruns have to spill or drift. These are the same "
                    "segments, counting overrun alone."
                ),
                "without_fit_mean_pct": round(
                    _mean([f.baseline_overrun_pct for f in fits]), 4),
                "with_fit_mean_pct": round(_mean([f.overrun_pct for f in fits]), 4),
                "without_fit_segments_overrunning": sum(
                    1 for f in fits if f.baseline_overrun_pct > 0),
                "with_fit_segments_overrunning": sum(1 for f in fits if f.overrun_pct > 0),
            },
            "direction": {
                "segments_underrunning": sum(1 for f in fits if f.signed_drift_pct < 0),
                "segments_overrunning": sum(1 for f in fits if f.signed_drift_pct > 0),
                "mean_signed_drift_pct": round(
                    _mean([f.signed_drift_pct for f in fits]), 4),
            },
            "improvement": {
                "mean_abs_drift_pct_delta": round(_mean(after) - _mean(before), 4),
                "mean_abs_drift_pct_reduction": (
                    round((1 - _mean(after) / _mean(before)) * 100.0, 2)
                    if _mean(before) > 0 else NOT_MEASURED
                ),
                "segments_improved": sum(1 for f in fits
                                         if f.abs_drift_pct < f.baseline_abs_drift_pct),
                "segments_worsened": sum(1 for f in fits
                                         if f.abs_drift_pct > f.baseline_abs_drift_pct),
            },
            "mean_attempts": round(_mean([float(f.attempts) for f in fits]), 4),
            "mean_final_pace": round(_mean([f.final_pace for f in fits]), 4),
            "clamped_segments": sum(1 for f in fits if f.clamped),
            "clamped_pct": round(sum(1 for f in fits if f.clamped) / count * 100.0, 2),
            "converged_segments": sum(1 for f in fits if f.converged),
            "converged_pct": round(sum(1 for f in fits if f.converged) / count * 100.0, 2),
        }
        # SPEC.md's flat field names, kept so the documented metrics.json schema still
        # resolves; they always describe the WITH-fit column.
        stats.update(_drift_block(after))
        return stats

    # --- output -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Build the metrics.json document in the schema defined by SPEC.md."""
        input_duration = self.input_info.get("duration_s")
        total_wall = self.total_wall_clock_s

        realtime_factor: float | str = NOT_MEASURED
        if isinstance(input_duration, (int, float)) and input_duration > 0:
            realtime_factor = round(total_wall / input_duration, 4)

        latency: dict[str, Any] = {
            stage: round(self.stage_latency_s[stage], 4) if stage in self.stage_latency_s else NOT_MEASURED
            for stage in STAGE_ORDER
        }
        latency["total_wall_clock"] = round(total_wall, 4)
        latency["realtime_factor"] = realtime_factor

        stage_calls = self.calls_by_stage()

        return {
            "input": self.input_info or NOT_MEASURED,
            "segments": {
                "count": self.segment_count if self.segment_count is not None else NOT_MEASURED,
                "mean_duration_s": (
                    round(self.segment_mean_duration_s, 4)
                    if self.segment_mean_duration_s is not None
                    else NOT_MEASURED
                ),
            },
            "latency_s": latency,
            "api": {
                "calls": {
                    "asr": stage_calls.get("asr", 0),
                    "translate": stage_calls.get("translate", 0),
                    "tts": stage_calls.get("tts", 0),
                    # M6 back-transcription also hits /speech-to-text, but it is scoring the
                    # dub rather than producing it, so it is counted separately instead of
                    # being folded into the ASR total.
                    "qc": stage_calls.get("qc", 0),
                },
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "estimated_cost_inr": round(self.total_cost_inr, 6),
                "cost_avoided_by_cache_inr": round(self.cost_avoided_inr, 6),
            },
            "duration_fit": self.duration_fit_stats(),
            "qc": self.qc or {
                "wer": NOT_MEASURED, "cer": NOT_MEASURED, "flagged_segments": NOT_MEASURED,
            },
            "api_call_log": [asdict(c) for c in self.api_calls],
        }

    def write(self, path: str | Path) -> Path:
        """Write metrics.json to disk and return the path written."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote metrics to %s", out)
        return out

    def log_summary(self) -> None:
        """Emit the mandatory end-of-run summary: cache hits/misses, cost, wall clock."""
        hits, misses = self.cache_hits, self.cache_misses
        total = hits + misses
        rate = f"{hits / total * 100:.0f}%" if total else "n/a"
        logger.info("=" * 68)
        logger.info("RUN SUMMARY")
        logger.info("  api results     : %d (%d cache hits, %d network calls, hit rate %s)",
                    total, hits, misses, rate)
        for stage, elapsed in self.stage_latency_s.items():
            logger.info("  stage %-11s: %.3fs", stage, elapsed)
        if self.network_calls:
            net_latency = _mean([c.latency_s for c in self.network_calls])
            logger.info("  mean net latency: %.3fs over %d call(s)", net_latency, len(self.network_calls))
        logger.info("  cost this run   : Rs %.4f", self.total_cost_inr)
        logger.info("  cost avoided    : Rs %.4f (served from cache)", self.cost_avoided_inr)
        logger.info("  wall clock      : %.3fs", self.total_wall_clock_s)
        logger.info("=" * 68)
