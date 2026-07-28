"""Tests for cost accounting, drift statistics, and the measured-only guarantee.

The cost formulas encode Sarvam's published prices; if these drift from the
pricing page every rupee this project reports becomes wrong. The NOT_MEASURED
guarantee is equally load-bearing: an unmeasured field must never serialise as 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.metrics import (
    NOT_MEASURED,
    ApiCall,
    MetricsCollector,
    SegmentFit,
    cost_stt_inr,
    cost_translate_inr,
    cost_tts_inr,
    percentile,
)


def make_call(**overrides: object) -> ApiCall:
    """Build an ApiCall with sensible defaults, overridable per test."""
    defaults = dict(
        stage="asr", endpoint="/speech-to-text", model="saaras:v3", cache_hit=False,
        latency_s=0.5, cost_inr=0.083333, cost_if_billed_inr=0.083333,
        billable_units=10.0, billable_unit_name="audio_seconds", http_status=200, attempts=1,
    )
    defaults.update(overrides)
    return ApiCall(**defaults)  # type: ignore[arg-type]


# --- cost model -----------------------------------------------------------------------

def test_stt_cost_matches_published_rate() -> None:
    """One hour of audio costs exactly Rs30 at the documented rate."""
    assert cost_stt_inr(3600) == pytest.approx(30.0)


def test_stt_cost_for_the_measured_10s_clip() -> None:
    """The 10s M1 clip costs Rs0.083333, the number reported in the milestone."""
    assert cost_stt_inr(10.0) == pytest.approx(30.0 * 10 / 3600, rel=1e-9)


def test_stt_cost_rounds_seconds_up_per_request() -> None:
    """Docs: 'rounded up to the nearest second in each request'."""
    assert cost_stt_inr(9.1) == cost_stt_inr(10.0)
    assert cost_stt_inr(9.0) < cost_stt_inr(9.1)


def test_stt_diarized_rate_is_higher() -> None:
    """Diarization is billed at Rs45/hour rather than Rs30/hour."""
    assert cost_stt_inr(3600, diarized=True) == pytest.approx(45.0)


def test_translate_and_tts_costs_match_published_rates() -> None:
    """Translate is Rs20/10K chars; Bulbul v3 is Rs30/10K chars."""
    assert cost_translate_inr(10_000) == pytest.approx(20.0)
    assert cost_tts_inr(10_000) == pytest.approx(30.0)


def test_costs_are_never_negative() -> None:
    """Defensive: a bad duration or character count cannot produce a negative charge."""
    assert cost_stt_inr(-5) == 0.0
    assert cost_translate_inr(-5) == 0.0


# --- statistics -----------------------------------------------------------------------

def test_percentile_interpolates() -> None:
    """p50 of an even-length list is the midpoint between the two central values."""
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


def test_percentile_rejects_empty_input() -> None:
    """An empty list has no percentile; it must raise rather than return 0."""
    with pytest.raises(ValueError):
        percentile([], 50)


def test_segment_fit_drift_percentage() -> None:
    """Drift is |achieved - target| as a percentage of target."""
    fit = SegmentFit(segment_id=1, target_duration_s=2.0, achieved_duration_s=2.2,
                     final_pace=1.1, attempts=2, clamped=False)
    assert fit.abs_drift_pct == pytest.approx(10.0)


# --- collector ------------------------------------------------------------------------

def test_cache_hits_cost_nothing_and_are_counted_separately(metrics: MetricsCollector) -> None:
    """A cache hit contributes zero cost and increments the hit counter, not the miss counter."""
    metrics.record_api_call(make_call(cache_hit=False))
    metrics.record_api_call(make_call(cache_hit=True, cost_inr=0.0, latency_s=0.01, attempts=0))

    assert metrics.cache_hits == 1
    assert metrics.cache_misses == 1
    assert metrics.total_cost_inr == pytest.approx(0.083333)
    assert metrics.cost_avoided_inr == pytest.approx(0.083333)


def test_unmeasured_fields_serialise_as_not_measured(metrics: MetricsCollector) -> None:
    """Nothing measured means nothing reported as 0 -- the core anti-fabrication rule."""
    doc = metrics.to_dict()
    assert doc["segments"]["count"] == NOT_MEASURED
    assert doc["duration_fit"]["mean_abs_drift_pct"] == NOT_MEASURED
    assert doc["qc"]["wer"] == NOT_MEASURED
    assert doc["latency_s"]["demux"] == NOT_MEASURED
    assert doc["latency_s"]["realtime_factor"] == NOT_MEASURED


def test_stage_timer_records_latency(metrics: MetricsCollector) -> None:
    """time_stage records a positive wall-clock duration under the stage name."""
    with metrics.time_stage("asr"):
        pass
    assert metrics.stage_latency_s["asr"] > 0
    assert metrics.to_dict()["latency_s"]["asr"] >= 0


def test_stage_timer_records_even_when_the_stage_raises(metrics: MetricsCollector) -> None:
    """A failing stage still contributes its latency, so crashes remain measurable."""
    with pytest.raises(RuntimeError):
        with metrics.time_stage("asr"):
            raise RuntimeError("boom")
    assert "asr" in metrics.stage_latency_s


def test_realtime_factor_is_computed_from_measured_duration(metrics: MetricsCollector) -> None:
    """realtime_factor appears only once the input duration has actually been measured."""
    metrics.set_input_info("clip.wav", duration_s=10.0, source_lang="en-IN", target_lang="hi-IN")
    doc = metrics.to_dict()
    assert isinstance(doc["latency_s"]["realtime_factor"], float)


def test_duration_fit_stats_from_recorded_segments(metrics: MetricsCollector) -> None:
    """Drift statistics are derived from recorded fits, including the >5% count."""
    metrics.duration_fit_enabled = True
    metrics.record_segment_fit(SegmentFit(1, 2.0, 2.0, 1.0, 1, False))   # 0% drift
    metrics.record_segment_fit(SegmentFit(2, 2.0, 2.2, 1.1, 2, False))   # 10% drift
    metrics.record_segment_fit(SegmentFit(3, 2.0, 2.02, 1.0, 1, True))   # 1% drift

    stats = metrics.duration_fit_stats()
    assert stats["enabled"] is True
    assert stats["segments_over_5pct"] == 1
    assert stats["clamped_segments"] == 1
    # duration_fit_stats rounds to 4 decimal places for readable metrics.json output.
    assert stats["mean_abs_drift_pct"] == pytest.approx((0 + 10 + 1) / 3, abs=1e-4)


def test_calls_by_stage_counts_only_network_calls(metrics: MetricsCollector) -> None:
    """Cache hits are not API calls and must not inflate the per-stage call counts."""
    metrics.record_api_call(make_call(stage="asr", cache_hit=False))
    metrics.record_api_call(make_call(stage="asr", cache_hit=True, cost_inr=0.0))
    metrics.record_api_call(make_call(stage="tts", cache_hit=False))

    assert metrics.calls_by_stage() == {"asr": 1, "tts": 1}
    assert metrics.to_dict()["api"]["calls"] == {"asr": 1, "translate": 0, "tts": 1}


def test_write_produces_valid_json(metrics: MetricsCollector, tmp_path: Path) -> None:
    """metrics.json is written as valid, re-readable JSON."""
    metrics.record_api_call(make_call())
    out = metrics.write(tmp_path / "metrics.json")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["api"]["cache_misses"] == 1
    assert doc["api_call_log"][0]["endpoint"] == "/speech-to-text"
