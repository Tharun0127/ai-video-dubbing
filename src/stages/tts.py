"""
TTS stage: synthesise each translated segment with Bulbul v3 under closed-loop duration fit.

Input:  output/segments.json from M3, each segment carrying `translation`.
Output: output/audio_segments/seg_NNN.wav (fitted) and seg_NNN_nofit.wav (the pace=1.0
        baseline), plus output/tts_report.json and the per-segment fit records that
        metrics.json aggregates into the before/after drift table.

THE PROBLEM
Hindi carries roughly 1.5x the syllables of the English source (measured in M3), so
synthesising a translation at neutral pace overruns its window and the dub walks out of
sync. The naive fix -- time-stretching the audio afterwards -- is what produces the
chipmunk artefacts of cheap dubbing. Instead this stage generates the speech at the right
tempo in the first place, using Bulbul's native `pace`, and closes the loop by measuring
what actually came back.

THE LOOP
    target = segment.end - segment.start
    attempt 1 is ALWAYS pace=1.0                 <- this is also the unfitted baseline
    ratio  = measured_duration / target
    if 0.95 <= ratio <= 1.05: converged
    else pace <- clamp(pace * ratio, 0.85, 1.25)
    at most 3 attempts; stop once the pace stops moving

Four decisions that make the loop trustworthy rather than plausible:

1. **Attempt 1 IS the baseline.** The before/after comparison reuses it rather than
   re-synthesising at pace=1.0, so both columns cover the same segments and the same text,
   and the comparison costs zero extra credits.

2. **Duration is measured from the returned audio, never reported.** The WAV's own frame
   count divided by its frame rate, via the stdlib `wave` module. A duration the API told
   us would be an assumption dressed as a measurement.

3. **Silence is trimmed before measuring.** Bulbul pads its output, and that padding does
   not scale with pace -- measured at up to 5.7% of a clip (docs/api-notes.md), which is
   larger than the +/-5% convergence band and could therefore decide a segment's verdict
   on its own. Trimming makes the measured duration a clean function of pace. The trimmed
   offsets are kept because M5 needs them to place the segment correctly.

4. **The clamp is 0.85-1.25, not the API's 0.5-2.0.** Beyond roughly +/-25% speech stops
   sounding human. Being deliberately more conservative than the API allows is the point,
   and the measured pace sweep supports it independently: above pace 1.25 the duration
   response nearly saturates (elasticity falls from about -1.5 to -0.2), so the extra
   range buys little anyway.

Example record (measured, samples/test_clip.mp4):
    {"segment_id": 0, "target_duration_s": 5.475, "baseline_duration_s": 1.94,
     "final_duration_s": 2.35, "final_pace": 0.85, "attempts": 2,
     "converged": false, "clamped": true, "residual_drift_pct": -57.1}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..audio import TrimmedAudio, trim_wav_silence, write_wav_bytes
from ..config import FIT_MIN_PACE_STEP, Config
from ..metrics import MetricsCollector, SegmentFit
from ..sarvam_client import SarvamClient

logger = logging.getLogger(__name__)

SEGMENTS_NAME = "segments.json"
REPORT_NAME = "tts_report.json"
AUDIO_DIR_NAME = "audio_segments"

#: Pace values are compared at this resolution; below it a "change" is float noise.
_PACE_EPSILON = 1e-9


class TtsError(RuntimeError):
    """Raised when synthesis or duration fitting cannot proceed safely."""


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Attempt:
    """One synthesis call and what was measured from the audio it returned."""

    index: int
    pace: float
    #: Duration after trimming -- the number the loop actually reasons about.
    duration_s: float
    trimmed: TrimmedAudio
    audio: bytes = field(repr=False)
    request_id: str | None = None
    cache_hit: bool = False

    def ratio(self, target_s: float) -> float:
        """Achieved duration over target duration; 1.0 is a perfect fit."""
        return self.duration_s / target_s if target_s > 0 else 0.0

    def to_dict(self, target_s: float) -> dict[str, Any]:
        """Serialise for the per-segment attempt log."""
        return {
            "attempt": self.index,
            "pace": round(self.pace, 4),
            "duration_s": round(self.duration_s, 6),
            "ratio": round(self.ratio(target_s), 6),
            "drift_pct": round((self.duration_s - target_s) / target_s * 100.0, 4)
            if target_s > 0 else None,
            "request_id": self.request_id,
            "cache_hit": self.cache_hit,
            **{f"audio_{k}": v for k, v in self.trimmed.to_dict().items()},
        }


@dataclass
class FitResult:
    """The outcome of fitting one segment, including every attempt made."""

    segment_id: int
    target_duration_s: float
    attempts: list[Attempt]
    converged: bool
    clamped: bool
    stop_reason: str

    @property
    def baseline(self) -> Attempt:
        """Attempt 1, at pace=1.0 -- the unfitted baseline."""
        return self.attempts[0]

    @property
    def final(self) -> Attempt:
        """The attempt whose audio is shipped: the closest fit achieved."""
        return min(self.attempts, key=lambda a: abs(a.ratio(self.target_duration_s) - 1.0))

    @property
    def residual_drift_pct(self) -> float:
        """Signed drift of the shipped audio; positive means it overruns its window."""
        if self.target_duration_s <= 0:
            return 0.0
        return ((self.final.duration_s - self.target_duration_s)
                / self.target_duration_s * 100.0)

    def to_segment_fit(self) -> SegmentFit:
        """Convert to the metrics record that feeds the aggregate before/after table."""
        return SegmentFit(
            segment_id=self.segment_id,
            target_duration_s=self.target_duration_s,
            achieved_duration_s=self.final.duration_s,
            baseline_duration_s=self.baseline.duration_s,
            final_pace=self.final.pace,
            attempts=len(self.attempts),
            clamped=self.clamped,
            converged=self.converged,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full per-segment record, exactly as the brief asks for it."""
        target = self.target_duration_s
        return {
            "segment_id": self.segment_id,
            "target_duration_s": round(target, 6),
            "baseline_duration_s": round(self.baseline.duration_s, 6),
            "final_duration_s": round(self.final.duration_s, 6),
            "final_pace": round(self.final.pace, 4),
            "attempts": len(self.attempts),
            "converged": self.converged,
            "clamped": self.clamped,
            "residual_drift_pct": round(self.residual_drift_pct, 4),
            "baseline_drift_pct": round(
                (self.baseline.duration_s - target) / target * 100.0, 4
            ) if target > 0 else None,
            "stop_reason": self.stop_reason,
            "attempt_log": [a.to_dict(target) for a in self.attempts],
        }


# --------------------------------------------------------------------------------------
# The fit loop
# --------------------------------------------------------------------------------------

def clamp_pace(proposed: float, *, pace_min: float, pace_max: float) -> tuple[float, bool]:
    """Clamp a proposed pace into the perceptual band; returns (pace, was_clamped)."""
    clamped = min(max(proposed, pace_min), pace_max)
    return clamped, abs(clamped - proposed) > _PACE_EPSILON


def fit_segment(
    text: str,
    target_duration_s: float,
    config: Config,
    client: SarvamClient,
    *,
    segment_id: int,
) -> FitResult:
    """Synthesise one segment, correcting pace until it fits its window or the loop stops."""
    if target_duration_s <= 0:
        raise TtsError(
            f"segment {segment_id} has a non-positive target duration "
            f"({target_duration_s}); M2 guarantees positive windows"
        )
    if not text.strip():
        raise TtsError(f"segment {segment_id} has no text to synthesise")

    max_attempts = 1 if not config.enable_duration_fit else config.fit_max_attempts
    attempts: list[Attempt] = []
    pace = 1.0
    clamped_ever = False
    converged = False
    stop_reason = "max_attempts"

    while True:
        attempts.append(_synthesise(text, pace, config, client, index=len(attempts) + 1))
        current = attempts[-1]
        ratio = current.ratio(target_duration_s)

        logger.info(
            "  segment %d attempt %d: pace=%.4f -> %.3fs vs target %.3fs (ratio %.3f)",
            segment_id, current.index, pace, current.duration_s, target_duration_s, ratio,
        )

        if config.fit_ratio_min <= ratio <= config.fit_ratio_max:
            converged, stop_reason = True, "converged"
            break

        if not config.enable_duration_fit:
            stop_reason = "fit_disabled"
            break

        if len(attempts) >= max_attempts:
            stop_reason = "max_attempts"
            break

        # Higher pace = faster = shorter audio (confirmed by real calls, see
        # docs/api-notes.md), so an overrun (ratio > 1) must RAISE the pace. If that sign
        # were reversed this multiplication would drive every segment away from target.
        proposed = pace * ratio
        next_pace, was_clamped = clamp_pace(
            proposed, pace_min=config.pace_min, pace_max=config.pace_max,
        )
        clamped_ever = clamped_ever or was_clamped

        if abs(next_pace - pace) < FIT_MIN_PACE_STEP:
            # Either the correction is too small to be audible, or the pace is already
            # pinned against the clamp and re-requesting it would return the same audio.
            stop_reason = "pinned_at_clamp" if was_clamped else "pace_step_below_threshold"
            logger.info(
                "  segment %d stopping: proposed pace %.4f -> %.4f, a %.4f step (%s)",
                segment_id, proposed, next_pace, abs(next_pace - pace), stop_reason,
            )
            break

        if was_clamped:
            logger.warning(
                "  segment %d: pace %.3f proposed, clamped to %.3f (perceptual limit "
                "[%.2f, %.2f]); a %.3fs window cannot hold this line naturally",
                segment_id, proposed, next_pace, config.pace_min, config.pace_max,
                target_duration_s,
            )
        pace = next_pace

    result = FitResult(
        segment_id=segment_id,
        target_duration_s=target_duration_s,
        attempts=attempts,
        converged=converged,
        clamped=clamped_ever,
        stop_reason=stop_reason,
    )
    logger.info(
        "  segment %d done: %d attempt(s), pace %.4f, %.3fs vs %.3fs "
        "(drift %+.1f%%, baseline %+.1f%%), converged=%s clamped=%s",
        segment_id, len(attempts), result.final.pace, result.final.duration_s,
        target_duration_s, result.residual_drift_pct,
        (result.baseline.duration_s - target_duration_s) / target_duration_s * 100.0,
        converged, clamped_ever,
    )
    return result


def _synthesise(
    text: str,
    pace: float,
    config: Config,
    client: SarvamClient,
    *,
    index: int,
) -> Attempt:
    """Make one TTS call, trim the result, and measure its duration from the audio itself."""
    before = len(client.metrics.api_calls)
    payload = client.text_to_speech(
        text,
        target_language_code=config.target_lang,
        model=config.tts_model,
        speaker=config.tts_speaker,
        pace=pace,
        speech_sample_rate=config.tts_sample_rate,
    )
    audio = client.audio_bytes(payload)
    trimmed = trim_wav_silence(audio, threshold_db=config.trim_threshold_db)

    if trimmed.all_silent:
        raise TtsError(
            f"synthesis at pace={pace} returned {trimmed.original_duration_s:.3f}s of "
            f"audio with no sample above {config.trim_threshold_db} dBFS. Shipping it "
            f"would put silence where a line should be."
        )

    recorded = client.metrics.api_calls[before:]
    return Attempt(
        index=index,
        pace=pace,
        duration_s=trimmed.duration_s,
        trimmed=trimmed,
        audio=trimmed.data,
        request_id=payload.get("request_id"),
        cache_hit=bool(recorded and recorded[-1].cache_hit),
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_tts(
    config: Config,
    client: SarvamClient,
    metrics: MetricsCollector,
    *,
    segments_path: str | Path | None = None,
) -> dict[str, Any]:
    """Synthesise and fit every translated segment; writes audio, tts_report.json, metrics."""
    path = Path(segments_path) if segments_path else config.output_dir / SEGMENTS_NAME
    if not path.exists():
        raise TtsError(
            f"{path} not found. Run the translate stage first "
            f"(python -m src.pipeline --input <video> --stage translate)."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise TtsError(f"{path} does not contain a non-empty list of segments")

    segments = [s for s in raw if (s.get("translation") or "").strip()]
    if not segments:
        raise TtsError(
            f"no segment in {path} has a translation. Run --stage translate first; "
            f"synthesising the English source would produce the wrong dub silently."
        )
    if len(segments) != len(raw):
        # Only reachable when M3 ran under --max-segments.
        logger.warning(
            "%d of %d segment(s) have no translation and will not be synthesised",
            len(raw) - len(segments), len(raw),
        )

    active = segments
    if config.max_segments is not None and config.max_segments < len(segments):
        active = segments[:config.max_segments]
        logger.warning("--max-segments %d: synthesising %d of %d segment(s)",
                       config.max_segments, len(active), len(segments))

    audio_dir = config.output_dir / AUDIO_DIR_NAME
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "synthesising %d segment(s) with %s/%s at %d Hz, duration fit %s "
        "(band [%.2f, %.2f], pace clamp [%.2f, %.2f], max %d attempts)",
        len(active), config.tts_model, config.tts_speaker, config.tts_sample_rate,
        "ENABLED" if config.enable_duration_fit else "DISABLED (--no-fit)",
        config.fit_ratio_min, config.fit_ratio_max,
        config.pace_min, config.pace_max, config.fit_max_attempts,
    )

    results: list[FitResult] = []
    files: list[dict[str, Any]] = []

    for segment in active:
        segment_id = int(segment["id"])
        target = float(segment["end"]) - float(segment["start"])
        result = fit_segment(
            str(segment["translation"]), target, config, client, segment_id=segment_id,
        )
        results.append(result)
        metrics.record_segment_fit(result.to_segment_fit())

        final_path = audio_dir / f"seg_{segment_id:03d}.wav"
        baseline_path = audio_dir / f"seg_{segment_id:03d}_nofit.wav"
        write_wav_bytes(final_path, result.final.audio)
        write_wav_bytes(baseline_path, result.baseline.audio)

        files.append({
            "segment_id": segment_id,
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": segment.get("text"),
            "translation": segment["translation"],
            "fitted_path": str(final_path),
            "baseline_path": str(baseline_path),
            "identical_to_baseline": result.final.index == 1,
            # M5 must add the lead trim back when placing this clip, or the segment
            # starts earlier than the speaker did.
            "lead_trim_s": round(result.final.trimmed.lead_trim_s, 6),
            "trail_trim_s": round(result.final.trimmed.trail_trim_s, 6),
        })

    metrics.duration_fit_enabled = config.enable_duration_fit
    fit_stats = metrics.duration_fit_stats()

    report = {
        "segments_path": str(path),
        "audio_dir": str(audio_dir),
        "tts": {
            "model": config.tts_model,
            "speaker": config.tts_speaker,
            "target_language_code": config.target_lang,
            "speech_sample_rate": config.tts_sample_rate,
            "output_audio_codec": "wav",
        },
        "fit": {
            "enabled": config.enable_duration_fit,
            "ratio_band": [config.fit_ratio_min, config.fit_ratio_max],
            "pace_clamp": [config.pace_min, config.pace_max],
            "api_pace_range": [0.5, 2.0],
            "max_attempts": config.fit_max_attempts,
            "trim_threshold_db": config.trim_threshold_db,
            "duration_source": "measured from returned WAV frames, after silence trim",
        },
        "duration_fit": fit_stats,
        "per_segment": [r.to_dict() for r in results],
        "files": files,
    }

    report_path = config.output_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _log_drift_table(results, fit_stats)
    logger.info("wrote %d fitted + %d baseline clip(s) to %s, and %s",
                len(files), len(files), audio_dir, report_path)
    return report


def _log_drift_table(results: Sequence[FitResult], stats: dict[str, Any]) -> None:
    """Print the before/after drift table -- the headline result of this milestone."""
    logger.info("duration fit, paired before/after over the same %d segment(s):", len(results))
    logger.info("  %3s %9s %11s %10s %11s %10s %6s %6s %6s",
                "id", "target", "no-fit", "drift%", "fitted", "drift%",
                "pace", "tries", "conv")
    for result in results:
        target = result.target_duration_s
        base_drift = (result.baseline.duration_s - target) / target * 100.0
        logger.info(
            "  %3d %9.3f %11.3f %+10.1f %11.3f %+10.1f %6.3f %6d %6s%s",
            result.segment_id, target, result.baseline.duration_s, base_drift,
            result.final.duration_s, result.residual_drift_pct, result.final.pace,
            len(result.attempts), "yes" if result.converged else "no",
            "  CLAMPED" if result.clamped else "",
        )

    before, after = stats["without_fit"], stats["with_fit"]
    logger.info("  %-14s %10s %10s %10s %10s", "", "mean", "p50", "p95", "max")
    logger.info("  %-14s %10.2f %10.2f %10.2f %10.2f", "abs drift, no-fit",
                before["mean_abs_drift_pct"], before["p50_abs_drift_pct"],
                before["p95_abs_drift_pct"], before["max_abs_drift_pct"])
    logger.info("  %-14s %10.2f %10.2f %10.2f %10.2f", "abs drift, fit",
                after["mean_abs_drift_pct"], after["p50_abs_drift_pct"],
                after["p95_abs_drift_pct"], after["max_abs_drift_pct"])
    logger.info(
        "  n=%d%s | converged %d/%d | clamped %d/%d (%.0f%%) | mean attempts %.2f | "
        "mean final pace %.3f",
        stats["n_segments"], " (SMALL SAMPLE)" if stats.get("small_sample") else "",
        stats["converged_segments"], stats["n_segments"],
        stats["clamped_segments"], stats["n_segments"], stats["clamped_pct"],
        stats["mean_attempts"], stats["mean_final_pace"],
    )
