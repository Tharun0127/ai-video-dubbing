"""Tests for the TTS stage: the duration-fit loop, clamping, trimming, and measurement.

No test opens a socket. The API is replaced by a synthetic synthesiser whose duration
response to `pace` is controlled by the test, which is what makes the loop's convergence,
clamping, and non-convergence paths reachable deterministically -- the real API cannot be
asked to overrun by exactly 40% on demand.

The synthetic model is duration = base / pace, i.e. exactly proportional. That is the
*ideal* the update rule assumes; the real API's measured elasticity is steeper than that
(docs/api-notes.md), which is recorded there rather than simulated here, because a test
that encodes today's measured non-linearity would fail the day Sarvam retunes the model.
"""

from __future__ import annotations

import io
import json
import math
import wave
from pathlib import Path
from typing import Any

import pytest

from src.audio import trim_wav_silence, wav_bytes_duration_s
from src.config import Config
from src.metrics import MetricsCollector, SegmentFit
from src.sarvam_client import SarvamAPIError, SarvamClient
from src.stages.tts import (
    Attempt,
    FitResult,
    TtsError,
    clamp_pace,
    fit_segment,
    run_tts,
)

from tests.test_sarvam_client import FakeResponse, build_client

SAMPLE_RATE = 24000


# --- synthetic audio --------------------------------------------------------------------

def make_wav(
    duration_s: float,
    *,
    lead_silence_s: float = 0.0,
    trail_silence_s: float = 0.0,
    amplitude: int = 12000,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Build a mono 16-bit WAV: optional silence, a loud tone, optional silence."""
    def frames(count: int, level: int) -> bytes:
        return b"".join(
            int(level * math.sin(2 * math.pi * 220 * n / sample_rate)).to_bytes(
                2, "little", signed=True
            )
            for n in range(count)
        )

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames(int(lead_silence_s * sample_rate), 0))
        writer.writeframes(frames(int(duration_s * sample_rate), amplitude))
        writer.writeframes(frames(int(trail_silence_s * sample_rate), 0))
    return buffer.getvalue()


class SynthClient:
    """A SarvamClient stand-in whose audio duration is a controlled function of pace."""

    def __init__(
        self,
        base_duration_s: float,
        *,
        elasticity: float = 1.0,
        lead_silence_s: float = 0.0,
        trail_silence_s: float = 0.0,
    ) -> None:
        """Model duration = base / pace**elasticity, with optional fixed silence padding."""
        self.base_duration_s = base_duration_s
        self.elasticity = elasticity
        self.lead_silence_s = lead_silence_s
        self.trail_silence_s = trail_silence_s
        self.paces: list[float] = []
        self.metrics = MetricsCollector()

    def text_to_speech(self, text: str, *, pace: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Return a synthetic WAV whose speech length depends on pace."""
        self.paces.append(round(pace, 4))
        duration = self.base_duration_s / (pace ** self.elasticity)
        return {
            "request_id": f"synth-{len(self.paces)}",
            "audios": [make_wav(
                duration,
                lead_silence_s=self.lead_silence_s,
                trail_silence_s=self.trail_silence_s,
            )],
        }

    @staticmethod
    def audio_bytes(payload: dict[str, Any]) -> bytes:
        """Return the WAV directly; the synthetic client skips base64."""
        return payload["audios"][0]


# --- clamping ----------------------------------------------------------------------------

@pytest.mark.parametrize("proposed,expected,clamped", [
    (1.0, 1.0, False),
    (1.10, 1.10, False),
    (1.40, 1.25, True),      # would overrun the perceptual limit
    (0.40, 0.85, True),
    (1.25, 1.25, False),     # exactly on the boundary is not a clamp
    (0.85, 0.85, False),
    (99.0, 1.25, True),
])
def test_clamp_pace(proposed: float, expected: float, clamped: bool) -> None:
    """The clamp is the perceptual limit, tighter than the API's 0.5-2.0 on purpose."""
    pace, was_clamped = clamp_pace(proposed, pace_min=0.85, pace_max=1.25)
    assert pace == pytest.approx(expected)
    assert was_clamped is clamped


def test_clamp_is_tighter_than_the_api_range(config: Config) -> None:
    """SPEC.md's central engineering call: never request the full API range."""
    assert config.pace_min > 0.5
    assert config.pace_max < 2.0


# --- silence trimming --------------------------------------------------------------------

def test_trim_removes_padding_and_reports_what_it_removed() -> None:
    """Trimming is what makes the measured duration a clean function of pace."""
    audio = make_wav(1.0, lead_silence_s=0.3, trail_silence_s=0.5)
    assert wav_bytes_duration_s(audio) == pytest.approx(1.8, abs=0.01)

    trimmed = trim_wav_silence(audio, threshold_db=-40.0)
    # The 20 ms keep-margin is deliberately given back at each end.
    assert trimmed.duration_s == pytest.approx(1.04, abs=0.02)
    assert trimmed.lead_trim_s == pytest.approx(0.28, abs=0.02)
    assert trimmed.trail_trim_s == pytest.approx(0.48, abs=0.02)
    assert trimmed.original_duration_s == pytest.approx(1.8, abs=0.01)
    assert not trimmed.all_silent


def test_trim_keeps_offsets_m5_needs() -> None:
    """M5 must add the lead trim back, or the segment starts earlier than the speaker did."""
    trimmed = trim_wav_silence(make_wav(1.0, lead_silence_s=0.4), threshold_db=-40.0)
    assert trimmed.lead_trim_s > 0.3
    assert trimmed.to_dict()["lead_trim_s"] == pytest.approx(trimmed.lead_trim_s)


def test_all_silent_audio_is_flagged_not_deleted() -> None:
    """Trimming silence to nothing would leave a hole in the timeline; refuse instead."""
    trimmed = trim_wav_silence(make_wav(1.0, amplitude=0), threshold_db=-40.0)
    assert trimmed.all_silent
    assert trimmed.duration_s == pytest.approx(1.0, abs=0.01)


def test_duration_is_measured_from_frames_not_from_a_reported_field() -> None:
    """A 2.5s WAV measures 2.5s regardless of what any API field might claim."""
    assert wav_bytes_duration_s(make_wav(2.5)) == pytest.approx(2.5, abs=0.01)


def test_unreadable_audio_fails_loudly() -> None:
    """Garbage bytes must not be silently treated as zero-length audio."""
    from src.audio import AudioError

    with pytest.raises(AudioError, match="not a readable PCM WAV"):
        wav_bytes_duration_s(b"not a wav at all")


# --- the fit loop ------------------------------------------------------------------------

def test_attempt_one_is_always_pace_1_and_is_the_baseline(config: Config) -> None:
    """The before/after comparison reuses attempt 1 rather than re-synthesising."""
    client = SynthClient(base_duration_s=2.0)
    result = fit_segment("नमस्ते", 2.0, config, client, segment_id=0)  # type: ignore[arg-type]

    assert client.paces[0] == 1.0
    assert result.baseline.pace == 1.0
    assert result.baseline.duration_s == pytest.approx(2.0, abs=0.01)


def test_a_segment_already_in_band_converges_in_one_attempt(config: Config) -> None:
    """No correction is worth a second credit when the first attempt already fits."""
    client = SynthClient(base_duration_s=2.0)
    result = fit_segment("नमस्ते", 2.0, config, client, segment_id=0)  # type: ignore[arg-type]

    assert len(result.attempts) == 1
    assert result.converged is True
    assert result.clamped is False
    assert result.stop_reason == "converged"
    assert client.paces == [1.0]


@pytest.mark.parametrize("ratio", [1.04, 0.96])
def test_the_convergence_band_is_inclusive(config: Config, ratio: float) -> None:
    """Drift inside +/-5% is accepted; chasing it further just spends credits."""
    client = SynthClient(base_duration_s=2.0 * ratio)
    result = fit_segment("नमस्ते", 2.0, config, client, segment_id=0)  # type: ignore[arg-type]
    assert result.converged is True
    assert len(result.attempts) == 1


def test_an_overrunning_segment_is_sped_up_and_converges(config: Config) -> None:
    """The core case: a 20% overrun is corrected by RAISING pace, because higher = faster."""
    client = SynthClient(base_duration_s=2.4)
    result = fit_segment("नमस्ते", 2.0, config, client, segment_id=0)  # type: ignore[arg-type]

    assert client.paces[0] == 1.0
    # ratio 1.2 -> pace 1.2, inside the clamp, and duration = 2.4/1.2 = 2.0 exactly.
    assert client.paces[1] == pytest.approx(1.2, abs=0.001)
    assert client.paces[1] > client.paces[0], "an overrun must increase pace, not lower it"
    assert result.converged is True
    assert result.clamped is False
    assert len(result.attempts) == 2
    assert result.final.duration_s == pytest.approx(2.0, abs=0.01)


def test_an_underrunning_segment_is_slowed_down(config: Config) -> None:
    """The mirror case, so a sign error in the update rule cannot pass unnoticed."""
    client = SynthClient(base_duration_s=1.8)
    result = fit_segment("नमस्ते", 2.0, config, client, segment_id=0)  # type: ignore[arg-type]

    assert client.paces[1] < 1.0, "an underrun must lower pace, not raise it"
    assert result.converged is True


def test_fitting_improves_on_the_baseline(config: Config) -> None:
    """The whole point: the shipped audio is closer to target than attempt 1 was."""
    client = SynthClient(base_duration_s=2.4)
    result = fit_segment("नमस्ते", 2.0, config, client, segment_id=0)  # type: ignore[arg-type]

    fit = result.to_segment_fit()
    assert fit.abs_drift_pct < fit.baseline_abs_drift_pct
    assert fit.baseline_duration_s == pytest.approx(2.4, abs=0.01)


# --- the clamp path ----------------------------------------------------------------------

def test_an_impossible_overrun_clamps_and_stops(config: Config) -> None:
    """A segment needing 2x speed-up is held at 1.25 and recorded as clamped, not spun on."""
    client = SynthClient(base_duration_s=4.0)  # target 2.0 -> ratio 2.0
    result = fit_segment("नमस्ते", 2.0, config, client, segment_id=7)  # type: ignore[arg-type]

    assert result.clamped is True
    assert result.converged is False
    assert max(client.paces) == pytest.approx(config.pace_max)
    assert all(p <= config.pace_max for p in client.paces), "never exceed the perceptual clamp"
    assert result.stop_reason == "pinned_at_clamp"
    # Attempt 1 at pace 1.0, attempt 2 at the clamp, then it stops rather than re-requesting
    # the same clamped pace forever.
    assert client.paces == [1.0, config.pace_max]


def test_an_impossible_underrun_clamps_at_the_floor(config: Config) -> None:
    """The measured real-world case on this clip: windows far longer than the speech."""
    client = SynthClient(base_duration_s=1.0)  # target 5.0 -> ratio 0.2
    result = fit_segment("नमस्ते", 5.0, config, client, segment_id=3)  # type: ignore[arg-type]

    assert result.clamped is True
    assert result.converged is False
    assert client.paces == [1.0, config.pace_min]
    assert all(p >= config.pace_min for p in client.paces)


def test_a_clamped_segment_still_ships_its_best_attempt(config: Config) -> None:
    """Clamping is not a failure to produce audio -- the closest attempt is still used."""
    client = SynthClient(base_duration_s=4.0)
    result = fit_segment("नमस्ते", 2.0, config, client, segment_id=0)  # type: ignore[arg-type]

    assert result.final.duration_s == min(a.duration_s for a in result.attempts)
    assert result.final.audio


def test_the_best_attempt_is_shipped_even_if_it_is_not_the_last(config: Config) -> None:
    """Measured on segment 0 of the real clip: a slower pace came back SHORTER, not longer.

    When the API moves the wrong way, shipping "the last attempt" would ship the worse
    one. The stage ships the attempt closest to target instead.
    """
    target = 5.0
    attempts = [
        Attempt(index=1, pace=1.0, duration_s=1.770,
                trimmed=trim_wav_silence(make_wav(1.77)), audio=b"a"),
        Attempt(index=2, pace=0.85, duration_s=1.680,
                trimmed=trim_wav_silence(make_wav(1.68)), audio=b"b"),
    ]
    result = FitResult(segment_id=0, target_duration_s=target, attempts=attempts,
                       converged=False, clamped=True, stop_reason="pinned_at_clamp")
    assert result.final.index == 1
    assert result.final.pace == 1.0


# --- the non-convergence path --------------------------------------------------------------

def test_non_convergence_stops_at_max_attempts(config: Config) -> None:
    """A segment that never lands in band must not spend credits forever.

    Elasticity 0 means pace has no effect on duration at all, so the loop can never
    converge. The 6% overrun is chosen to keep every proposed pace inside the clamp --
    a larger one would pin against the clamp and stop earlier for a different reason,
    which is a different test.
    """
    client = SynthClient(base_duration_s=2.12, elasticity=0.0)
    result = fit_segment("नमस्ते", 2.0, config, client, segment_id=0)  # type: ignore[arg-type]

    assert result.converged is False
    assert result.clamped is False, "this test must exercise max_attempts, not the clamp"
    assert len(result.attempts) == config.fit_max_attempts
    assert len(client.paces) == config.fit_max_attempts
    assert result.stop_reason == "max_attempts"


def test_max_attempts_is_respected_when_lowered(config: Config) -> None:
    """--fit-max-attempts caps synthesis calls per segment, including the baseline."""
    cfg = Config(**{**config.__dict__, "fit_max_attempts": 2})
    client = SynthClient(base_duration_s=2.12, elasticity=0.0)
    result = fit_segment("नमस्ते", 2.0, cfg, client, segment_id=0)  # type: ignore[arg-type]
    assert len(result.attempts) == 2


def test_a_tiny_pace_correction_is_not_worth_a_credit(config: Config) -> None:
    """Below the step threshold the change is inaudible; stop rather than re-synthesise.

    Unreachable with the default +/-5% band: the smallest out-of-band ratio moves pace by
    at least 0.05 x 0.85 = 0.0425, above the 0.02 threshold. It becomes reachable exactly
    when the band is tightened, which is when the guard earns its keep -- so the test
    tightens it.
    """
    cfg = Config(**{**config.__dict__, "fit_ratio_min": 0.995, "fit_ratio_max": 1.005})
    client = SynthClient(base_duration_s=2.0 * 1.008, elasticity=0.0)
    result = fit_segment("नमस्ते", 2.0, cfg, client, segment_id=0)  # type: ignore[arg-type]

    assert result.converged is False
    assert result.clamped is False
    assert result.stop_reason == "pace_step_below_threshold"
    # The guard fires before the second call, so the pointless synthesis never happens.
    assert len(result.attempts) == 1
    assert client.paces == [1.0]


# --- --no-fit -----------------------------------------------------------------------------

def test_no_fit_makes_exactly_one_call_at_pace_1(config: Config) -> None:
    """--no-fit must be independently runnable, not just a reinterpretation of attempt 1."""
    cfg = Config(**{**config.__dict__, "enable_duration_fit": False})
    client = SynthClient(base_duration_s=4.0)  # would otherwise trigger correction
    result = fit_segment("नमस्ते", 2.0, cfg, client, segment_id=0)  # type: ignore[arg-type]

    assert client.paces == [1.0]
    assert len(result.attempts) == 1
    assert result.stop_reason == "fit_disabled"
    assert result.clamped is False
    assert result.final.pace == 1.0


def test_no_fit_baseline_equals_the_fitted_runs_attempt_one(config: Config) -> None:
    """The comparison table may reuse attempt 1 precisely because they are the same call."""
    fitted = fit_segment(
        "नमस्ते", 2.0, config, SynthClient(base_duration_s=4.0), segment_id=0,  # type: ignore[arg-type]
    )
    unfitted = fit_segment(
        "नमस्ते", 2.0, Config(**{**config.__dict__, "enable_duration_fit": False}),
        SynthClient(base_duration_s=4.0), segment_id=0,  # type: ignore[arg-type]
    )
    assert fitted.baseline.pace == unfitted.final.pace == 1.0
    assert fitted.baseline.duration_s == pytest.approx(unfitted.final.duration_s, abs=1e-6)


# --- input validation -----------------------------------------------------------------------

def test_a_non_positive_window_is_rejected(config: Config) -> None:
    """M2 guarantees positive windows; a zero one would divide by zero downstream."""
    with pytest.raises(TtsError, match="non-positive target duration"):
        fit_segment("नमस्ते", 0.0, config, SynthClient(1.0), segment_id=0)  # type: ignore[arg-type]


def test_empty_text_is_rejected(config: Config) -> None:
    """Synthesising nothing would bill for nothing and leave a hole in the timeline."""
    with pytest.raises(TtsError, match="no text"):
        fit_segment("   ", 2.0, config, SynthClient(1.0), segment_id=0)  # type: ignore[arg-type]


def test_all_silent_synthesis_is_rejected(config: Config) -> None:
    """Shipping silence where a line should be is worse than failing the run."""
    class SilentClient(SynthClient):
        def text_to_speech(self, text: str, *, pace: float = 1.0, **kwargs: Any) -> dict[str, Any]:
            self.paces.append(pace)
            return {"request_id": "silent", "audios": [make_wav(1.0, amplitude=0)]}

    with pytest.raises(TtsError, match="no sample above"):
        fit_segment("नमस्ते", 2.0, config, SilentClient(1.0), segment_id=0)  # type: ignore[arg-type]


# --- aggregate metrics -----------------------------------------------------------------------

def _fit(segment_id: int, target: float, baseline: float, achieved: float,
         *, pace: float = 1.0, attempts: int = 1, clamped: bool = False,
         converged: bool = True) -> SegmentFit:
    """Build a SegmentFit for aggregate-statistics tests."""
    return SegmentFit(
        segment_id=segment_id, target_duration_s=target, achieved_duration_s=achieved,
        baseline_duration_s=baseline, final_pace=pace, attempts=attempts,
        clamped=clamped, converged=converged,
    )


def test_paired_before_after_is_reported_over_the_same_segments() -> None:
    """Both columns must describe the same segments, or the comparison means nothing."""
    metrics = MetricsCollector()
    metrics.duration_fit_enabled = True
    metrics.record_segment_fit(_fit(0, 2.0, 2.4, 2.0))
    metrics.record_segment_fit(_fit(1, 4.0, 5.0, 4.1))

    stats = metrics.duration_fit_stats()
    assert stats["n_segments"] == 2
    assert stats["without_fit"]["mean_abs_drift_pct"] == pytest.approx(22.5)
    assert stats["with_fit"]["mean_abs_drift_pct"] == pytest.approx(1.25)
    assert stats["improvement"]["segments_improved"] == 2
    assert stats["improvement"]["segments_worsened"] == 0


def test_small_sample_is_flagged() -> None:
    """With n=4 a p95 is arithmetic, not evidence; the flag says so in the file itself."""
    metrics = MetricsCollector()
    metrics.record_segment_fit(_fit(0, 2.0, 2.4, 2.0))
    assert metrics.duration_fit_stats()["small_sample"] is True


def test_clamp_rate_is_reported_as_count_and_percentage() -> None:
    """A clamp count without an n is unreadable; report both."""
    metrics = MetricsCollector()
    for index in range(4):
        metrics.record_segment_fit(_fit(index, 2.0, 4.0, 3.0, clamped=index < 3,
                                        converged=False))
    stats = metrics.duration_fit_stats()
    assert stats["clamped_segments"] == 3
    assert stats["clamped_pct"] == pytest.approx(75.0)
    assert stats["converged_segments"] == 0


def test_overrun_is_reported_separately_from_symmetric_drift() -> None:
    """An underrun is padded with silence and costs nothing; an overrun breaks sync."""
    metrics = MetricsCollector()
    metrics.record_segment_fit(_fit(0, 10.0, 2.0, 2.0))   # massive underrun
    metrics.record_segment_fit(_fit(1, 10.0, 12.0, 12.0))  # 20% overrun

    stats = metrics.duration_fit_stats()
    assert stats["with_fit"]["mean_abs_drift_pct"] == pytest.approx(50.0)
    assert stats["overrun_only"]["with_fit_mean_pct"] == pytest.approx(10.0)
    assert stats["overrun_only"]["with_fit_segments_overrunning"] == 1
    assert stats["direction"]["segments_underrunning"] == 1
    assert stats["direction"]["segments_overrunning"] == 1


def test_unmeasured_fit_reports_not_measured_rather_than_zero() -> None:
    """A zero here would read as a perfect fit; the file must say nothing was measured."""
    from src.metrics import NOT_MEASURED

    stats = MetricsCollector().duration_fit_stats()
    assert stats["with_fit"]["mean_abs_drift_pct"] == NOT_MEASURED
    assert stats["clamped_segments"] == NOT_MEASURED


# --- client-level validation -------------------------------------------------------------------

@pytest.fixture
def tts_response() -> dict[str, Any]:
    """A minimal /text-to-speech response carrying a real base64 WAV."""
    import base64

    return {
        "request_id": "20260729_probe",
        "audios": [base64.b64encode(make_wav(1.0)).decode("ascii")],
    }


def test_tts_request_carries_the_documented_json_body(
    config: Config, tts_response: dict[str, Any]
) -> None:
    """The body matches the confirmed schema exactly, with pace sent as a number."""
    client, session, _ = build_client(config, [FakeResponse(200, tts_response)])
    client.text_to_speech("नमस्ते", target_language_code="hi-IN", pace=1.25)

    call = session.calls[0]
    assert call["url"] == "https://api.sarvam.ai/text-to-speech"
    assert call["json"] == {
        "model": "bulbul:v3",
        "target_language_code": "hi-IN",
        "speaker": "shubh",
        "pace": 1.25,
        "speech_sample_rate": 24000,
        "output_audio_codec": "wav",
        "text": "नमस्ते",
    }


def test_pace_is_part_of_the_cache_key(config: Config, tts_response: dict[str, Any]) -> None:
    """Each attempt of the loop must be its own entry, or re-runs would replay one pace."""
    from src.cache import DiskCache

    cache = DiskCache(config.cache_dir)
    client, session, _ = build_client(config, [FakeResponse(200, tts_response)] * 2, cache=cache)
    client.text_to_speech("नमस्ते", target_language_code="hi-IN", pace=1.0)
    client.text_to_speech("नमस्ते", target_language_code="hi-IN", pace=1.25)
    assert len(session.calls) == 2


def test_repeating_the_whole_loop_is_free(config: Config, tts_response: dict[str, Any]) -> None:
    """The M1 guarantee applies per attempt: a re-run issues zero calls and costs zero."""
    from src.cache import DiskCache

    metrics = MetricsCollector()
    cache = DiskCache(config.cache_dir)
    warm, session, _ = build_client(config, [FakeResponse(200, tts_response)] * 2, metrics, cache)
    for pace in (1.0, 1.25):
        warm.text_to_speech("नमस्ते", target_language_code="hi-IN", pace=pace)

    replay, session2, _ = build_client(config, [], metrics, DiskCache(config.cache_dir))
    for pace in (1.0, 1.25):
        replay.text_to_speech("नमस्ते", target_language_code="hi-IN", pace=pace)

    assert len(session.calls) == 2 and len(session2.calls) == 0
    assert metrics.cache_hits == 2
    assert sum(c.cost_inr for c in metrics.api_calls if c.cache_hit) == 0.0


def test_tts_is_billed_per_input_character(config: Config, tts_response: dict[str, Any]) -> None:
    """Rs 30 per 10k characters for bulbul:v3, from the published price list."""
    metrics = MetricsCollector()
    client, _, _ = build_client(config, [FakeResponse(200, tts_response)], metrics)
    client.text_to_speech("x" * 400, target_language_code="hi-IN")

    call = metrics.api_calls[0]
    assert call.billable_units == 400
    assert call.cost_inr == pytest.approx(400 * 30.0 / 10_000)


def test_pace_outside_the_api_range_is_rejected_before_the_network(config: Config) -> None:
    """The clamp should make this unreachable; the guard is there in case it does not."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="pace"):
        client.text_to_speech("नमस्ते", target_language_code="hi-IN", pace=2.5)
    assert session.calls == []


def test_an_unknown_speaker_is_rejected(config: Config) -> None:
    """Speakers are model-specific and lowercase; catch a typo locally."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="not a bulbul:v3 voice"):
        client.text_to_speech("नमस्ते", target_language_code="hi-IN", speaker="Anushka")
    assert session.calls == []


def test_a_language_bulbul_cannot_speak_is_rejected(config: Config) -> None:
    """bulbul:v3 covers 11 languages, fewer than the translator's 23."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="not supported"):
        client.text_to_speech("नमस्ते", target_language_code="sat-IN")
    assert session.calls == []


def test_over_length_text_is_rejected(config: Config) -> None:
    """bulbul:v3 caps at 2500 chars; catch it locally rather than paying for a 422."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="2500"):
        client.text_to_speech("x" * 2501, target_language_code="hi-IN")
    assert session.calls == []


def test_a_response_without_audio_fails_loudly(config: Config) -> None:
    """A missing audios array must not be silently treated as an empty clip."""
    client, _, _ = build_client(config, [FakeResponse(200, {"request_id": "x"})])
    payload = client.text_to_speech("नमस्ते", target_language_code="hi-IN")
    with pytest.raises(SarvamAPIError, match="no audio"):
        client.audio_bytes(payload)


def test_invalid_base64_fails_loudly(config: Config) -> None:
    """Corrupt audio is a failure, not a zero-length segment."""
    client, _, _ = build_client(config, [FakeResponse(200, {"audios": ["!!!not base64!!!"]})])
    payload = client.text_to_speech("नमस्ते", target_language_code="hi-IN")
    with pytest.raises(SarvamAPIError, match="base64"):
        client.audio_bytes(payload)


# --- end-to-end stage --------------------------------------------------------------------------

SEGMENTS = [
    {"id": 0, "start": 0.0, "end": 2.0, "text": "Okay, Jay, pick a card.",
     "translation": "ठीक है, Jay, एक card चुन लीजिए।", "speaker": None},
    {"id": 1, "start": 2.0, "end": 6.0, "text": "How did you do that?",
     "translation": "आपने ये कैसे किया?", "speaker": None},
]


class StageClient(SynthClient):
    """SynthClient that also base64-encodes, matching the real response shape."""

    @staticmethod
    def audio_bytes(payload: dict[str, Any]) -> bytes:
        """Return the WAV bytes from the synthetic payload."""
        return payload["audios"][0]


def test_run_tts_writes_fitted_and_baseline_audio(config: Config, tmp_path: Path) -> None:
    """Both clips are persisted: the fitted one to ship, the baseline as the comparison."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    (output_dir / "segments.json").write_text(
        json.dumps(SEGMENTS, ensure_ascii=False), encoding="utf-8"
    )

    cfg = Config(**{**config.__dict__, "output_dir": output_dir})
    metrics = MetricsCollector()
    report = run_tts(cfg, StageClient(base_duration_s=2.4), metrics)  # type: ignore[arg-type]

    audio_dir = output_dir / "audio_segments"
    for segment_id in (0, 1):
        assert (audio_dir / f"seg_{segment_id:03d}.wav").exists()
        assert (audio_dir / f"seg_{segment_id:03d}_nofit.wav").exists()

    assert len(report["per_segment"]) == 2
    assert (output_dir / "tts_report.json").exists()
    assert len(metrics.segment_fits) == 2


def test_run_tts_records_every_field_spec_asks_for(config: Config, tmp_path: Path) -> None:
    """The per-segment record is the deliverable of this milestone."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    (output_dir / "segments.json").write_text(
        json.dumps(SEGMENTS, ensure_ascii=False), encoding="utf-8"
    )
    cfg = Config(**{**config.__dict__, "output_dir": output_dir})
    report = run_tts(cfg, StageClient(base_duration_s=2.4), MetricsCollector())  # type: ignore[arg-type]

    record = report["per_segment"][0]
    for field in ("target_duration_s", "baseline_duration_s", "final_duration_s",
                  "final_pace", "attempts", "converged", "clamped", "residual_drift_pct"):
        assert field in record, field


def test_run_tts_refuses_untranslated_segments(config: Config, tmp_path: Path) -> None:
    """Synthesising the English source would silently produce the wrong dub."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    (output_dir / "segments.json").write_text(
        json.dumps([{"id": 0, "start": 0.0, "end": 2.0, "text": "hello"}]), encoding="utf-8"
    )
    cfg = Config(**{**config.__dict__, "output_dir": output_dir})
    with pytest.raises(TtsError, match="no segment .* has a translation"):
        run_tts(cfg, StageClient(2.0), MetricsCollector())  # type: ignore[arg-type]


def test_run_tts_names_the_stage_to_run_when_input_is_missing(
    config: Config, tmp_path: Path
) -> None:
    """A missing segments.json points at --stage translate rather than crashing obscurely."""
    cfg = Config(**{**config.__dict__, "output_dir": tmp_path / "empty"})
    with pytest.raises(TtsError, match="--stage translate"):
        run_tts(cfg, StageClient(2.0), MetricsCollector())  # type: ignore[arg-type]
