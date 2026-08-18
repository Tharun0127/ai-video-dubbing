"""Tests for the assemble stage: timeline placement, fades, mixing, and overlap detection.

No test opens a socket and none calls ffmpeg for the mixing paths -- the timeline is built
from WAVs written with the stdlib, so the arithmetic that decides sync is verified exactly
rather than approximately.

The properties that matter here are positional: a clip must land at the sample its segment
started at, a clip that overruns must stay audible rather than be truncated, and the
finished track must be exactly as long as the source. Each of those is asserted directly.
"""

from __future__ import annotations

import json
import math
import wave
from array import array
from pathlib import Path
from typing import Any

import pytest

from src.audio import (
    apply_fade_int16,
    int16_from_wav_bytes,
    mix_int16,
    read_wav_int16,
    seconds_to_frames,
    silent_int16,
)
from src.config import Config
from src.stages.assemble import (
    AssembleError,
    Placement,
    detect_overlaps,
    run_assemble,
)

SAMPLE_RATE = 24000


# --- fixtures ---------------------------------------------------------------------------

def write_tone_wav(
    path: Path,
    duration_s: float,
    *,
    amplitude: int = 12000,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
) -> Path:
    """Write a mono 16-bit tone of a known duration, so its frame count is predictable."""
    frames = int(round(duration_s * sample_rate))
    samples = array("h", (
        int(amplitude * math.sin(2 * math.pi * 220 * (n // channels) / sample_rate))
        for n in range(frames * channels)
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    return path


def write_silent_16k(path: Path, duration_s: float) -> Path:
    """Write the demuxed-source stand-in whose length defines the timeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * int(round(duration_s * 16000)))
    return path


@pytest.fixture
def assembled(tmp_path: Path, config: Config) -> tuple[Config, dict[str, Any]]:
    """A config plus a tts_report.json describing three clips on a 20-second timeline."""
    out = config.output_dir
    write_silent_16k(out / "audio_16k_mono.wav", 20.0)

    files = []
    for index, (start, end, duration, lead) in enumerate(
        [(0.0, 5.0, 2.0, 0.05), (5.0, 12.0, 3.0, 0.0), (12.0, 20.0, 1.5, 0.02)]
    ):
        fitted = out / "audio_segments" / f"seg_{index:03d}.wav"
        baseline = out / "audio_segments" / f"seg_{index:03d}_nofit.wav"
        write_tone_wav(fitted, duration)
        write_tone_wav(baseline, duration + 0.5)
        files.append({
            "segment_id": index,
            "start": start,
            "end": end,
            "text": f"source {index}",
            "translation": f"translation {index}",
            "fitted_path": str(fitted),
            "baseline_path": str(baseline),
            "identical_to_baseline": False,
            "lead_trim_s": lead,
            "trail_trim_s": 0.0,
        })

    report = {"files": files, "audio_dir": str(out / "audio_segments")}
    (out / "tts_report.json").write_text(json.dumps(report), encoding="utf-8")
    return config, report


# --- primitives ---------------------------------------------------------------------------

def test_silence_is_allocated_at_the_requested_length() -> None:
    """A silent canvas holds exactly frames x channels zeroed samples."""
    canvas = silent_int16(100, 2)
    assert len(canvas) == 200
    assert set(canvas) == {0}


def test_fade_ramps_both_ends_and_leaves_the_middle_alone() -> None:
    """The fade touches only the first and last few milliseconds of a clip."""
    samples = array("h", [10000] * 2400)  # 100 ms at 24 kHz
    fade_frames = apply_fade_int16(samples, channels=1, frame_rate=24000, fade_ms=15.0)

    assert fade_frames == 360
    assert samples[0] == 0
    assert samples[-1] == 0
    assert samples[fade_frames // 2] < 10000
    assert samples[1200] == 10000  # the middle is untouched


def test_fade_never_exceeds_half_a_short_clip() -> None:
    """A clip shorter than two fade lengths still fades symmetrically instead of overlapping."""
    samples = array("h", [10000] * 100)
    fade_frames = apply_fade_int16(samples, channels=1, frame_rate=24000, fade_ms=15.0)

    assert fade_frames == 50
    assert samples[0] == 0
    assert samples[-1] == 0


def test_a_zero_fade_is_a_no_op() -> None:
    """--fade-ms 0 disables the ramp rather than silencing a sample."""
    samples = array("h", [10000] * 100)
    assert apply_fade_int16(samples, channels=1, frame_rate=24000, fade_ms=0.0) == 0
    assert set(samples) == {10000}


def test_mixing_writes_at_the_requested_offset() -> None:
    """A clip lands at exactly the frame it was placed at -- this is what sync means."""
    canvas = silent_int16(100)
    result = mix_int16(canvas, array("h", [500] * 10), offset_frames=40)

    assert result["written"] == 10
    assert canvas[39] == 0
    assert canvas[40] == 500
    assert canvas[49] == 500
    assert canvas[50] == 0


def test_mixing_sums_overlaps_instead_of_truncating_them() -> None:
    """Where two clips collide both stay audible, and the collision is counted."""
    canvas = silent_int16(100)
    mix_int16(canvas, array("h", [100] * 20), offset_frames=0)
    result = mix_int16(canvas, array("h", [200] * 20), offset_frames=10)

    assert canvas[5] == 100          # only the first clip
    assert canvas[15] == 300         # both, summed
    assert canvas[25] == 200         # only the second clip
    assert result["overlapped"] == 10


def test_mixing_clamps_instead_of_wrapping() -> None:
    """A sum past full scale is clamped and counted; wrapping would be an audible crack."""
    canvas = silent_int16(10)
    mix_int16(canvas, array("h", [30000] * 10), offset_frames=0)
    result = mix_int16(canvas, array("h", [30000] * 10), offset_frames=0)

    assert set(canvas) == {32767}
    assert result["clipped"] == 10


def test_mixing_past_the_end_reports_what_it_dropped() -> None:
    """Audio that falls off the end of the timeline is counted, never silently lost."""
    canvas = silent_int16(10)
    result = mix_int16(canvas, array("h", [100] * 20), offset_frames=5)

    assert result["written"] == 5
    assert result["truncated"] == 15


def test_mixing_entirely_past_the_end_writes_nothing() -> None:
    """An offset beyond the canvas drops the whole clip and says so."""
    canvas = silent_int16(10)
    result = mix_int16(canvas, array("h", [100] * 4), offset_frames=50)

    assert result["written"] == 0
    assert result["truncated"] == 4


def test_seconds_to_frames_rounds_to_the_nearest_sample() -> None:
    """Rounding, not truncation: truncating every placement would drift the timeline early."""
    assert seconds_to_frames(1.0, 24000) == 24000
    assert seconds_to_frames(0.9999999, 24000) == 24000
    assert seconds_to_frames(0.00002, 24000) == 0


# --- overlap detection ----------------------------------------------------------------------

def make_placement(segment_id: int, placed_start_s: float, duration_s: float) -> Placement:
    """Build a Placement with only the fields overlap detection reads."""
    return Placement(
        segment_id=segment_id, window_start_s=placed_start_s,
        window_end_s=placed_start_s + duration_s, placed_start_s=placed_start_s,
        duration_s=duration_s, lead_trim_s=0.0, source_path=Path("x.wav"), fitted=True,
        fade_frames=0, clipped_samples=0, overlapped_frames=0, truncated_frames=0,
    )


def test_adjacent_clips_are_not_an_overlap() -> None:
    """A clip ending exactly where the next begins is contiguous, not colliding."""
    assert detect_overlaps([make_placement(0, 0.0, 5.0), make_placement(1, 5.0, 3.0)]) == []


def test_an_overrunning_clip_is_detected_with_its_size() -> None:
    """An overlap is reported with which segments collided and by how much."""
    overlaps = detect_overlaps([make_placement(0, 0.0, 6.0), make_placement(1, 5.0, 3.0)])

    assert len(overlaps) == 1
    assert overlaps[0]["segment_ids"] == [0, 1]
    assert overlaps[0]["overlap_s"] == pytest.approx(1.0)


def test_overlaps_are_found_in_timeline_order_not_list_order() -> None:
    """Clips are compared by where they were placed, not by the order they arrived in."""
    overlaps = detect_overlaps([make_placement(1, 5.0, 3.0), make_placement(0, 0.0, 6.0)])
    assert [o["segment_ids"] for o in overlaps] == [[0, 1]]


# --- the stage ------------------------------------------------------------------------------

def test_the_track_is_exactly_as_long_as_the_source(
    assembled: tuple[Config, dict[str, Any]]
) -> None:
    """The dubbed track matches the source length, so the mux cannot shorten the output."""
    config, _ = assembled
    report = run_assemble(config)

    assert report["timeline"]["written_duration_s"] == pytest.approx(20.0, abs=1e-3)
    assert report["timeline"]["overflow_past_source_s"] == pytest.approx(0.0)

    info, _ = read_wav_int16(config.output_dir / "dubbed_audio.wav")
    assert info.duration_s == pytest.approx(20.0, abs=1e-3)
    assert info.frame_rate == SAMPLE_RATE


def test_each_clip_is_placed_at_its_segment_start_plus_the_lead_trim(
    assembled: tuple[Config, dict[str, Any]]
) -> None:
    """M4 trimmed the synthesiser's onset padding to measure speech; M5 puts it back."""
    config, _ = assembled
    report = run_assemble(config)

    placed = {p["segment_id"]: p for p in report["placements"]}
    assert placed[0]["placed_start_s"] == pytest.approx(0.05)
    assert placed[1]["placed_start_s"] == pytest.approx(5.0)
    assert placed[2]["placed_start_s"] == pytest.approx(12.02)
    assert all(p["lead_trim_restored_s"] >= 0 for p in report["placements"])


def test_the_audio_really_starts_where_the_report_says(
    assembled: tuple[Config, dict[str, Any]]
) -> None:
    """The report is checked against the samples, not trusted: silence before, speech after."""
    config, _ = assembled
    run_assemble(config)
    _, samples = read_wav_int16(config.output_dir / "dubbed_audio.wav")

    def loudest_near(second: float, window_ms: float = 20.0) -> int:
        """Peak amplitude in a short window -- a single sample can land on a zero crossing."""
        start = seconds_to_frames(second, SAMPLE_RATE)
        span = seconds_to_frames(window_ms / 1000.0, SAMPLE_RATE)
        return max(abs(s) for s in samples[start:start + span])

    # Segment 1 is placed at 5.0 s with no lead trim, and its fade means the first samples
    # ramp from zero -- so probe just inside the clip rather than exactly on the boundary.
    assert loudest_near(4.9) == 0
    assert loudest_near(5.5) > 1000
    # The gap between segment 1 (ends 8.0 s) and segment 2 (starts 12.02 s) stays silent.
    assert loudest_near(10.0) == 0


def test_gaps_between_segments_stay_silent(assembled: tuple[Config, dict[str, Any]]) -> None:
    """Speech occupies only what was synthesised; the rest of the timeline is silence."""
    config, _ = assembled
    report = run_assemble(config)

    assert report["placement"]["speech_s"] == pytest.approx(6.5, abs=1e-3)
    assert report["placement"]["silence_s"] == pytest.approx(13.5, abs=1e-3)


def test_no_fit_assembles_the_unfitted_baseline_clips(
    assembled: tuple[Config, dict[str, Any]], tmp_path: Path
) -> None:
    """--no-fit builds a genuinely unfitted dub, which is what makes the A/B audible."""
    config, _ = assembled
    unfitted = Config(
        api_key=config.api_key, input_path=config.input_path,
        output_path=config.output_path, cache_dir=config.cache_dir,
        output_dir=config.output_dir, enable_duration_fit=False,
    )
    report = run_assemble(unfitted)

    assert report["placement"]["clip_variant"] == "baseline_nofit"
    assert all(p["source_path"].endswith("_nofit.wav") for p in report["placements"])
    # The baseline clips are each 0.5 s longer than the fitted ones by construction.
    assert report["placement"]["speech_s"] == pytest.approx(8.0, abs=1e-3)


def test_a_missing_clip_fails_loudly(assembled: tuple[Config, dict[str, Any]]) -> None:
    """A clip named in the report but absent from disk stops the run; it would be a silent hole."""
    config, _ = assembled
    Path(json.loads((config.output_dir / "tts_report.json").read_text(encoding="utf-8"))
         ["files"][1]["fitted_path"]).unlink()

    with pytest.raises(AssembleError, match="does not exist"):
        run_assemble(config)


def test_clips_with_different_sample_rates_are_rejected(
    assembled: tuple[Config, dict[str, Any]]
) -> None:
    """Mixing 24 kHz and 16 kHz clips without resampling would play one of them at the wrong pitch."""
    config, _ = assembled
    report = json.loads((config.output_dir / "tts_report.json").read_text(encoding="utf-8"))
    write_tone_wav(Path(report["files"][1]["fitted_path"]), 3.0, sample_rate=16000)

    with pytest.raises(AssembleError, match="do not share one audio format"):
        run_assemble(config)


def test_a_clip_overrunning_the_source_extends_the_track_and_is_reported(
    tmp_path: Path, config: Config
) -> None:
    """A final line that runs past the source is kept and measured, not cut off."""
    out = config.output_dir
    write_silent_16k(out / "audio_16k_mono.wav", 5.0)
    clip = write_tone_wav(out / "audio_segments" / "seg_000.wav", 4.0)
    (out / "tts_report.json").write_text(json.dumps({"files": [{
        "segment_id": 0, "start": 3.0, "end": 5.0, "fitted_path": str(clip),
        "baseline_path": str(clip), "lead_trim_s": 0.0, "trail_trim_s": 0.0,
    }]}), encoding="utf-8")

    report = run_assemble(config)

    assert report["timeline"]["overflow_past_source_s"] == pytest.approx(2.0, abs=1e-3)
    assert report["timeline"]["written_duration_s"] == pytest.approx(7.0, abs=1e-3)
    assert report["placement"]["frames_dropped_past_end"] == 0


def test_overlapping_clips_are_mixed_and_reported(tmp_path: Path, config: Config) -> None:
    """Two clips sharing a window both survive into the track, and the overlap is a finding."""
    out = config.output_dir
    write_silent_16k(out / "audio_16k_mono.wav", 10.0)
    first = write_tone_wav(out / "audio_segments" / "seg_000.wav", 4.0)
    second = write_tone_wav(out / "audio_segments" / "seg_001.wav", 3.0)
    (out / "tts_report.json").write_text(json.dumps({"files": [
        {"segment_id": 0, "start": 0.0, "end": 3.0, "fitted_path": str(first),
         "baseline_path": str(first), "lead_trim_s": 0.0, "trail_trim_s": 0.0},
        {"segment_id": 1, "start": 3.0, "end": 8.0, "fitted_path": str(second),
         "baseline_path": str(second), "lead_trim_s": 0.0, "trail_trim_s": 0.0},
    ]}), encoding="utf-8")

    report = run_assemble(config)

    assert report["overlaps"]["count"] == 1
    assert report["overlaps"]["total_overlap_s"] == pytest.approx(1.0, abs=1e-3)
    assert report["placements"][1]["overlapped_frames"] > 0


def test_missing_tts_report_names_the_stage_to_run(config: Config) -> None:
    """The error tells the user which stage to run, not just that a file is absent."""
    with pytest.raises(AssembleError, match="--stage tts"):
        run_assemble(config)


def test_an_empty_clip_list_is_rejected(config: Config) -> None:
    """A report with no clips would produce a silent 'dub'."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "tts_report.json").write_text(json.dumps({"files": []}), encoding="utf-8")

    with pytest.raises(AssembleError, match="no synthesised clips"):
        run_assemble(config)


def test_fill_percentage_is_measured_against_the_window(
    assembled: tuple[Config, dict[str, Any]]
) -> None:
    """Fill exposes how much of its window each line actually uses -- the M4 drift, restated."""
    config, _ = assembled
    report = run_assemble(config)

    placed = {p["segment_id"]: p for p in report["placements"]}
    assert placed[0]["fill_pct"] == pytest.approx(40.0, abs=0.5)   # 2.0 s of 5.0 s
    assert placed[1]["fill_pct"] == pytest.approx(42.86, abs=0.5)  # 3.0 s of 7.0 s


def test_int16_parsing_rejects_non_16_bit_audio() -> None:
    """8-bit audio would be misread as noise if it were parsed as int16."""
    import io as _io

    buffer = _io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(1)
        writer.setframerate(24000)
        writer.writeframes(b"\x80" * 100)

    with pytest.raises(Exception, match="16-bit"):
        int16_from_wav_bytes(buffer.getvalue())
