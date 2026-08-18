"""Tests for the mux stage: the video must survive untouched and the dub must be present.

These tests do call real ffmpeg, on a one-second clip generated on the fly. That is
deliberate and is what SPEC.md asks for: the whole claim of this stage is that `-c:v copy`
passes the picture through unchanged, and only a real remux can demonstrate that. The
fixture is small enough that the suite stays fast.

The pure verification logic is tested separately with synthetic probes, so the failure
paths (a re-encoded video, a truncated duration, a missing audio track) are reachable
without having to persuade ffmpeg to actually misbehave.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from src.config import Config
from src.stages.mux import (
    MuxError,
    _verify,
    probe_audio_stream,
    probe_video_stream,
    run_mux,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

VIDEO_DURATION_S = 1.0


# --- fixtures ---------------------------------------------------------------------------

@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """A 1-second 160x120 H.264 clip with its own audio track, built by ffmpeg."""
    path = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={VIDEO_DURATION_S}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={VIDEO_DURATION_S}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return path


@pytest.fixture
def dubbed_wav(tmp_path: Path) -> Path:
    """A 1-second 24 kHz mono PCM WAV standing in for the assembled dub."""
    path = tmp_path / "out" / "dubbed_audio.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x11\x11" * 24000)
    return path


@pytest.fixture
def mux_config(tmp_path: Path, tiny_video: Path, dubbed_wav: Path) -> Config:
    """A Config pointing at the generated clip, writing into the temp output directory."""
    return Config(
        api_key="sk_test_not_a_real_key",
        input_path=tiny_video,
        output_path=tmp_path / "out" / "dubbed.mp4",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )


# --- the real remux -----------------------------------------------------------------------

def test_the_video_stream_is_copied_not_re_encoded(mux_config: Config) -> None:
    """Codec, size, pixel format, and frame count all survive the remux identically."""
    report = run_mux(mux_config)

    source, output = report["source"]["video"], report["output"]["video"]
    assert output["codec_name"] == source["codec_name"] == "h264"
    assert (output["width"], output["height"]) == (source["width"], source["height"])
    assert output["pix_fmt"] == source["pix_fmt"]
    assert all(check["passed"] for check in report["checks"].values())


def test_the_output_carries_the_dubbed_audio_not_the_source_audio(mux_config: Config) -> None:
    """The new track is the dub: 24 kHz mono, matching the assembled WAV, not the 44.1 kHz source."""
    run_mux(mux_config)

    audio = probe_audio_stream(mux_config.output_path)
    assert audio is not None
    assert audio["sample_rate"] == 24000
    assert audio["channels"] == 1
    assert audio["codec_name"] == "aac"


def test_the_output_is_as_long_as_the_source(mux_config: Config) -> None:
    """A dub that is shorter than its video has dropped something."""
    report = run_mux(mux_config)

    assert report["checks"]["duration_matches_source"]["passed"]
    assert report["output"]["duration_s"] == pytest.approx(VIDEO_DURATION_S, abs=0.25)


def test_a_report_is_written_with_the_exact_command(mux_config: Config) -> None:
    """The ffmpeg command is on the record, so the claim about -c:v copy is checkable."""
    run_mux(mux_config)

    report = json.loads((mux_config.output_dir / "mux_report.json").read_text(encoding="utf-8"))
    assert "-c:v" in report["ffmpeg_command"]
    assert report["ffmpeg_command"][report["ffmpeg_command"].index("-c:v") + 1] == "copy"
    assert "Demucs" in report["audio_policy"]


def test_an_audio_only_input_degrades_to_an_audio_only_output(
    tmp_path: Path, dubbed_wav: Path
) -> None:
    """A source with no picture still produces an artefact rather than failing the run."""
    source_audio = tmp_path / "source.wav"
    with wave.open(str(source_audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16000)

    config = Config(
        api_key="sk_test_not_a_real_key",
        input_path=source_audio,
        output_path=tmp_path / "out" / "dubbed.m4a",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )
    report = run_mux(config)

    assert report["output"]["video"] is None
    assert report["output"]["audio"]["codec_name"] == "aac"
    assert config.output_path.exists()


def test_a_missing_dub_names_the_stage_to_run(mux_config: Config, dubbed_wav: Path) -> None:
    """The error points at --stage assemble instead of just reporting a missing file."""
    dubbed_wav.unlink()

    with pytest.raises(MuxError, match="--stage assemble"):
        run_mux(mux_config)


def test_probing_a_file_with_no_video_returns_none(dubbed_wav: Path) -> None:
    """A WAV has no video stream, and that is a None rather than an exception."""
    assert probe_video_stream(dubbed_wav) is None
    assert probe_audio_stream(dubbed_wav) is not None


# --- verification logic, without needing ffmpeg to misbehave ---------------------------------

def test_verification_fails_when_the_video_was_re_encoded() -> None:
    """A changed codec means the picture was re-encoded, which is exactly what -c:v copy prevents."""
    checks = _verify(
        source_video={"codec_name": "h264", "width": 640, "height": 360,
                      "pix_fmt": "yuv420p", "nb_frames": 100},
        output_video={"codec_name": "hevc", "width": 640, "height": 360,
                      "pix_fmt": "yuv420p", "nb_frames": 100},
        output_audio={"codec_name": "aac"},
        source_duration_s=10.0, output_duration_s=10.0,
    )

    assert checks["video_codec_name_unchanged"]["passed"] is False
    assert checks["video_width_unchanged"]["passed"] is True


def test_verification_fails_on_a_dropped_frame() -> None:
    """A frame count that shrank means the copy truncated the picture."""
    checks = _verify(
        source_video={"codec_name": "h264", "width": 640, "height": 360,
                      "pix_fmt": "yuv420p", "nb_frames": 100},
        output_video={"codec_name": "h264", "width": 640, "height": 360,
                      "pix_fmt": "yuv420p", "nb_frames": 99},
        output_audio={"codec_name": "aac"},
        source_duration_s=10.0, output_duration_s=10.0,
    )

    assert checks["video_frame_count_unchanged"]["passed"] is False


def test_verification_fails_when_there_is_no_audio_track() -> None:
    """A silent 'dub' is the failure this check exists to catch."""
    checks = _verify(
        source_video=None, output_video=None, output_audio=None,
        source_duration_s=10.0, output_duration_s=10.0,
    )

    assert checks["audio_track_present"]["passed"] is False


def test_verification_tolerates_container_rounding_but_not_truncation() -> None:
    """Tens of milliseconds are AAC priming; seconds are a lost stream."""
    within = _verify(
        source_video=None, output_video=None, output_audio={"codec_name": "aac"},
        source_duration_s=36.107, output_duration_s=36.130,
    )
    beyond = _verify(
        source_video=None, output_video=None, output_audio={"codec_name": "aac"},
        source_duration_s=36.107, output_duration_s=30.000,
    )

    assert within["duration_matches_source"]["passed"] is True
    assert beyond["duration_matches_source"]["passed"] is False
    assert beyond["duration_matches_source"]["drift_s"] == pytest.approx(6.107, abs=1e-3)


def test_a_failed_check_stops_the_run(mux_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """A remux that silently re-encoded must fail the stage, not be logged and shipped."""
    import src.stages.mux as mux_module

    real_probe = mux_module.probe_video_stream
    calls: list[Path] = []

    def fake_probe(path):  # type: ignore[no-untyped-def]
        """Report a different codec for the output only, simulating a silent re-encode."""
        calls.append(Path(path))
        probed = real_probe(path)
        if probed and len(calls) > 1:
            probed["codec_name"] = "hevc"
        return probed

    monkeypatch.setattr(mux_module, "probe_video_stream", fake_probe)

    with pytest.raises(MuxError, match="post-mux check"):
        run_mux(mux_config)
