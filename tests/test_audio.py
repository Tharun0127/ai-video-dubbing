"""Tests for audio probing, run against real ffmpeg on tiny generated fixtures."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.audio import AudioError, probe_duration_s, probe_stream_info

ffmpeg_required = pytest.mark.skipif(
    subprocess.run(["ffprobe", "-version"], capture_output=True, check=False).returncode != 0,
    reason="ffprobe not on PATH",
)


def test_wav_duration_is_read_without_ffmpeg(tiny_wav: Path) -> None:
    """The stdlib fast path measures a PCM WAV exactly, no subprocess needed."""
    assert probe_duration_s(tiny_wav) == pytest.approx(0.5)


def test_missing_file_raises(tmp_path: Path) -> None:
    """Probing a nonexistent file fails with the path in the message."""
    with pytest.raises(AudioError, match="does not exist"):
        probe_duration_s(tmp_path / "nope.wav")


@ffmpeg_required
def test_stream_info_matches_the_generated_fixture(tmp_path: Path) -> None:
    """ffprobe reports the sample rate and channel count we asked ffmpeg to write."""
    clip = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(clip)],
        check=True, capture_output=True,
    )
    info = probe_stream_info(clip)
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["duration_s"] == pytest.approx(1.0, abs=0.05)


@ffmpeg_required
def test_non_wav_duration_falls_back_to_ffprobe(tmp_path: Path) -> None:
    """A non-WAV container is measured through ffprobe."""
    clip = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-ac", "1", "-y", str(clip)],
        check=True, capture_output=True,
    )
    assert probe_duration_s(clip) == pytest.approx(1.0, abs=0.1)
