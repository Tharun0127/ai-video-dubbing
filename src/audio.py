"""
Audio utilities: ffmpeg/ffprobe wrappers and duration probing.

M1 scope is deliberately narrow: the client needs an accurate audio duration to
compute per-second STT cost, and the ASR stage needs to enforce the 30-second
REST cap. Silence detection, chunking, and fades arrive with M2/M5.

Input:  a path to an audio or video file.
Output: measured duration in seconds (and, later, chunk boundaries).
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import subprocess
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioError(RuntimeError):
    """Raised when a media file cannot be probed or processed."""


def ffprobe_path() -> str:
    """Return the ffprobe executable path, raising if it is not on PATH."""
    found = shutil.which("ffprobe")
    if not found:
        raise AudioError(
            "ffprobe not found on PATH. Install ffmpeg (https://ffmpeg.org/download.html) "
            "and ensure its bin/ directory is on PATH."
        )
    return found


def probe_duration_s(path: str | Path) -> float:
    """Return the duration of an audio/video file in seconds, measured not estimated."""
    media = Path(path)
    if not media.exists():
        raise AudioError(f"media file does not exist: {media}")

    # Fast path: a plain PCM WAV can be read exactly without spawning a process.
    if media.suffix.lower() == ".wav":
        with contextlib.suppress(wave.Error, EOFError, OSError):
            with wave.open(str(media), "rb") as handle:
                frames, rate = handle.getnframes(), handle.getframerate()
                if rate > 0:
                    return frames / float(rate)

    result = subprocess.run(
        [
            ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(media),
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise AudioError(f"ffprobe failed on {media}: {result.stderr.strip()}")

    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise AudioError(f"could not parse ffprobe output for {media}: {result.stdout!r}") from exc

    logger.debug("probed duration of %s: %.3fs", media.name, duration)
    return duration


def probe_stream_info(path: str | Path) -> dict[str, object]:
    """Return measured duration, sample rate, and channel count for the first audio stream."""
    media = Path(path)
    result = subprocess.run(
        [
            ffprobe_path(), "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,codec_name",
            "-show_entries", "format=duration", "-of", "json", str(media),
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise AudioError(f"ffprobe failed on {media}: {result.stderr.strip()}")

    parsed = json.loads(result.stdout)
    streams = parsed.get("streams") or [{}]
    stream = streams[0]
    return {
        "duration_s": float(parsed.get("format", {}).get("duration", 0.0)),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": stream.get("channels"),
        "codec_name": stream.get("codec_name"),
    }
