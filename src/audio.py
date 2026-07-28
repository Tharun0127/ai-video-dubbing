"""
Audio utilities: ffmpeg/ffprobe wrappers, duration probing, silence detection, slicing.

M1 needed only duration probing (to price STT per second and to enforce the 30-second
REST cap). M2 adds the three primitives the ASR stage is built on:

  * `extract_audio_16k_mono` -- real format conversion to the shape Saaras wants
    (16 kHz mono 16-bit PCM WAV), leaving the source video stream untouched.
  * `detect_silences`        -- ffmpeg's `silencedetect` filter, parsed into typed spans.
  * `write_wav_slice`        -- sample-accurate WAV cutting via the stdlib, so a chunk
    written twice is byte-identical and therefore hashes to the same cache key.

Slicing deliberately avoids ffmpeg: re-encoding through a subprocess risks a
non-deterministic byte stream (encoder metadata, timestamp rounding), and a chunk whose
bytes change between runs would silently miss the cache and re-spend credits.

Input:  a path to an audio or video file.
Output: measured durations, silence spans, and 16 kHz mono WAV files.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Sample rate Sarvam documents as recommended for /speech-to-text.
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM

#: `silencedetect` writes these to stderr, one per line, e.g. "silence_start: 20.878125".
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


class AudioError(RuntimeError):
    """Raised when a media file cannot be probed or processed."""


@dataclass(frozen=True)
class Silence:
    """One detected silent span on the audio timeline, in seconds from the file start."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Length of the silent span in seconds."""
        return self.end_s - self.start_s

    @property
    def midpoint_s(self) -> float:
        """Centre of the silent span -- the safest place to cut, being furthest from speech."""
        return (self.start_s + self.end_s) / 2.0

    def to_dict(self) -> dict[str, float]:
        """Serialise for the chunk report."""
        return {
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "midpoint_s": round(self.midpoint_s, 6),
        }


def ffprobe_path() -> str:
    """Return the ffprobe executable path, raising if it is not on PATH."""
    found = shutil.which("ffprobe")
    if not found:
        raise AudioError(
            "ffprobe not found on PATH. Install ffmpeg (https://ffmpeg.org/download.html) "
            "and ensure its bin/ directory is on PATH."
        )
    return found


def ffmpeg_path() -> str:
    """Return the ffmpeg executable path, raising if it is not on PATH."""
    found = shutil.which("ffmpeg")
    if not found:
        raise AudioError(
            "ffmpeg not found on PATH. Install ffmpeg (https://ffmpeg.org/download.html) "
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


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------

def extract_audio_16k_mono(
    source: str | Path,
    dest: str | Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict[str, object]:
    """Extract the audio stream to 16 kHz mono 16-bit PCM WAV, leaving the video untouched."""
    src, out = Path(source), Path(dest)
    if not src.exists():
        raise AudioError(f"input media does not exist: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # -vn drops the video (we never re-encode it; M5 remuxes the original stream with
    # -c:v copy). -map_metadata -1 and the bitexact flags strip encoder/creation-time
    # tags, so the same input always produces byte-identical output -- which is what
    # keeps the downstream chunk hashes, and therefore the cache keys, stable.
    command = [
        ffmpeg_path(), "-v", "error", "-nostdin", "-y",
        "-i", str(src),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-map_metadata", "-1",
        "-fflags", "+bitexact", "-flags:a", "+bitexact",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AudioError(
            f"ffmpeg failed extracting audio from {src} -> {out}: {result.stderr.strip()}"
        )
    if not out.exists() or out.stat().st_size == 0:
        raise AudioError(f"ffmpeg produced no audio for {src}; is there an audio stream?")

    info = probe_stream_info(out)
    logger.info(
        "demux: %s -> %s (%d Hz, %d ch, pcm_s16le, %.3fs)",
        src.name, out.name, sample_rate, TARGET_CHANNELS, info["duration_s"],
    )
    return info


# --------------------------------------------------------------------------------------
# Silence detection
# --------------------------------------------------------------------------------------

def detect_silences(
    wav_path: str | Path,
    *,
    threshold_db: float = -40.0,
    min_silence_s: float = 0.30,
    total_duration_s: float | None = None,
) -> list[Silence]:
    """Detect silent spans with ffmpeg's `silencedetect` filter, returned as typed spans."""
    media = Path(wav_path)
    if not media.exists():
        raise AudioError(f"audio file does not exist: {media}")
    if min_silence_s <= 0:
        raise AudioError(f"min_silence_s must be > 0, got {min_silence_s}")

    duration = total_duration_s if total_duration_s is not None else probe_duration_s(media)

    command = [
        ffmpeg_path(), "-v", "info", "-nostdin",
        "-i", str(media),
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence_s}",
        "-f", "null", "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AudioError(f"silencedetect failed on {media}: {result.stderr.strip()}")

    silences = _parse_silencedetect(result.stderr, total_duration_s=duration)
    logger.info(
        "silence detection on %s at %.1f dB / %.2fs min: %d span(s)",
        media.name, threshold_db, min_silence_s, len(silences),
    )
    for span in silences:
        logger.debug("  silence %.3fs -> %.3fs (%.3fs)", span.start_s, span.end_s, span.duration_s)
    return silences


def _parse_silencedetect(stderr: str, *, total_duration_s: float) -> list[Silence]:
    """Parse silencedetect's stderr into ordered, clamped Silence spans."""
    spans: list[Silence] = []
    pending_start: float | None = None

    for line in stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            # A second silence_start without an intervening silence_end would mean ffmpeg
            # changed its output contract; drop the stale one rather than pairing wrongly.
            if pending_start is not None:
                logger.warning("silencedetect emitted two starts without an end; dropping %.3f",
                               pending_start)
            pending_start = max(0.0, float(start_match.group(1)))
            continue

        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            end = min(total_duration_s, float(end_match.group(1)))
            if end > pending_start:
                spans.append(Silence(pending_start, end))
            pending_start = None

    # A silence running to EOF is reported with a start but sometimes no end.
    if pending_start is not None and total_duration_s > pending_start:
        spans.append(Silence(pending_start, total_duration_s))

    return sorted(spans, key=lambda s: s.start_s)


# --------------------------------------------------------------------------------------
# Slicing
# --------------------------------------------------------------------------------------

def read_wav_params(wav_path: str | Path) -> dict[str, int]:
    """Return channels, sample width, frame rate, and frame count of a PCM WAV."""
    with wave.open(str(wav_path), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "frame_rate": handle.getframerate(),
            "frames": handle.getnframes(),
        }


def write_wav_slice(
    source_wav: str | Path,
    dest_wav: str | Path,
    start_s: float,
    end_s: float,
) -> float:
    """Copy [start_s, end_s) of a PCM WAV into a new WAV; returns the slice's real duration."""
    src, out = Path(source_wav), Path(dest_wav)
    if not src.exists():
        raise AudioError(f"source wav does not exist: {src}")
    if end_s <= start_s:
        raise AudioError(f"slice end ({end_s}) must be after start ({start_s})")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with wave.open(str(src), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            total_frames = reader.getnframes()

            # round(), not int(): a boundary at 20.9935s must land on the nearest sample,
            # not be truncated, or consecutive slices would drift apart by up to a sample
            # each and stop tiling the timeline exactly.
            first = min(max(0, round(start_s * rate)), total_frames)
            last = min(max(first, round(end_s * rate)), total_frames)
            if last == first:
                raise AudioError(
                    f"slice [{start_s:.3f}, {end_s:.3f}) of {src.name} is empty "
                    f"(file holds {total_frames / rate:.3f}s)"
                )
            reader.setpos(first)
            frames = reader.readframes(last - first)
    except wave.Error as exc:
        raise AudioError(f"{src} is not a readable PCM WAV: {exc}") from exc

    with wave.open(str(out), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)

    duration = (last - first) / float(rate)
    logger.debug("sliced %s [%.3f, %.3f) -> %s (%.3fs)", src.name, start_s, end_s, out.name, duration)
    return duration


def frames_to_seconds(frames: int, frame_rate: int) -> float:
    """Convert a frame count to seconds; the inverse of the rounding used when slicing."""
    if frame_rate <= 0:
        raise AudioError(f"frame_rate must be positive, got {frame_rate}")
    return frames / float(frame_rate)


def db_to_amplitude(db: float) -> float:
    """Convert a dBFS value to a linear amplitude ratio (0.0-1.0 for dBFS <= 0)."""
    return math.pow(10.0, db / 20.0)
