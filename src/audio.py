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

import array
import contextlib
import io
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


def amplitude_to_db(amplitude: float) -> float:
    """Convert a linear amplitude ratio to dBFS; silence maps to -inf."""
    if amplitude <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(amplitude)


# --------------------------------------------------------------------------------------
# In-memory WAV handling (M4: the TTS endpoint returns base64 WAV, never a file)
# --------------------------------------------------------------------------------------

#: Sample-value ranges by width, used to normalise a peak to dBFS.
_FULL_SCALE = {1: 128.0, 2: 32768.0, 4: 2147483648.0}
_ARRAY_TYPECODE = {1: "b", 2: "h", 4: "i"}


@dataclass(frozen=True)
class WavInfo:
    """Measured parameters of an in-memory PCM WAV."""

    channels: int
    sample_width: int
    frame_rate: int
    frames: int

    @property
    def duration_s(self) -> float:
        """Duration in seconds, computed from the frame count -- never from a reported field."""
        return self.frames / float(self.frame_rate) if self.frame_rate else 0.0


@dataclass(frozen=True)
class TrimmedAudio:
    """A WAV with leading/trailing silence removed, plus what was removed.

    `lead_trim_s` is what M5 must add back when placing this segment: the synthesiser's
    leading pad was part of the original timing, and dropping it silently shifts the
    segment earlier by that much.
    """

    data: bytes
    duration_s: float
    original_duration_s: float
    lead_trim_s: float
    trail_trim_s: float
    peak_dbfs: float
    #: True when no sample anywhere exceeded the threshold, so nothing was trimmed.
    all_silent: bool = False

    def to_dict(self) -> dict[str, float | bool]:
        """Serialise for the per-segment TTS record."""
        return {
            "duration_s": round(self.duration_s, 6),
            "untrimmed_duration_s": round(self.original_duration_s, 6),
            "lead_trim_s": round(self.lead_trim_s, 6),
            "trail_trim_s": round(self.trail_trim_s, 6),
            "peak_dbfs": round(self.peak_dbfs, 3) if math.isfinite(self.peak_dbfs) else None,
            "all_silent": self.all_silent,
        }


def read_wav_bytes(data: bytes) -> tuple[WavInfo, bytes]:
    """Parse an in-memory WAV; returns its measured parameters and the raw PCM frames."""
    if not data:
        raise AudioError("empty audio payload: the API returned no bytes")
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            info = WavInfo(
                channels=handle.getnchannels(),
                sample_width=handle.getsampwidth(),
                frame_rate=handle.getframerate(),
                frames=handle.getnframes(),
            )
            frames = handle.readframes(info.frames)
    except (wave.Error, EOFError) as exc:
        raise AudioError(
            f"audio payload is not a readable PCM WAV ({len(data)} bytes, "
            f"starts with {data[:4]!r}): {exc}"
        ) from exc

    if info.frame_rate <= 0:
        raise AudioError(f"WAV reports a frame rate of {info.frame_rate}")
    return info, frames


def wav_bytes_duration_s(data: bytes) -> float:
    """Measure a WAV's duration from its own frame count and frame rate."""
    info, _ = read_wav_bytes(data)
    return info.duration_s


def write_wav_bytes(dest: str | Path, data: bytes) -> Path:
    """Write WAV bytes to disk, creating parent directories; returns the path written."""
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def _frame_peaks(frames: bytes, info: WavInfo) -> array.array:
    """Return the per-frame peak absolute sample value, collapsing channels."""
    typecode = _ARRAY_TYPECODE.get(info.sample_width)
    if typecode is None:
        raise AudioError(
            f"unsupported sample width {info.sample_width} bytes; expected 1, 2, or 4"
        )
    samples = array.array(typecode)
    samples.frombytes(frames[:len(frames) - (len(frames) % info.sample_width)])

    if info.channels == 1:
        return array.array("d", (abs(s) for s in samples))

    peaks = array.array("d")
    step = info.channels
    for index in range(0, len(samples) - step + 1, step):
        peaks.append(max(abs(s) for s in samples[index:index + step]))
    return peaks


def trim_wav_silence(
    data: bytes,
    *,
    threshold_db: float = -40.0,
    window_ms: float = 10.0,
    keep_margin_ms: float = 20.0,
) -> TrimmedAudio:
    """Strip leading and trailing silence from an in-memory WAV, keeping what was removed.

    Why this exists: a synthesiser pads its output with a little silence at each end, and
    that padding does not scale with the pace parameter. Measuring an untrimmed clip
    therefore mixes a fixed offset into a quantity the duration-fit loop assumes is
    proportional to pace, which biases every correction it computes. Trimming first makes
    the measured duration a clean function of pace.

    `keep_margin_ms` of the removed silence is given back at each end so the attack of a
    plosive is never clipped.
    """
    info, frames = read_wav_bytes(data)
    if info.frames == 0:
        raise AudioError("audio payload contains zero frames")

    peaks = _frame_peaks(frames, info)
    full_scale = _FULL_SCALE.get(info.sample_width, 32768.0)
    peak_dbfs = amplitude_to_db((max(peaks) if peaks else 0.0) / full_scale)

    threshold = db_to_amplitude(threshold_db) * full_scale
    window = max(1, int(info.frame_rate * window_ms / 1000.0))

    first_loud: int | None = None
    last_loud: int | None = None
    for start in range(0, len(peaks), window):
        if max(peaks[start:start + window], default=0.0) >= threshold:
            if first_loud is None:
                first_loud = start
            last_loud = min(len(peaks), start + window)

    original_duration = info.duration_s

    if first_loud is None or last_loud is None:
        # Nothing above the threshold anywhere. Trimming would delete the segment, so
        # return it untouched and let the caller decide -- silently emitting zero frames
        # here would show up much later as a hole in the dubbed timeline.
        logger.warning(
            "TTS audio never exceeded %.1f dBFS (peak %.1f dBFS, %.3fs); leaving it untrimmed",
            threshold_db, peak_dbfs, original_duration,
        )
        return TrimmedAudio(
            data=data, duration_s=original_duration, original_duration_s=original_duration,
            lead_trim_s=0.0, trail_trim_s=0.0, peak_dbfs=peak_dbfs, all_silent=True,
        )

    margin = max(0, int(info.frame_rate * keep_margin_ms / 1000.0))
    start_frame = max(0, first_loud - margin)
    end_frame = min(info.frames, last_loud + margin)

    width = info.sample_width * info.channels
    trimmed_frames = frames[start_frame * width:end_frame * width]

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(info.channels)
        writer.setsampwidth(info.sample_width)
        writer.setframerate(info.frame_rate)
        writer.writeframes(trimmed_frames)

    trimmed = TrimmedAudio(
        data=buffer.getvalue(),
        duration_s=(end_frame - start_frame) / float(info.frame_rate),
        original_duration_s=original_duration,
        lead_trim_s=start_frame / float(info.frame_rate),
        trail_trim_s=(info.frames - end_frame) / float(info.frame_rate),
        peak_dbfs=peak_dbfs,
    )
    logger.debug(
        "trimmed %.3fs -> %.3fs (lead %.3fs, trail %.3fs, peak %.1f dBFS)",
        original_duration, trimmed.duration_s, trimmed.lead_trim_s,
        trimmed.trail_trim_s, peak_dbfs,
    )
    return trimmed


# --------------------------------------------------------------------------------------
# Timeline assembly (M5: place synthesised clips on one silent track)
# --------------------------------------------------------------------------------------

#: 16-bit PCM sample bounds. Mixing two clips that overlap can exceed these, and a wrapped
#: sample is an audible crack, so mixing clamps and counts instead of letting it wrap.
INT16_MIN = -32768
INT16_MAX = 32767


def silent_int16(frames: int, channels: int = 1) -> array.array:
    """Allocate `frames` of digital silence as an int16 sample array."""
    if frames < 0:
        raise AudioError(f"frame count must be >= 0, got {frames}")
    if channels < 1:
        raise AudioError(f"channel count must be >= 1, got {channels}")
    return array.array("h", bytes(frames * channels * 2))


def int16_from_wav_bytes(data: bytes) -> tuple[WavInfo, array.array]:
    """Parse an in-memory 16-bit PCM WAV into its parameters and an int16 sample array."""
    info, frames = read_wav_bytes(data)
    if info.sample_width != 2:
        raise AudioError(
            f"timeline assembly handles 16-bit PCM only, got {info.sample_width * 8}-bit. "
            f"Re-synthesise at 16-bit or convert before assembling."
        )
    samples = array.array("h")
    samples.frombytes(frames[:len(frames) - (len(frames) % 2)])
    return info, samples


def read_wav_int16(path: str | Path) -> tuple[WavInfo, array.array]:
    """Read a 16-bit PCM WAV from disk into its parameters and an int16 sample array."""
    media = Path(path)
    if not media.exists():
        raise AudioError(f"audio file does not exist: {media}")
    return int16_from_wav_bytes(media.read_bytes())


def apply_fade_int16(
    samples: array.array,
    *,
    channels: int,
    frame_rate: int,
    fade_ms: float,
) -> int:
    """Apply a linear fade in and out in place; returns the fade length actually used, in frames.

    A synthesised clip starts and ends at a non-zero sample, and dropping that onto a silent
    track puts a step discontinuity at each boundary -- an audible click. A short linear ramp
    removes it. The ramp is capped at half the clip so a very short segment still fades
    symmetrically instead of fading in past its own midpoint.
    """
    if fade_ms < 0:
        raise AudioError(f"fade_ms must be >= 0, got {fade_ms}")
    if channels < 1 or frame_rate <= 0:
        raise AudioError(f"invalid clip format: {channels} ch at {frame_rate} Hz")

    total_frames = len(samples) // channels
    fade_frames = min(int(frame_rate * fade_ms / 1000.0), total_frames // 2)
    if fade_frames <= 0:
        return 0

    for frame in range(fade_frames):
        gain_in = frame / fade_frames
        gain_out = gain_in
        head = frame * channels
        tail = (total_frames - 1 - frame) * channels
        for channel in range(channels):
            samples[head + channel] = int(samples[head + channel] * gain_in)
            samples[tail + channel] = int(samples[tail + channel] * gain_out)
    return fade_frames


def mix_int16(
    canvas: array.array,
    clip: array.array,
    *,
    offset_frames: int,
    channels: int = 1,
) -> dict[str, int]:
    """Add `clip` into `canvas` at `offset_frames`, clamping; returns what was written and clipped.

    Additive rather than overwriting: where two segments overlap, overwriting would delete
    the tail of the earlier line outright, while summing keeps both audible so the overlap
    is something a listener (and the QC report) can actually detect.
    """
    if offset_frames < 0:
        raise AudioError(f"offset must be >= 0 frames, got {offset_frames}")
    if channels < 1:
        raise AudioError(f"channel count must be >= 1, got {channels}")

    start = offset_frames * channels
    available = len(canvas) - start
    if available <= 0:
        return {"written": 0, "truncated": len(clip) // channels, "clipped": 0, "overlapped": 0}

    writable = min(len(clip), available)
    clipped = 0
    overlapped = 0

    for index in range(writable):
        position = start + index
        existing = canvas[position]
        if existing:
            overlapped += 1
        total = existing + clip[index]
        if total > INT16_MAX:
            total = INT16_MAX
            clipped += 1
        elif total < INT16_MIN:
            total = INT16_MIN
            clipped += 1
        canvas[position] = total

    return {
        "written": writable // channels,
        "truncated": (len(clip) - writable) // channels,
        "clipped": clipped,
        "overlapped": overlapped // channels,
    }


def write_int16_wav(
    dest: str | Path,
    samples: array.array,
    *,
    channels: int,
    frame_rate: int,
) -> Path:
    """Write an int16 sample array to a PCM WAV; returns the path written."""
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        writer.writeframes(samples.tobytes())
    return out


def seconds_to_frames(seconds: float, frame_rate: int) -> int:
    """Convert seconds to a frame index, rounding to the nearest sample (never truncating)."""
    if frame_rate <= 0:
        raise AudioError(f"frame_rate must be positive, got {frame_rate}")
    return max(0, round(seconds * frame_rate))
