"""
Mux stage: remux the original video stream with the dubbed audio track.

Input:  config.input_path (the untouched source) and output/dubbed_audio.wav from M5.
Output: config.output_path (default output/dubbed.mp4) + output/mux_report.json.

THE ONE RULE HERE: `-c:v copy`. The video stream is copied through bit for bit, never
re-encoded, so the picture in the dub is the same picture as the source -- no generation
loss, and a fraction of the wall clock a re-encode would cost. Only the audio is encoded,
because PCM cannot live in an MP4; that is AAC at 192 kbit/s.

v1 REPLACES the audio track entirely. Background music and effects in the source are lost
with the original speech -- preserving them needs source separation (Demucs), which SPEC.md
lists as a deliberate non-goal for v1. This is stated in the report rather than left for a
listener to discover.

The output is probed after writing, and the checks are assertions rather than logs: that
the video codec, resolution, and frame count survived unchanged, and that the audio track
is the dubbed one and the right length. A remux that silently re-encoded or truncated
would otherwise look like a success.

Example (measured, samples/test_clip.mp4):
    h264 1280x720 copied unchanged; audio aac 24000 Hz mono; 36.107 s -> 36.107 s
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from ..audio import AudioError, ffmpeg_path, ffprobe_path, probe_duration_s
from ..config import (
    DUBBED_AUDIO_NAME,
    MUX_AUDIO_CODEC,
    MUX_DURATION_TOLERANCE_S,
    Config,
)

logger = logging.getLogger(__name__)

REPORT_NAME = "mux_report.json"


class MuxError(RuntimeError):
    """Raised when the dubbed video cannot be produced or fails its post-mux checks."""


def probe_video_stream(path: str | Path) -> dict[str, Any] | None:
    """Return the first video stream's codec, size, and frame count; None if there is none."""
    result = subprocess.run(
        [
            ffprobe_path(), "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt,duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise MuxError(f"ffprobe failed on {path}: {result.stderr.strip()}")

    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        return None

    stream = streams[0]
    return {
        "codec_name": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pix_fmt": stream.get("pix_fmt"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "nb_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "duration_s": float(stream["duration"]) if stream.get("duration") else None,
    }


def probe_audio_stream(path: str | Path) -> dict[str, Any] | None:
    """Return the first audio stream's codec, sample rate, and channel count; None if absent."""
    result = subprocess.run(
        [
            ffprobe_path(), "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise MuxError(f"ffprobe failed on {path}: {result.stderr.strip()}")

    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        return None

    stream = streams[0]
    return {
        "codec_name": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": stream.get("channels"),
        "duration_s": float(stream["duration"]) if stream.get("duration") else None,
    }


def run_mux(
    config: Config,
    *,
    dubbed_audio_path: str | Path | None = None,
) -> dict[str, Any]:
    """Combine the source video stream with the dubbed audio; writes the output and its report."""
    if config.input_path is None or not config.input_path.exists():
        raise MuxError("mux requires --input, pointing at the original media")

    audio_path = Path(dubbed_audio_path) if dubbed_audio_path else config.output_dir / DUBBED_AUDIO_NAME
    if not audio_path.exists():
        raise MuxError(
            f"{audio_path} not found. Run the assemble stage first "
            f"(python -m src.pipeline --input <video> --stage assemble)."
        )

    source_video = probe_video_stream(config.input_path)
    source_audio = probe_audio_stream(config.input_path)
    source_duration_s = probe_duration_s(config.input_path)
    dub_duration_s = probe_duration_s(audio_path)

    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if source_video is None:
        # An audio-only input (a bare WAV, say) has no picture to preserve. Producing an
        # audio-only output is still a useful artefact, so this degrades with a WARNING
        # rather than failing the run outright.
        logger.warning(
            "%s has no video stream; writing an audio-only output to %s",
            config.input_path, out_path,
        )
        command = [
            ffmpeg_path(), "-v", "error", "-nostdin", "-y",
            "-i", str(audio_path),
            "-c:a", MUX_AUDIO_CODEC, "-b:a", config.audio_bitrate,
            str(out_path),
        ]
    else:
        command = [
            ffmpeg_path(), "-v", "error", "-nostdin", "-y",
            "-i", str(config.input_path),
            "-i", str(audio_path),
            "-map", "0:v:0",        # the picture, from the source
            "-map", "1:a:0",        # the audio, from the dub -- the source track is dropped
            "-c:v", "copy",         # never re-encode the video
            "-c:a", MUX_AUDIO_CODEC,
            "-b:a", config.audio_bitrate,
            str(out_path),
        ]

    logger.info("mux: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise MuxError(
            f"ffmpeg failed muxing {config.input_path} + {audio_path} -> {out_path}: "
            f"{result.stderr.strip()}"
        )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise MuxError(f"ffmpeg reported success but {out_path} is empty")

    output_video = probe_video_stream(out_path)
    output_audio = probe_audio_stream(out_path)
    output_duration_s = probe_duration_s(out_path)

    checks = _verify(
        source_video=source_video,
        output_video=output_video,
        output_audio=output_audio,
        source_duration_s=source_duration_s,
        output_duration_s=output_duration_s,
    )

    report = {
        "input_path": str(config.input_path),
        "dubbed_audio_path": str(audio_path),
        "output_path": str(out_path),
        "ffmpeg_command": command,
        "source": {
            "duration_s": round(source_duration_s, 6),
            "video": source_video,
            "audio": source_audio,
        },
        "dubbed_audio": {"duration_s": round(dub_duration_s, 6)},
        "output": {
            "duration_s": round(output_duration_s, 6),
            "video": output_video,
            "audio": output_audio,
            "size_bytes": out_path.stat().st_size,
        },
        "checks": checks,
        "audio_policy": (
            "v1 replaces the source audio track entirely; background music and effects are "
            "not preserved. Source separation (Demucs) is a documented non-goal for v1."
        ),
        "video_policy": "video copied with -c:v copy: no re-encode, no generation loss",
    }

    report_path = config.output_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    failed = [name for name, check in checks.items() if check["passed"] is False]
    if failed:
        raise MuxError(
            f"the muxed output failed {len(failed)} post-mux check(s): {failed}. "
            f"Detail in {report_path}."
        )

    logger.info(
        "mux: wrote %s (%.3fs, %s video + %s audio, %.1f MB) and %s",
        out_path, output_duration_s,
        (output_video or {}).get("codec_name", "none"),
        (output_audio or {}).get("codec_name", "none"),
        out_path.stat().st_size / 1e6, report_path,
    )
    return report


def _verify(
    *,
    source_video: dict[str, Any] | None,
    output_video: dict[str, Any] | None,
    output_audio: dict[str, Any] | None,
    source_duration_s: float,
    output_duration_s: float,
) -> dict[str, dict[str, Any]]:
    """Check that the video survived untouched and the dub is present and the right length."""
    checks: dict[str, dict[str, Any]] = {}

    if source_video is not None:
        for field in ("codec_name", "width", "height", "pix_fmt"):
            source_value = source_video.get(field)
            output_value = (output_video or {}).get(field)
            checks[f"video_{field}_unchanged"] = {
                "passed": source_value == output_value,
                "expected": source_value,
                "actual": output_value,
                "why": "-c:v copy must pass the video through bit for bit",
            }

        source_frames = source_video.get("nb_frames")
        output_frames = (output_video or {}).get("nb_frames")
        if source_frames and output_frames:
            checks["video_frame_count_unchanged"] = {
                "passed": source_frames == output_frames,
                "expected": source_frames,
                "actual": output_frames,
                "why": "a dropped frame means the copy truncated the picture",
            }

    checks["audio_track_present"] = {
        "passed": output_audio is not None,
        "expected": "one audio stream",
        "actual": output_audio,
        "why": "a video with no audio track is a silent 'dub'",
    }

    drift = abs(output_duration_s - source_duration_s)
    checks["duration_matches_source"] = {
        "passed": drift <= MUX_DURATION_TOLERANCE_S,
        "expected": round(source_duration_s, 6),
        "actual": round(output_duration_s, 6),
        "drift_s": round(drift, 6),
        "tolerance_s": MUX_DURATION_TOLERANCE_S,
        "why": (
            "the dub must be as long as the source; container rounding and AAC priming "
            "account for a few tens of milliseconds, more means a stream was truncated"
        ),
    }

    for name, check in checks.items():
        logger.info(
            "  mux check %-32s %s (expected %s, got %s)",
            name, "PASS" if check["passed"] else "FAIL", check["expected"], check["actual"],
        )
    return checks


def assert_ffmpeg_available() -> None:
    """Fail early, with an actionable message, if ffmpeg or ffprobe is missing."""
    try:
        ffmpeg_path()
        ffprobe_path()
    except AudioError as exc:
        raise MuxError(str(exc)) from exc
