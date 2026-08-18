"""
Assemble stage: place every synthesised clip on one silent timeline the length of the source.

Input:  output/tts_report.json (the per-segment clips M4 wrote, with their windows and the
        silence M4 trimmed off each end) and the demuxed source audio, for its true length.
Output: output/dubbed_audio.wav -- one continuous track, exactly as long as the source --
        plus output/assemble_report.json.

WHAT THIS STAGE IS FOR
M4 produced N separate clips with no shared clock. Sync is created here, and only here:
each clip is written at the timestamp its source segment started, and the gaps between
them stay silent. Nothing is time-stretched, and no clip is moved to make room for
another -- a clip that overruns its window is placed where it belongs and the overlap is
recorded, because hiding it by nudging clips later would turn one late line into a dub
that slides progressively out of sync.

FOUR DECISIONS

1. **Placement restores M4's lead trim.** M4 measured each clip with its silence trimmed,
   because the synthesiser's padding does not scale with `pace` and would have biased the
   fit loop. That padding was still part of the clip's own onset, so placing at
   `start + lead_trim_s` puts the speech exactly where the untrimmed clip would have put
   it. Trimming was a measurement device, not an edit.

2. **Overlaps are mixed, not truncated.** Where a clip runs into the next one, the samples
   are summed (with clamping) rather than one being cut off. An overlap that is audible is
   one QC can flag; an overlap silently truncated is a missing half-sentence nobody sees.

3. **A 15 ms linear fade on each end.** A clip starts and ends mid-waveform; dropping that
   onto silence puts a step discontinuity at the boundary, which is heard as a click.

4. **The track is the length of the source, not the length of the speech.** The mux then
   has an audio stream that matches the video frame for frame, and a dub that ends early
   cannot silently shorten the output.

Example placement (measured, samples/test_clip.mp4):
    {"segment_id": 1, "window": [5.475, 13.183], "placed_start_s": 5.595,
     "placed_end_s": 8.135, "fill_pct": 32.9, "overlap_with_next_s": 0.0}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..audio import (
    AudioError,
    apply_fade_int16,
    mix_int16,
    probe_duration_s,
    read_wav_int16,
    seconds_to_frames,
    silent_int16,
    write_int16_wav,
)
from ..config import DUBBED_AUDIO_NAME, Config
from .demux import DEMUXED_WAV_NAME
from .tts import REPORT_NAME as TTS_REPORT_NAME

logger = logging.getLogger(__name__)

REPORT_NAME = "assemble_report.json"

#: Timeline arithmetic is sample-accurate; anything above this is a real bug, not rounding.
TIME_EPSILON_S = 1e-3


class AssembleError(RuntimeError):
    """Raised when the dubbed timeline cannot be built safely."""


@dataclass
class Placement:
    """One synthesised clip positioned on the dubbed timeline."""

    segment_id: int
    window_start_s: float
    window_end_s: float
    placed_start_s: float
    duration_s: float
    lead_trim_s: float
    source_path: Path
    fitted: bool
    fade_frames: int
    clipped_samples: int
    overlapped_frames: int
    truncated_frames: int

    @property
    def placed_end_s(self) -> float:
        """Where this clip stops speaking on the dubbed timeline."""
        return self.placed_start_s + self.duration_s

    @property
    def window_duration_s(self) -> float:
        """The time budget this clip was given by the source segment."""
        return self.window_end_s - self.window_start_s

    @property
    def fill_pct(self) -> float:
        """How much of its window the clip actually fills; under 100% leaves silence."""
        if self.window_duration_s <= 0:
            return 0.0
        return self.duration_s / self.window_duration_s * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the assemble report."""
        return {
            "segment_id": self.segment_id,
            "window": [round(self.window_start_s, 6), round(self.window_end_s, 6)],
            "window_duration_s": round(self.window_duration_s, 6),
            "placed_start_s": round(self.placed_start_s, 6),
            "placed_end_s": round(self.placed_end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "lead_trim_restored_s": round(self.lead_trim_s, 6),
            "fill_pct": round(self.fill_pct, 4),
            "source_path": str(self.source_path),
            "clip_variant": "fitted" if self.fitted else "baseline_nofit",
            "fade_frames": self.fade_frames,
            "clipped_samples": self.clipped_samples,
            "overlapped_frames": self.overlapped_frames,
            "truncated_frames": self.truncated_frames,
        }


def detect_overlaps(placements: Sequence[Placement]) -> list[dict[str, Any]]:
    """Find every pair of clips whose placed audio collides on the timeline."""
    overlaps: list[dict[str, Any]] = []
    ordered = sorted(placements, key=lambda p: p.placed_start_s)

    for left, right in zip(ordered, ordered[1:]):
        overlap = left.placed_end_s - right.placed_start_s
        if overlap > TIME_EPSILON_S:
            overlaps.append({
                "segment_ids": [left.segment_id, right.segment_id],
                "overlap_s": round(overlap, 6),
                "overlap_pct_of_later_clip": round(
                    overlap / right.duration_s * 100.0, 4
                ) if right.duration_s > 0 else None,
                "left_placed_end_s": round(left.placed_end_s, 6),
                "right_placed_start_s": round(right.placed_start_s, 6),
            })
    return overlaps


def run_assemble(
    config: Config,
    *,
    tts_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one continuous dubbed WAV from the M4 clips; writes it and assemble_report.json."""
    report_path = Path(tts_report_path) if tts_report_path else config.output_dir / TTS_REPORT_NAME
    if not report_path.exists():
        raise AssembleError(
            f"{report_path} not found. Run the TTS stage first "
            f"(python -m src.pipeline --input <video> --stage tts)."
        )

    tts_report = json.loads(report_path.read_text(encoding="utf-8"))
    files = tts_report.get("files") or []
    if not files:
        raise AssembleError(f"{report_path} lists no synthesised clips to assemble")

    # --- how long the track must be ----------------------------------------------------
    timeline_duration_s, duration_source = _resolve_timeline_duration(config)
    logger.info("assemble: timeline is %.3fs (from %s)", timeline_duration_s, duration_source)

    # --- work out the canvas format from the clips themselves ---------------------------
    use_fitted = config.enable_duration_fit
    clips: list[tuple[dict[str, Any], Path]] = []
    for entry in sorted(files, key=lambda f: float(f["start"])):
        key = "fitted_path" if use_fitted else "baseline_path"
        clip_path = Path(entry[key])
        if not clip_path.exists():
            raise AssembleError(
                f"segment {entry['segment_id']} names {clip_path}, which does not exist. "
                f"Re-run --stage tts to regenerate the clips."
            )
        clips.append((entry, clip_path))

    frame_rate, channels = _assert_uniform_format(clips)
    logger.info(
        "assemble: %d %s clip(s) at %d Hz / %d ch, %.0f ms fades",
        len(clips), "fitted" if use_fitted else "UNFITTED baseline (--no-fit)",
        frame_rate, channels, config.fade_ms,
    )

    total_frames = seconds_to_frames(timeline_duration_s, frame_rate)
    if total_frames <= 0:
        raise AssembleError(
            f"the source is {timeline_duration_s:.3f}s, which is not long enough to hold a dub"
        )

    # Room for a clip that overruns the source end, so a late final line is mixed in and
    # reported rather than silently cut. The track is trimmed back to the source length
    # afterwards only if nothing needed the extra room.
    last_needed_s = max(
        float(e["start"]) + float(e.get("lead_trim_s") or 0.0) + probe_duration_s(p)
        for e, p in clips
    )
    canvas_frames = max(total_frames, seconds_to_frames(last_needed_s, frame_rate))
    canvas = silent_int16(canvas_frames, channels)

    # --- place every clip ---------------------------------------------------------------
    placements: list[Placement] = []
    for entry, clip_path in clips:
        info, samples = read_wav_int16(clip_path)
        # Restoring the lead trim puts the speech exactly where the untrimmed clip would
        # have started; M4 removed it only so the fit loop measured speech, not padding.
        lead_trim_s = float(entry.get("lead_trim_s") or 0.0)
        placed_start_s = float(entry["start"]) + lead_trim_s

        fade_frames = apply_fade_int16(
            samples, channels=channels, frame_rate=frame_rate, fade_ms=config.fade_ms,
        )
        mixed = mix_int16(
            canvas, samples,
            offset_frames=seconds_to_frames(placed_start_s, frame_rate),
            channels=channels,
        )

        placement = Placement(
            segment_id=int(entry["segment_id"]),
            window_start_s=float(entry["start"]),
            window_end_s=float(entry["end"]),
            placed_start_s=placed_start_s,
            duration_s=info.duration_s,
            lead_trim_s=lead_trim_s,
            source_path=clip_path,
            fitted=use_fitted,
            fade_frames=fade_frames,
            clipped_samples=mixed["clipped"],
            overlapped_frames=mixed["overlapped"],
            truncated_frames=mixed["truncated"],
        )
        placements.append(placement)

        logger.info(
            "  segment %d: %.3fs -> %.3fs (%.3fs of a %.3fs window, %.0f%% filled)",
            placement.segment_id, placement.placed_start_s, placement.placed_end_s,
            placement.duration_s, placement.window_duration_s, placement.fill_pct,
        )
        if mixed["truncated"]:
            logger.warning(
                "  segment %d: %d frame(s) fell past the end of the timeline and were dropped",
                placement.segment_id, mixed["truncated"],
            )
        if mixed["clipped"]:
            logger.warning(
                "  segment %d: %d sample(s) clamped while mixing an overlap",
                placement.segment_id, mixed["clipped"],
            )

    # --- verify -------------------------------------------------------------------------
    overlaps = detect_overlaps(placements)
    for overlap in overlaps:
        logger.warning(
            "overlap: segment %d runs %.3fs into segment %d (ends %.3fs, next starts %.3fs); "
            "both are audible and this is a QC finding, not a silent truncation",
            overlap["segment_ids"][0], overlap["overlap_s"], overlap["segment_ids"][1],
            overlap["left_placed_end_s"], overlap["right_placed_start_s"],
        )

    overflow_s = max(0.0, max(p.placed_end_s for p in placements) - timeline_duration_s)
    if overflow_s > TIME_EPSILON_S:
        logger.warning(
            "the last clip ends %.3fs past the %.3fs source; the dubbed track is that much "
            "longer and the mux will report the mismatch",
            overflow_s, timeline_duration_s,
        )
    else:
        canvas = canvas[:total_frames * channels]

    # --- write --------------------------------------------------------------------------
    out_path = config.output_dir / DUBBED_AUDIO_NAME
    write_int16_wav(out_path, canvas, channels=channels, frame_rate=frame_rate)

    written_duration_s = (len(canvas) // channels) / float(frame_rate)
    speech_s = sum(p.duration_s for p in placements)

    report = {
        "dubbed_audio_path": str(out_path),
        "tts_report_path": str(report_path),
        "timeline": {
            "source_duration_s": round(timeline_duration_s, 6),
            "source_duration_from": duration_source,
            "written_duration_s": round(written_duration_s, 6),
            "overflow_past_source_s": round(overflow_s, 6),
            "frame_rate": frame_rate,
            "channels": channels,
            "sample_width_bytes": 2,
        },
        "placement": {
            "clip_variant": "fitted" if use_fitted else "baseline_nofit",
            "clips_placed": len(placements),
            "fade_ms": config.fade_ms,
            "lead_trim_restored": True,
            "speech_s": round(speech_s, 6),
            "silence_s": round(max(0.0, written_duration_s - speech_s), 6),
            "speech_pct_of_timeline": round(speech_s / written_duration_s * 100.0, 4)
            if written_duration_s > 0 else None,
            "clipped_samples_total": sum(p.clipped_samples for p in placements),
            "frames_dropped_past_end": sum(p.truncated_frames for p in placements),
        },
        "overlaps": {
            "count": len(overlaps),
            "total_overlap_s": round(sum(o["overlap_s"] for o in overlaps), 6),
            "detail": overlaps,
        },
        "placements": [p.to_dict() for p in placements],
    }

    assemble_report_path = config.output_dir / REPORT_NAME
    assemble_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    logger.info(
        "assemble: wrote %s (%.3fs, %d clip(s), %.1f%% speech, %d overlap(s)) and %s",
        out_path, written_duration_s, len(placements),
        report["placement"]["speech_pct_of_timeline"] or 0.0, len(overlaps),
        assemble_report_path,
    )
    return report


def _resolve_timeline_duration(config: Config) -> tuple[float, str]:
    """Return the length the dubbed track must have, and where that length was measured."""
    demuxed = config.output_dir / DEMUXED_WAV_NAME
    if demuxed.exists():
        # The demuxed WAV is what every timestamp downstream is anchored to (see demux.py),
        # so it -- not the container -- defines the timeline.
        return probe_duration_s(demuxed), str(demuxed)
    if config.input_path is not None and config.input_path.exists():
        return probe_duration_s(config.input_path), str(config.input_path)
    raise AssembleError(
        f"cannot determine the timeline length: neither {demuxed} nor the input file exists. "
        f"Run --stage demux first."
    )


def _assert_uniform_format(clips: Sequence[tuple[dict[str, Any], Path]]) -> tuple[int, int]:
    """Assert every clip shares one sample rate and channel count; returns (rate, channels)."""
    formats: dict[tuple[int, int], list[int]] = {}
    for entry, path in clips:
        try:
            info, _ = read_wav_int16(path)
        except AudioError as exc:
            raise AssembleError(f"segment {entry['segment_id']} ({path}): {exc}") from exc
        formats.setdefault((info.frame_rate, info.channels), []).append(int(entry["segment_id"]))

    if len(formats) > 1:
        detail = "; ".join(
            f"{rate} Hz/{channels} ch: segments {ids}" for (rate, channels), ids in formats.items()
        )
        raise AssembleError(
            f"the clips do not share one audio format, so they cannot be mixed onto one "
            f"timeline without resampling: {detail}. Re-run --stage tts with a single "
            f"--tts-sample-rate."
        )
    return next(iter(formats))
