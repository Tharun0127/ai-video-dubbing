"""
ASR stage: transcribe audio with Sarvam Saaras v3 under the 30-second REST cap.

The synchronous /speech-to-text endpoint rejects audio longer than 30 s (HTTP 422), so a
36 s clip cannot be sent in one call. This module implements the chunked-REST strategy
from SPEC.md option (a):

    demuxed 16 kHz WAV
      -> detect silences (ffmpeg silencedetect, threshold tunable, default -40 dB)
      -> plan sub-30 s chunks whose boundaries sit at silence midpoints, never mid-word
      -> one Saaras call per chunk (cached, so a re-run costs Rs 0)
      -> stitch: shift each chunk's local timestamps by that chunk's global offset
      -> post-process: merge segments < 1 s, split segments > 15 s at a sentence boundary
      -> output/segments.json as [{id, start, end, text, speaker?}]

Two design decisions worth stating plainly, because they shape everything downstream:

1. **Segments tile the timeline.** Every segment's `end` is exactly the next segment's
   `start`, and the set covers [0, duration] with no gap and no overlap. Cut points sit at
   the *midpoint* of a detected silence, so the pause preceding a phrase is charged to the
   segment that will speak it -- which is also the time budget the M4 duration-fit loop
   gets to work with.

2. **Boundaries come from local silence detection, not from the API.** The sync endpoint's
   `with_timestamps` returns a single span covering the whole chunk (confirmed by real
   call, see docs/api-notes.md), so it carries no usable intra-chunk timing. The stitching
   maths here is nonetheless written for the general multi-span case and unit-tested that
   way, so richer timestamps (or the Batch API) drop in without a rewrite.

Input:  a 16 kHz mono PCM WAV from the demux stage.
Output: output/segments.json, output/asr_report.json, and output/chunks/chunk_NNN.wav.

Example segment (measured, samples/test_clip.mp4):
    {"id": 0, "start": 0.0, "end": 10.674, "text": "Almighty God, ...", "speaker": null}
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..audio import Silence, detect_silences, probe_duration_s, write_wav_slice
from ..config import ASR_MAX_AUDIO_S, Config
from ..metrics import MetricsCollector
from ..sarvam_client import SarvamClient
from .demux import DEMUXED_WAV_NAME

logger = logging.getLogger(__name__)

#: Timestamps are compared at millisecond scale; WAV slicing is sample-accurate (62.5 us
#: at 16 kHz), so anything above this tolerance is a real bug, not rounding.
TIME_EPSILON_S = 1e-3

#: End-of-sentence punctuation, including the Devanagari danda for the M6 back-transcript.
_SENTENCE_END_RE = re.compile(r"[.!?।]+[\"')\]”’]*\s+")
_CLAUSE_END_RE = re.compile(r"[,;:—-]+\s+")
_WORD_END_RE = re.compile(r"\s+")

#: Preference order when choosing where to split an over-long segment.
_SPLIT_KIND_RANK = {"sentence": 0, "clause": 1, "word": 2}


class AsrError(RuntimeError):
    """Raised when chunk planning, transcription, or stitching cannot proceed safely."""


class StitchError(AsrError):
    """Raised when chunk timelines are inconsistent -- a silent sync bug if left unchecked."""


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    """One sub-30 s slice of the source audio, positioned on the global timeline."""

    index: int
    start_s: float
    end_s: float
    #: How this chunk's *end* boundary was chosen: a detected silence, the hard 30 s cap,
    #: or the end of the file. "hard" is the only one that risks cutting mid-word.
    cut_source: str = "eof"
    path: Path | None = None

    @property
    def duration_s(self) -> float:
        """Length of the chunk in seconds."""
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the chunk report."""
        return {
            "index": self.index,
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "cut_source": self.cut_source,
            "path": str(self.path) if self.path else None,
        }


@dataclass(frozen=True)
class LocalSpan:
    """A transcribed span of speech, in seconds relative to whichever timeline it is on."""

    start_s: float
    end_s: float
    text: str

    @property
    def duration_s(self) -> float:
        """Length of the span in seconds."""
        return self.end_s - self.start_s


@dataclass
class Segment:
    """One normalised output segment; `to_spec_dict` emits the SPEC.md contract exactly."""

    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    #: Provenance, written to asr_report.json rather than segments.json.
    chunk_index: int | None = None
    boundary_source: str = "silence"
    origin: list[int] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        """Length of the segment in seconds."""
        return self.end - self.start

    def to_spec_dict(self) -> dict[str, Any]:
        """The exact shape SPEC.md defines for segments.json."""
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speaker": self.speaker,
        }

    def to_debug_dict(self) -> dict[str, Any]:
        """The full record, including how the boundaries were derived."""
        return {
            **self.to_spec_dict(),
            "duration_s": round(self.duration_s, 3),
            "chunk_index": self.chunk_index,
            "boundary_source": self.boundary_source,
            "origin_chunks": sorted(set(self.origin)) or ([self.chunk_index]
                                                          if self.chunk_index is not None else []),
        }


@dataclass(frozen=True)
class ChunkPlan:
    """The accepted chunk layout plus the detection settings that produced it."""

    chunks: list[Chunk]
    silences: list[Silence]
    threshold_db: float
    min_silence_s: float
    attempts: list[dict[str, Any]]

    @property
    def hard_cuts(self) -> int:
        """Number of boundaries placed at the 30 s cap rather than in a silence."""
        return sum(1 for c in self.chunks if c.cut_source == "hard")

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the chunk report."""
        return {
            "chunk_count": len(self.chunks),
            "threshold_db": self.threshold_db,
            "min_silence_s": self.min_silence_s,
            "hard_cuts": self.hard_cuts,
            "detection_attempts": self.attempts,
            "silences": [s.to_dict() for s in self.silences],
            "chunks": [c.to_dict() for c in self.chunks],
        }


# --------------------------------------------------------------------------------------
# 1. Chunk planning
# --------------------------------------------------------------------------------------

def plan_chunks(
    total_duration_s: float,
    silences: Sequence[Silence],
    *,
    max_chunk_s: float = ASR_MAX_AUDIO_S,
    min_chunk_s: float = 1.0,
) -> list[Chunk]:
    """Lay out contiguous sub-max_chunk_s chunks, cutting at silence midpoints where possible."""
    if total_duration_s <= 0:
        raise AsrError(f"total_duration_s must be positive, got {total_duration_s}")
    if max_chunk_s <= 0:
        raise AsrError(f"max_chunk_s must be positive, got {max_chunk_s}")

    if total_duration_s <= max_chunk_s:
        return [Chunk(index=0, start_s=0.0, end_s=total_duration_s, cut_source="eof")]

    # Cut at the centre of a silence: that is the point furthest from speech on either
    # side, so neither chunk loses a leading or trailing phoneme.
    cuts = sorted(
        s.midpoint_s for s in silences
        if 0.0 < s.midpoint_s < total_duration_s
    )

    boundaries: list[tuple[float, str]] = []
    cursor = 0.0

    while total_duration_s - cursor > max_chunk_s:
        remaining = total_duration_s - cursor
        # Chunks still needed to cover the tail. Aiming at an equal split of what is left
        # (rather than greedily filling to 30 s) avoids a 30 s chunk followed by a 6 s
        # runt, and keeps every request comfortably inside the cap.
        chunks_left = math.ceil(remaining / max_chunk_s)
        target = cursor + remaining / chunks_left

        lowest = cursor + min_chunk_s
        # Never cut so early that the tail can no longer be covered by the chunks left.
        lowest = max(lowest, total_duration_s - max_chunk_s * (chunks_left - 1))
        highest = cursor + max_chunk_s

        usable = [c for c in cuts if lowest <= c <= highest]
        if usable:
            chosen = min(usable, key=lambda c: (abs(c - target), c))
            source = "silence"
        else:
            # No silence anywhere in the admissible window: the speaker never paused.
            # Cutting at the cap is the only option and it may land mid-word, so it is a
            # WARNING, recorded on the chunk and surfaced in the report.
            chosen = highest
            source = "hard"
            logger.warning(
                "no silence between %.3fs and %.3fs; cutting at the %.0fs cap (%.3fs) -- "
                "this boundary may fall mid-word",
                lowest, highest, max_chunk_s, chosen,
            )
        boundaries.append((chosen, source))
        cursor = chosen

    boundaries.append((total_duration_s, "eof"))

    chunks: list[Chunk] = []
    start = 0.0
    for index, (end, source) in enumerate(boundaries):
        chunks.append(Chunk(index=index, start_s=start, end_s=end, cut_source=source))
        start = end

    _assert_contiguous(chunks, total_duration_s)
    return chunks


def plan_chunks_adaptive(
    wav_path: str | Path,
    total_duration_s: float,
    *,
    max_chunk_s: float = ASR_MAX_AUDIO_S,
    threshold_db: float = -40.0,
    min_silence_s: float = 0.30,
    min_chunk_s: float = 1.0,
) -> ChunkPlan:
    """Plan chunks, relaxing the silence threshold only if the default finds no safe cut."""
    # A clip whose noise floor sits above the threshold yields no interior silence at all
    # (samples/test_clip.mp4 has a mean volume of -26 dB, so -40 dB finds nothing but the
    # trailing pause). Rather than silently hard-cutting mid-word, escalate through
    # progressively more permissive settings and record which one was accepted.
    ladder: list[tuple[float, float]] = [
        (threshold_db, min_silence_s),
        (threshold_db + 5.0, min_silence_s),
        (threshold_db + 10.0, min_silence_s),
        (threshold_db + 10.0, max(0.10, min_silence_s * 0.5)),
        (threshold_db + 15.0, max(0.10, min_silence_s * 0.5)),
    ]

    attempts: list[dict[str, Any]] = []
    best: ChunkPlan | None = None

    for level, (level_db, level_min_silence) in enumerate(ladder):
        silences = detect_silences(
            wav_path,
            threshold_db=level_db,
            min_silence_s=level_min_silence,
            total_duration_s=total_duration_s,
        )
        chunks = plan_chunks(
            total_duration_s, silences, max_chunk_s=max_chunk_s, min_chunk_s=min_chunk_s,
        )
        plan = ChunkPlan(
            chunks=chunks, silences=silences,
            threshold_db=level_db, min_silence_s=level_min_silence, attempts=[],
        )
        attempts.append({
            "level": level,
            "threshold_db": level_db,
            "min_silence_s": round(level_min_silence, 4),
            "silences_found": len(silences),
            "chunks": len(chunks),
            "hard_cuts": plan.hard_cuts,
            "accepted": plan.hard_cuts == 0,
        })

        if best is None or plan.hard_cuts < best.hard_cuts:
            best = plan
        if plan.hard_cuts == 0:
            if level > 0:
                logger.warning(
                    "silence threshold relaxed from %.1f dB to %.1f dB (min %.2fs): the default "
                    "found no interior silence to cut on in this audio",
                    threshold_db, level_db, level_min_silence,
                )
            return ChunkPlan(
                chunks=plan.chunks, silences=plan.silences,
                threshold_db=level_db, min_silence_s=level_min_silence, attempts=attempts,
            )

    assert best is not None  # the ladder is never empty
    logger.warning(
        "no silence threshold avoided a hard cut; keeping %.1f dB with %d hard boundary/ies",
        best.threshold_db, best.hard_cuts,
    )
    return ChunkPlan(
        chunks=best.chunks, silences=best.silences,
        threshold_db=best.threshold_db, min_silence_s=best.min_silence_s, attempts=attempts,
    )


def _assert_contiguous(chunks: Sequence[Chunk], total_duration_s: float) -> None:
    """Fail loudly if chunks do not tile [0, total] exactly -- a gap here silently drops audio."""
    if not chunks:
        raise StitchError("chunk plan is empty")
    if abs(chunks[0].start_s) > TIME_EPSILON_S:
        raise StitchError(f"first chunk starts at {chunks[0].start_s:.6f}s, expected 0.0")
    if abs(chunks[-1].end_s - total_duration_s) > TIME_EPSILON_S:
        raise StitchError(
            f"last chunk ends at {chunks[-1].end_s:.6f}s but the audio is "
            f"{total_duration_s:.6f}s -- {total_duration_s - chunks[-1].end_s:.6f}s would be lost"
        )
    for left, right in zip(chunks, chunks[1:]):
        if abs(left.end_s - right.start_s) > TIME_EPSILON_S:
            gap = right.start_s - left.end_s
            raise StitchError(
                f"chunk {left.index} ends at {left.end_s:.6f}s but chunk {right.index} starts at "
                f"{right.start_s:.6f}s ({'gap' if gap > 0 else 'overlap'} of {abs(gap):.6f}s)"
            )
        if right.duration_s <= 0:
            raise StitchError(f"chunk {right.index} has non-positive duration")


# --------------------------------------------------------------------------------------
# 2. Timestamp offset correction -- the stitching maths
# --------------------------------------------------------------------------------------

def apply_offset(
    spans: Sequence[LocalSpan],
    offset_s: float,
    *,
    chunk_duration_s: float,
    tolerance_s: float = TIME_EPSILON_S,
) -> list[LocalSpan]:
    """Shift chunk-local span timestamps onto the global timeline by that chunk's start offset."""
    if offset_s < 0:
        raise StitchError(f"chunk offset must be >= 0, got {offset_s}")
    if chunk_duration_s <= 0:
        raise StitchError(f"chunk duration must be > 0, got {chunk_duration_s}")

    shifted: list[LocalSpan] = []
    previous_end = 0.0

    for position, span in enumerate(spans):
        if span.end_s < span.start_s - tolerance_s:
            raise StitchError(
                f"span {position} ends ({span.end_s:.6f}s) before it starts ({span.start_s:.6f}s)"
            )
        if span.start_s < -tolerance_s:
            raise StitchError(
                f"span {position} has a negative chunk-local start ({span.start_s:.6f}s); "
                f"timestamps from the API are relative to the chunk, not the file"
            )
        if span.start_s < previous_end - tolerance_s:
            raise StitchError(
                f"span {position} starts at {span.start_s:.6f}s, before span {position - 1} "
                f"ended at {previous_end:.6f}s -- spans must be ordered and non-overlapping"
            )
        if span.end_s > chunk_duration_s + tolerance_s:
            raise StitchError(
                f"span {position} ends at {span.end_s:.6f}s but its chunk is only "
                f"{chunk_duration_s:.6f}s long; the API returned a timestamp outside the audio"
            )

        # Clamp before shifting: a span sitting a few microseconds past the chunk end
        # (float noise) must not push the global timeline past the next chunk's start.
        local_start = min(max(0.0, span.start_s), chunk_duration_s)
        local_end = min(max(local_start, span.end_s), chunk_duration_s)
        previous_end = local_end

        shifted.append(LocalSpan(
            start_s=offset_s + local_start,
            end_s=offset_s + local_end,
            text=span.text,
        ))

    return shifted


def local_spans_from_payload(
    payload: dict[str, Any],
    chunk_duration_s: float,
) -> tuple[list[LocalSpan], str]:
    """Read chunk-local spans out of a Saaras response; returns (spans, granularity label)."""
    transcript = (payload.get("transcript") or "").strip()

    # The timestamps block has been observed both nested under "timestamps" and inlined at
    # the top level; accept either rather than depending on one shape.
    block = payload.get("timestamps")
    if not isinstance(block, dict):
        block = payload if "start_time_seconds" in payload else {}

    words = block.get("words")
    starts = block.get("start_time_seconds")
    ends = block.get("end_time_seconds")

    usable = (
        isinstance(words, list) and isinstance(starts, list) and isinstance(ends, list)
        and len(words) == len(starts) == len(ends) and len(words) > 0
    )

    if usable:
        spans: list[LocalSpan] = []
        for word, start, end in zip(words, starts, ends):
            try:
                spans.append(LocalSpan(float(start), float(end), str(word).strip()))
            except (TypeError, ValueError):
                logger.warning("unparseable timestamp entry (%r, %r, %r); ignoring block",
                               word, start, end)
                spans = []
                break
        if len(spans) > 1:
            return spans, "api_multi_span"
        if len(spans) == 1:
            # Exactly what the sync endpoint does today: one span for the whole chunk.
            # Keep the API's own extent but carry the full transcript text.
            span = spans[0]
            return [LocalSpan(span.start_s, min(span.end_s, chunk_duration_s),
                              transcript or span.text)], "api_single_span"

    return [LocalSpan(0.0, chunk_duration_s, transcript)], "chunk_fallback"


def stitch_chunk_transcripts(
    chunks: Sequence[Chunk],
    spans_per_chunk: Sequence[Sequence[LocalSpan]],
    *,
    total_duration_s: float,
) -> list[Segment]:
    """Correct each chunk's timestamps by its global offset and join them into one timeline."""
    if len(chunks) != len(spans_per_chunk):
        raise StitchError(
            f"{len(chunks)} chunk(s) but {len(spans_per_chunk)} transcript group(s); "
            f"every chunk must produce exactly one group or sync is already broken"
        )
    _assert_contiguous(chunks, total_duration_s)

    segments: list[Segment] = []

    # `boundary_source` always describes how a segment's START was derived. A chunk's
    # cut_source describes how its END was chosen, so the chunk that opens at a seam
    # inherits the *previous* chunk's cut_source, not its own.
    start_source = ["file_start"] + [left.cut_source for left in chunks[:-1]]

    for order, (chunk, spans) in enumerate(zip(chunks, spans_per_chunk)):
        if not spans:
            raise StitchError(
                f"chunk {chunk.index} produced no spans; an empty chunk would drop "
                f"{chunk.duration_s:.3f}s of the timeline"
            )
        shifted = apply_offset(spans, chunk.start_s, chunk_duration_s=chunk.duration_s)

        # Pin the outer edges to the chunk boundary. Chunks are exactly contiguous, so
        # this is what makes segment[i].end == segment[i+1].start across the seam: no gap
        # to lose audio in, no overlap to double-count it.
        shifted[0] = LocalSpan(chunk.start_s, shifted[0].end_s, shifted[0].text)
        shifted[-1] = LocalSpan(shifted[-1].start_s, chunk.end_s, shifted[-1].text)

        for position, span in enumerate(shifted):
            # Within a chunk, close any gap between consecutive spans by extending the
            # earlier one; the silence between phrases belongs to the phrase that precedes
            # it, and the timeline must stay fully covered.
            end = shifted[position + 1].start_s if position + 1 < len(shifted) else span.end_s
            segments.append(Segment(
                id=len(segments),
                start=span.start_s,
                end=max(end, span.start_s),
                text=span.text.strip(),
                speaker=None,
                chunk_index=chunk.index,
                boundary_source=start_source[order] if position == 0 else "api_span",
                origin=[chunk.index],
            ))

    assert_timeline_is_sound(segments, total_duration_s)
    return segments


def assert_timeline_is_sound(segments: Sequence[Segment], total_duration_s: float) -> None:
    """Assert segments tile [0, total] exactly: ordered, positive, no gap, no overlap."""
    if not segments:
        raise StitchError("no segments were produced")
    if abs(segments[0].start) > TIME_EPSILON_S:
        raise StitchError(f"timeline starts at {segments[0].start:.6f}s, expected 0.0")
    if abs(segments[-1].end - total_duration_s) > TIME_EPSILON_S:
        raise StitchError(
            f"timeline ends at {segments[-1].end:.6f}s but the audio is {total_duration_s:.6f}s"
        )
    for segment in segments:
        if segment.duration_s <= 0:
            raise StitchError(
                f"segment {segment.id} has non-positive duration "
                f"({segment.start:.6f} -> {segment.end:.6f})"
            )
    for left, right in zip(segments, segments[1:]):
        delta = right.start - left.end
        if abs(delta) > TIME_EPSILON_S:
            raise StitchError(
                f"segment {left.id} ends at {left.end:.6f}s but segment {right.id} starts at "
                f"{right.start:.6f}s ({'gap' if delta > 0 else 'overlap'} of {abs(delta):.6f}s)"
            )


# --------------------------------------------------------------------------------------
# 3. Post-processing: merge the too-short, split the too-long
# --------------------------------------------------------------------------------------

def merge_short_segments(
    segments: Sequence[Segment],
    *,
    min_segment_s: float = 1.0,
) -> tuple[list[Segment], int]:
    """Merge each sub-min_segment_s segment into its shorter neighbour; returns (segments, merges)."""
    working = [_copy_segment(s) for s in segments]
    merges = 0

    while len(working) > 1:
        index = next((i for i, s in enumerate(working) if s.duration_s < min_segment_s), None)
        if index is None:
            break

        previous = working[index - 1] if index > 0 else None
        following = working[index + 1] if index + 1 < len(working) else None

        # Absorb into the shorter neighbour: that keeps durations even, whereas always
        # merging left grows one segment without bound on a run of short fragments.
        if previous is None:
            partner_index = index + 1
        elif following is None:
            partner_index = index - 1
        else:
            partner_index = index - 1 if previous.duration_s <= following.duration_s else index + 1

        low, high = sorted((index, partner_index))
        first, second = working[low], working[high]
        merged = Segment(
            id=first.id,
            start=first.start,
            end=second.end,
            text=" ".join(part for part in (first.text.strip(), second.text.strip()) if part),
            speaker=first.speaker or second.speaker,
            chunk_index=first.chunk_index,
            boundary_source="merged",
            origin=sorted(set(first.origin) | set(second.origin)),
        )
        logger.debug(
            "merge: segment %.3f-%.3f (%.3fs) absorbed into %.3f-%.3f",
            working[index].start, working[index].end, working[index].duration_s,
            merged.start, merged.end,
        )
        working[low:high + 1] = [merged]
        merges += 1

    if len(working) == 1 and working[0].duration_s < min_segment_s:
        logger.warning(
            "the only segment is %.3fs, shorter than the %.2fs minimum, and has no neighbour "
            "to merge into", working[0].duration_s, min_segment_s,
        )

    return _renumber(working), merges


def split_long_segments(
    segments: Sequence[Segment],
    silences: Sequence[Silence],
    *,
    max_segment_s: float = 15.0,
    min_piece_s: float = 1.0,
    snap_window_s: float = 1.5,
) -> tuple[list[Segment], int]:
    """Split each over-long segment at a sentence boundary; returns (segments, splits)."""
    result: list[Segment] = []
    splits = 0

    for segment in segments:
        pieces, made = _split_recursive(
            segment, silences,
            max_segment_s=max_segment_s, min_piece_s=min_piece_s, snap_window_s=snap_window_s,
            depth=0,
        )
        result.extend(pieces)
        splits += made

    return _renumber(result), splits


def _split_recursive(
    segment: Segment,
    silences: Sequence[Silence],
    *,
    max_segment_s: float,
    min_piece_s: float,
    snap_window_s: float,
    depth: int,
) -> tuple[list[Segment], int]:
    """Split one segment until every piece fits, or until no legal split point remains."""
    if segment.duration_s <= max_segment_s or depth >= 8:
        return [segment], 0

    chosen = _choose_split(
        segment, silences, min_piece_s=min_piece_s, snap_window_s=snap_window_s,
    )
    if chosen is None:
        logger.warning(
            "segment %.3f-%.3f is %.3fs (over the %.1fs limit) but has no sentence, clause, or "
            "word boundary that leaves >= %.2fs on both sides; leaving it whole",
            segment.start, segment.end, segment.duration_s, max_segment_s, min_piece_s,
        )
        return [segment], 0

    split_time, char_index, kind = chosen
    left_text = segment.text[:char_index].strip()
    right_text = segment.text[char_index:].strip()

    left = Segment(
        id=segment.id, start=segment.start, end=split_time, text=left_text,
        speaker=segment.speaker, chunk_index=segment.chunk_index,
        boundary_source=segment.boundary_source, origin=list(segment.origin),
    )
    right = Segment(
        id=segment.id, start=split_time, end=segment.end, text=right_text,
        speaker=segment.speaker, chunk_index=segment.chunk_index,
        boundary_source=kind, origin=list(segment.origin),
    )
    logger.debug(
        "split: %.3f-%.3f (%.3fs) -> %.3f + %.3f at %.3fs via %s",
        segment.start, segment.end, segment.duration_s,
        left.duration_s, right.duration_s, split_time, kind,
    )

    left_pieces, left_splits = _split_recursive(
        left, silences, max_segment_s=max_segment_s, min_piece_s=min_piece_s,
        snap_window_s=snap_window_s, depth=depth + 1,
    )
    right_pieces, right_splits = _split_recursive(
        right, silences, max_segment_s=max_segment_s, min_piece_s=min_piece_s,
        snap_window_s=snap_window_s, depth=depth + 1,
    )
    return left_pieces + right_pieces, 1 + left_splits + right_splits


def _choose_split(
    segment: Segment,
    silences: Sequence[Silence],
    *,
    min_piece_s: float,
    snap_window_s: float,
) -> tuple[float, int, str] | None:
    """Pick where to cut an over-long segment: (time, char index, how the time was derived)."""
    text = segment.text
    if len(text.strip()) < 2:
        return None

    lowest = segment.start + min_piece_s
    highest = segment.end - min_piece_s
    if highest <= lowest:
        return None

    ideal = (segment.start + segment.end) / 2.0
    interior = [
        s.midpoint_s for s in silences
        if lowest <= s.midpoint_s <= highest
    ]

    best: tuple[tuple[int, int, float], float, int, str] | None = None

    for char_index, kind in _text_boundaries(text):
        # Without intra-segment word timings, a character offset maps to a time only by
        # assuming a constant speaking rate. That estimate is used *solely* to decide which
        # candidate is nearest a real pause -- when a silence is within snap_window_s the
        # emitted timestamp is the measured silence, not the estimate.
        estimated = segment.start + (char_index / len(text)) * segment.duration_s
        nearest = min(interior, key=lambda m: abs(m - estimated), default=None)

        if nearest is not None and abs(nearest - estimated) <= snap_window_s:
            time_s, source, anchored = nearest, f"{kind}+silence", 0
        else:
            time_s, source, anchored = estimated, f"{kind}+estimated", 1

        if not lowest <= time_s <= highest:
            continue

        score = (_SPLIT_KIND_RANK[kind], anchored, abs(time_s - ideal))
        if best is None or score < best[0]:
            best = (score, time_s, char_index, source)

    if best is None:
        return None
    return best[1], best[2], best[3]


def _text_boundaries(text: str) -> list[tuple[int, str]]:
    """Return (character index, kind) for every place the text may legally be cut."""
    found: dict[int, str] = {}
    for pattern, kind in ((_SENTENCE_END_RE, "sentence"),
                          (_CLAUSE_END_RE, "clause"),
                          (_WORD_END_RE, "word")):
        for match in pattern.finditer(text):
            index = match.end()
            if 0 < index < len(text):
                # A position matched by several patterns keeps the strongest one.
                if index not in found or _SPLIT_KIND_RANK[kind] < _SPLIT_KIND_RANK[found[index]]:
                    found[index] = kind
    return sorted(found.items())


def _copy_segment(segment: Segment) -> Segment:
    """Deep-enough copy so post-processing never mutates the caller's segment list."""
    origin = list(segment.origin)
    if not origin and segment.chunk_index is not None:
        origin = [segment.chunk_index]
    return Segment(
        id=segment.id, start=segment.start, end=segment.end, text=segment.text,
        speaker=segment.speaker, chunk_index=segment.chunk_index,
        boundary_source=segment.boundary_source, origin=origin,
    )


def _renumber(segments: Sequence[Segment]) -> list[Segment]:
    """Reassign sequential ids after a merge or split changed the segment count."""
    return [
        Segment(
            id=index, start=s.start, end=s.end, text=s.text, speaker=s.speaker,
            chunk_index=s.chunk_index, boundary_source=s.boundary_source,
            origin=list(s.origin),
        )
        for index, s in enumerate(segments)
    ]


def post_process(
    segments: Sequence[Segment],
    silences: Sequence[Silence],
    *,
    min_segment_s: float = 1.0,
    max_segment_s: float = 15.0,
    total_duration_s: float | None = None,
) -> tuple[list[Segment], dict[str, Any]]:
    """Merge short segments then split long ones, re-asserting the timeline afterwards."""
    before = len(segments)

    # Merge first: splitting first could create a piece under the minimum, which the merge
    # pass would then glue straight back together.
    merged, merges = merge_short_segments(segments, min_segment_s=min_segment_s)
    after_merge = len(merged)

    split, splits = split_long_segments(
        merged, silences, max_segment_s=max_segment_s, min_piece_s=min_segment_s,
    )

    if total_duration_s is not None:
        assert_timeline_is_sound(split, total_duration_s)

    report = {
        "min_segment_s": min_segment_s,
        "max_segment_s": max_segment_s,
        "count_before": before,
        "count_after_merge": after_merge,
        "count_after_split": len(split),
        "merges_applied": merges,
        "splits_applied": splits,
        "still_under_min": sum(1 for s in split if s.duration_s < min_segment_s),
        "still_over_max": sum(1 for s in split if s.duration_s > max_segment_s),
    }
    logger.info(
        "post-process: %d segment(s) -> %d after %d merge(s) -> %d after %d split(s)",
        before, after_merge, merges, len(split), splits,
    )
    return split, report


# --------------------------------------------------------------------------------------
# 4. Seam verification
# --------------------------------------------------------------------------------------

def verify_seams(
    chunks: Sequence[Chunk],
    segments: Sequence[Segment],
    silences: Sequence[Silence],
    *,
    context_words: int = 6,
) -> list[dict[str, Any]]:
    """Check every chunk boundary for gaps, overlaps, and whether it landed inside a silence."""
    seams: list[dict[str, Any]] = []

    for left, right in zip(chunks, chunks[1:]):
        boundary = left.end_s
        delta = right.start_s - left.end_s

        enclosing = next(
            (s for s in silences if s.start_s <= boundary <= s.end_s), None
        )

        # The segments that meet at this boundary after post-processing.
        before_seg = max(
            (s for s in segments if s.end <= boundary + TIME_EPSILON_S),
            key=lambda s: s.end, default=None,
        )
        after_seg = min(
            (s for s in segments if s.start >= boundary - TIME_EPSILON_S),
            key=lambda s: s.start, default=None,
        )

        tail = " ".join(before_seg.text.split()[-context_words:]) if before_seg else ""
        head = " ".join(after_seg.text.split()[:context_words]) if after_seg else ""

        seams.append({
            "between_chunks": [left.index, right.index],
            "boundary_s": round(boundary, 6),
            "gap_or_overlap_s": round(delta, 9),
            "contiguous": abs(delta) <= TIME_EPSILON_S,
            "cut_source": left.cut_source,
            "inside_detected_silence": enclosing is not None,
            "silence_span_s": enclosing.to_dict() if enclosing else None,
            "margin_to_speech_s": (
                round(min(boundary - enclosing.start_s, enclosing.end_s - boundary), 6)
                if enclosing else None
            ),
            "segment_before_id": before_seg.id if before_seg else None,
            "segment_after_id": after_seg.id if after_seg else None,
            "text_before_seam": tail,
            "text_after_seam": head,
        })

    return seams


# --------------------------------------------------------------------------------------
# 5. Orchestration
# --------------------------------------------------------------------------------------

def run_asr(
    config: Config,
    client: SarvamClient,
    metrics: MetricsCollector,
    *,
    wav_path: str | Path | None = None,
) -> dict[str, Any]:
    """Chunk, transcribe, stitch, and post-process; writes segments.json and asr_report.json."""
    audio = Path(wav_path) if wav_path else config.output_dir / DEMUXED_WAV_NAME
    if not audio.exists():
        raise AsrError(
            f"{audio} not found. Run the demux stage first "
            f"(python -m src.pipeline --input <video> --stage demux)."
        )

    total_duration_s = probe_duration_s(audio)
    logger.info("ASR input: %s (%.3fs)", audio, total_duration_s)

    # --- plan -------------------------------------------------------------------------
    plan = plan_chunks_adaptive(
        audio, total_duration_s,
        max_chunk_s=config.max_chunk_s,
        threshold_db=config.silence_threshold_db,
        min_silence_s=config.min_silence_s,
    )
    chunks = plan.chunks
    logger.info(
        "chunk plan: %d chunk(s) at %.1f dB / %.2fs min silence (%d hard cut(s))",
        len(chunks), plan.threshold_db, plan.min_silence_s, plan.hard_cuts,
    )
    for chunk in chunks:
        logger.info(
            "  chunk %d: %8.3fs -> %8.3fs (%6.3fs, boundary from %s)",
            chunk.index, chunk.start_s, chunk.end_s, chunk.duration_s, chunk.cut_source,
        )

    for chunk in chunks:
        if chunk.duration_s > ASR_MAX_AUDIO_S:
            raise AsrError(
                f"chunk {chunk.index} is {chunk.duration_s:.3f}s, over the "
                f"{ASR_MAX_AUDIO_S:.0f}s REST cap; the planner is broken"
            )

    # --max-segments caps chunks here so development runs cost a fraction of a full one.
    active = chunks
    if config.max_segments is not None and config.max_segments < len(chunks):
        active = chunks[:config.max_segments]
        logger.warning(
            "--max-segments %d: transcribing %d of %d chunk(s); the timeline will cover "
            "only the first %.3fs", config.max_segments, len(active), len(chunks),
            active[-1].end_s,
        )

    # --- transcribe -------------------------------------------------------------------
    chunk_dir = config.output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    written: list[Chunk] = []
    spans_per_chunk: list[list[LocalSpan]] = []
    raw: list[dict[str, Any]] = []

    for chunk in active:
        chunk_path = chunk_dir / f"chunk_{chunk.index:03d}.wav"
        real_duration = write_wav_slice(audio, chunk_path, chunk.start_s, chunk.end_s)
        written.append(Chunk(chunk.index, chunk.start_s, chunk.end_s, chunk.cut_source, chunk_path))

        payload = client.speech_to_text(
            chunk_path,
            model=config.asr_model,
            mode=config.asr_mode,
            language_code=config.source_lang,
            with_timestamps=True,
            audio_duration_s=real_duration,
        )
        spans, granularity = local_spans_from_payload(payload, real_duration)
        spans_per_chunk.append(spans)
        raw.append({
            "chunk_index": chunk.index,
            "chunk_path": str(chunk_path),
            "chunk_start_s": round(chunk.start_s, 6),
            "chunk_end_s": round(chunk.end_s, 6),
            "sliced_duration_s": round(real_duration, 6),
            "request_id": payload.get("request_id"),
            "language_code": payload.get("language_code"),
            "timestamp_granularity": granularity,
            "spans_returned": len(spans),
            "transcript": payload.get("transcript", ""),
        })
        logger.info(
            "  chunk %d transcribed (%s, %d span(s)): %s",
            chunk.index, granularity, len(spans),
            (payload.get("transcript") or "")[:90],
        )

    # --- stitch -----------------------------------------------------------------------
    covered = written[-1].end_s if written else 0.0
    stitched = stitch_chunk_transcripts(written, spans_per_chunk, total_duration_s=covered)
    logger.info("stitched %d chunk(s) into %d segment(s)", len(written), len(stitched))

    # --- post-process -----------------------------------------------------------------
    segments, pp_report = post_process(
        stitched, plan.silences,
        min_segment_s=config.min_segment_s,
        max_segment_s=config.max_segment_s,
        total_duration_s=covered,
    )

    seams = verify_seams(written, segments, plan.silences)
    for seam in seams:
        status = "OK" if seam["contiguous"] else "BROKEN"
        logger.info(
            "seam %d|%d at %.6fs: %s (delta %+.9fs, in silence: %s)",
            seam["between_chunks"][0], seam["between_chunks"][1], seam["boundary_s"],
            status, seam["gap_or_overlap_s"], seam["inside_detected_silence"],
        )
        if not seam["contiguous"]:
            raise StitchError(f"chunk seam at {seam['boundary_s']:.6f}s is not contiguous")

    # --- write ------------------------------------------------------------------------
    segments_path = config.output_dir / "segments.json"
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    segments_path.write_text(
        json.dumps([s.to_spec_dict() for s in segments], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "audio_path": str(audio),
        "audio_duration_s": round(total_duration_s, 6),
        "timeline_covered_s": round(covered, 6),
        "asr": {
            "model": config.asr_model,
            "mode": config.asr_mode,
            "language_code": config.source_lang,
            "max_chunk_s": config.max_chunk_s,
        },
        "plan": plan.to_dict(),
        "chunks_transcribed": len(written),
        "raw_transcripts": raw,
        "stitching": {
            "segments_from_stitch": len(stitched),
            "seams_checked": len(seams),
            "seams_contiguous": sum(1 for s in seams if s["contiguous"]),
            "seams": seams,
        },
        "post_processing": pp_report,
        "segments": [s.to_debug_dict() for s in segments],
    }
    report_path = config.output_dir / "asr_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics.segment_count = len(segments)
    if segments:
        metrics.segment_mean_duration_s = sum(s.duration_s for s in segments) / len(segments)

    logger.info("wrote %s (%d segments) and %s", segments_path, len(segments), report_path)
    return report
