"""
Tests for chunk planning and timestamp offset correction, on synthetic boundaries only.

Not a single test here touches audio, ffmpeg, or the Sarvam API. The stitching maths is
where an off-by-one silently destroys lip sync -- a segment shifted by one chunk offset
still *looks* plausible in segments.json -- so it is verified against hand-computed
numbers, independently of whatever a real transcript happens to contain.

The multi-span cases matter even though today's sync endpoint returns one span per chunk
(see docs/api-notes.md): they are what proves the offset arithmetic generalises to the
Batch API's real word timings without a rewrite.
"""

from __future__ import annotations

import pytest

from src.audio import Silence
from src.stages.asr import (
    Chunk,
    LocalSpan,
    Segment,
    StitchError,
    apply_offset,
    assert_timeline_is_sound,
    local_spans_from_payload,
    merge_short_segments,
    plan_chunks,
    post_process,
    split_long_segments,
    stitch_chunk_transcripts,
    verify_seams,
)


def make_chunks(*boundaries: float, sources: list[str] | None = None) -> list[Chunk]:
    """Build contiguous chunks from an ascending list of boundary times starting at 0."""
    edges = [0.0, *boundaries]
    kinds = sources or ["silence"] * (len(edges) - 2) + ["eof"]
    return [
        Chunk(index=i, start_s=edges[i], end_s=edges[i + 1], cut_source=kinds[i])
        for i in range(len(edges) - 1)
    ]


# ======================================================================================
# apply_offset -- the core arithmetic
# ======================================================================================

def test_offset_shifts_every_span_by_the_chunk_start() -> None:
    """A span 2s into a chunk that starts at 25.27s lands at 27.27s globally."""
    spans = [LocalSpan(2.0, 5.0, "hello"), LocalSpan(5.0, 9.5, "world")]

    shifted = apply_offset(spans, 25.27, chunk_duration_s=10.837)

    assert [s.start_s for s in shifted] == pytest.approx([27.27, 30.27])
    assert [s.end_s for s in shifted] == pytest.approx([30.27, 34.77])
    assert [s.text for s in shifted] == ["hello", "world"]


def test_offset_of_zero_is_the_identity_for_the_first_chunk() -> None:
    """Chunk 0 starts at 0.0, so its local times are already global."""
    spans = [LocalSpan(0.0, 3.25, "first")]

    shifted = apply_offset(spans, 0.0, chunk_duration_s=25.27)

    assert (shifted[0].start_s, shifted[0].end_s) == (0.0, 3.25)


def test_offsets_accumulate_correctly_across_three_chunks() -> None:
    """Each chunk is shifted by its own start, never by a running sum of durations."""
    chunks = make_chunks(10.0, 22.5, 30.0)
    local = [LocalSpan(1.0, 2.0, "x")]

    starts = [apply_offset(local, c.start_s, chunk_duration_s=c.duration_s)[0].start_s
              for c in chunks]

    # The classic bug is shifting chunk 2 by chunk 1's duration (12.5) instead of by its
    # own global start (22.5); pinning the literals here makes that regression fail loudly.
    assert starts == [1.0, 11.0, 23.5]


def test_span_ending_a_hair_past_the_chunk_is_clamped_not_rejected() -> None:
    """Float noise at the chunk edge is clamped, so it cannot bleed into the next chunk."""
    spans = [LocalSpan(0.0, 10.0000005, "edge")]

    shifted = apply_offset(spans, 5.0, chunk_duration_s=10.0)

    assert shifted[0].end_s == 15.0


def test_span_well_past_the_chunk_end_is_an_error() -> None:
    """A timestamp outside the audio means the API or the plan is wrong; never silently clamp it."""
    with pytest.raises(StitchError, match="outside the audio"):
        apply_offset([LocalSpan(0.0, 40.0, "too long")], 0.0, chunk_duration_s=10.0)


def test_negative_local_start_is_an_error() -> None:
    """API timestamps are chunk-relative; a negative one means they were misread as global."""
    with pytest.raises(StitchError, match="negative chunk-local start"):
        apply_offset([LocalSpan(-0.5, 2.0, "bad")], 10.0, chunk_duration_s=20.0)


def test_out_of_order_spans_are_an_error() -> None:
    """Overlapping spans within a chunk would double-count audio downstream."""
    spans = [LocalSpan(0.0, 5.0, "a"), LocalSpan(3.0, 8.0, "b")]

    with pytest.raises(StitchError, match="ordered and non-overlapping"):
        apply_offset(spans, 0.0, chunk_duration_s=10.0)


def test_negative_chunk_offset_is_an_error() -> None:
    """A chunk cannot start before the file does."""
    with pytest.raises(StitchError, match="offset must be >= 0"):
        apply_offset([LocalSpan(0.0, 1.0, "a")], -1.0, chunk_duration_s=10.0)


# ======================================================================================
# stitch_chunk_transcripts
# ======================================================================================

def test_stitching_two_chunks_produces_one_contiguous_timeline() -> None:
    """Two chunks meeting at 25.27s yield segments that touch exactly, with no gap."""
    chunks = make_chunks(25.27, 36.107)
    spans = [[LocalSpan(0.0, 25.27, "first half")], [LocalSpan(0.0, 10.837, "second half")]]

    segments = stitch_chunk_transcripts(chunks, spans, total_duration_s=36.107)

    assert [(s.start, s.end) for s in segments] == [(0.0, 25.27), (25.27, 36.107)]
    assert segments[0].end == segments[1].start
    assert [s.id for s in segments] == [0, 1]


def test_stitching_preserves_intra_chunk_span_timings() -> None:
    """Multi-span chunks keep their internal boundaries, offset onto the global timeline."""
    chunks = make_chunks(10.0, 20.0)
    spans = [
        [LocalSpan(0.0, 4.0, "a"), LocalSpan(4.0, 10.0, "b")],
        [LocalSpan(0.0, 6.0, "c"), LocalSpan(6.0, 10.0, "d")],
    ]

    segments = stitch_chunk_transcripts(chunks, spans, total_duration_s=20.0)

    assert [(s.start, s.end, s.text) for s in segments] == [
        (0.0, 4.0, "a"), (4.0, 10.0, "b"), (10.0, 16.0, "c"), (16.0, 20.0, "d"),
    ]


def test_stitching_pins_chunk_edges_so_the_seam_never_gaps() -> None:
    """Even if the API under-reports a chunk's extent, the seam stays exactly contiguous."""
    chunks = make_chunks(10.0, 20.0)
    # The API claims speech ended at 9.1s and the next chunk's began at 0.4s -- 1.3s that
    # would otherwise fall into a hole between segments.
    spans = [[LocalSpan(0.0, 9.1, "left")], [LocalSpan(0.4, 10.0, "right")]]

    segments = stitch_chunk_transcripts(chunks, spans, total_duration_s=20.0)

    assert segments[0].end == 10.0
    assert segments[1].start == 10.0
    assert segments[0].end == segments[1].start


def test_stitching_rejects_a_gap_between_chunks() -> None:
    """A chunk plan with a hole drops audio; it must fail rather than produce silent drift."""
    chunks = [Chunk(0, 0.0, 10.0), Chunk(1, 10.5, 20.0)]

    with pytest.raises(StitchError, match="gap of 0.500000s"):
        stitch_chunk_transcripts(
            chunks, [[LocalSpan(0.0, 10.0, "a")], [LocalSpan(0.0, 9.5, "b")]],
            total_duration_s=20.0,
        )


def test_stitching_rejects_overlapping_chunks() -> None:
    """Overlapping chunks would transcribe the same words twice."""
    chunks = [Chunk(0, 0.0, 10.0), Chunk(1, 9.0, 20.0)]

    with pytest.raises(StitchError, match="overlap of 1.000000s"):
        stitch_chunk_transcripts(
            chunks, [[LocalSpan(0.0, 10.0, "a")], [LocalSpan(0.0, 11.0, "b")]],
            total_duration_s=20.0,
        )


def test_stitching_rejects_a_mismatched_transcript_count() -> None:
    """One transcript group per chunk, always -- an off-by-one here destroys sync."""
    chunks = make_chunks(10.0, 20.0)

    with pytest.raises(StitchError, match="2 chunk\\(s\\) but 1 transcript group"):
        stitch_chunk_transcripts(chunks, [[LocalSpan(0.0, 10.0, "only one")]],
                                 total_duration_s=20.0)


def test_stitching_rejects_a_chunk_with_no_spans() -> None:
    """An empty chunk would silently drop its slice of the timeline."""
    chunks = make_chunks(10.0, 20.0)

    with pytest.raises(StitchError, match="produced no spans"):
        stitch_chunk_transcripts(chunks, [[LocalSpan(0.0, 10.0, "a")], []],
                                 total_duration_s=20.0)


def test_stitching_rejects_a_plan_that_does_not_reach_the_end() -> None:
    """Chunks must cover the whole file, or the tail is silently untranscribed."""
    chunks = make_chunks(10.0, 20.0)

    with pytest.raises(StitchError, match="would be lost"):
        stitch_chunk_transcripts(
            chunks, [[LocalSpan(0.0, 10.0, "a")], [LocalSpan(0.0, 10.0, "b")]],
            total_duration_s=36.107,
        )


# ======================================================================================
# plan_chunks
# ======================================================================================

def test_short_audio_is_a_single_chunk() -> None:
    """Audio under the cap needs no splitting and issues one API call."""
    chunks = plan_chunks(20.0, [Silence(9.0, 9.5)], max_chunk_s=30.0)

    assert len(chunks) == 1
    assert (chunks[0].start_s, chunks[0].end_s, chunks[0].cut_source) == (0.0, 20.0, "eof")


def test_over_cap_audio_is_split_at_a_silence_midpoint() -> None:
    """The boundary lands at the centre of the chosen pause, furthest from speech."""
    chunks = plan_chunks(36.107, [Silence(25.062, 25.478)], max_chunk_s=30.0)

    assert len(chunks) == 2
    assert chunks[0].end_s == pytest.approx(25.27, abs=1e-3)
    assert chunks[0].cut_source == "silence"
    assert chunks[1].end_s == pytest.approx(36.107)
    assert all(c.duration_s <= 30.0 for c in chunks)


def test_planner_balances_rather_than_greedily_filling_to_the_cap() -> None:
    """Given a choice, the planner picks the pause nearest an even split, not the latest one."""
    silences = [Silence(9.9, 10.1), Silence(19.9, 20.1), Silence(28.9, 29.1)]

    chunks = plan_chunks(40.0, silences, max_chunk_s=30.0)

    # Greedy-to-the-cap would cut at 29.0 and leave an 11s runt; balanced picks 20.0.
    assert chunks[0].end_s == pytest.approx(20.0)
    assert [round(c.duration_s, 3) for c in chunks] == [20.0, 20.0]


def test_planner_never_exceeds_the_cap_even_with_no_usable_silence() -> None:
    """With no pause in range the planner hard-cuts at the cap and flags it as risky."""
    chunks = plan_chunks(50.0, [Silence(0.0, 0.2)], max_chunk_s=30.0)

    assert all(c.duration_s <= 30.0 for c in chunks)
    assert chunks[0].cut_source == "hard"
    assert chunks[0].end_s == pytest.approx(30.0)


def test_planner_never_strands_a_tail_it_cannot_cover() -> None:
    """A cut is rejected if the remaining audio would no longer fit in the chunks left."""
    # Cutting at the 1.0s pause would strand 69s for a single 30s chunk.
    chunks = plan_chunks(70.0, [Silence(0.9, 1.1), Silence(39.9, 40.1)], max_chunk_s=30.0)

    assert all(c.duration_s <= 30.0 for c in chunks)
    assert chunks[0].end_s > 1.1


def test_planner_covers_the_timeline_exactly() -> None:
    """Chunks tile [0, duration] with no gap and no overlap, for any silence layout."""
    silences = [Silence(t, t + 0.3) for t in (7.0, 15.0, 24.0, 33.0, 41.0, 55.0)]

    chunks = plan_chunks(62.0, silences, max_chunk_s=30.0)

    assert chunks[0].start_s == 0.0
    assert chunks[-1].end_s == pytest.approx(62.0)
    for left, right in zip(chunks, chunks[1:]):
        assert left.end_s == right.start_s


# ======================================================================================
# local_spans_from_payload
# ======================================================================================

def test_payload_without_timestamps_falls_back_to_the_whole_chunk() -> None:
    """No timestamp block means one span covering the chunk -- never a dropped transcript."""
    spans, granularity = local_spans_from_payload({"transcript": "hello there"}, 10.837)

    assert granularity == "chunk_fallback"
    assert (spans[0].start_s, spans[0].end_s, spans[0].text) == (0.0, 10.837, "hello there")


def test_single_span_timestamps_keep_the_full_transcript() -> None:
    """The sync endpoint's one-span-per-chunk shape carries the transcript, not the word list."""
    payload = {
        "transcript": "How did you do that? I'm amazed.",
        "timestamps": {"words": ["How did you do that? I'm amazed."],
                       "start_time_seconds": [0.0], "end_time_seconds": [10.0]},
    }

    spans, granularity = local_spans_from_payload(payload, 10.837)

    assert granularity == "api_single_span"
    assert len(spans) == 1
    assert spans[0].text == "How did you do that? I'm amazed."


def test_multi_span_timestamps_are_used_directly() -> None:
    """Real per-word timings (Batch API) are passed through without the fallback."""
    payload = {
        "transcript": "one two",
        "timestamps": {"words": ["one", "two"],
                       "start_time_seconds": [0.5, 2.0], "end_time_seconds": [1.5, 3.0]},
    }

    spans, granularity = local_spans_from_payload(payload, 10.0)

    assert granularity == "api_multi_span"
    assert [(s.start_s, s.end_s, s.text) for s in spans] == [(0.5, 1.5, "one"), (2.0, 3.0, "two")]


def test_ragged_timestamp_arrays_fall_back_instead_of_crashing() -> None:
    """Mismatched array lengths are a degraded response, not a reason to lose the transcript."""
    payload = {
        "transcript": "one two",
        "timestamps": {"words": ["one", "two"], "start_time_seconds": [0.0],
                       "end_time_seconds": [1.0, 2.0]},
    }

    spans, granularity = local_spans_from_payload(payload, 10.0)

    assert granularity == "chunk_fallback"
    assert spans[0].text == "one two"


# ======================================================================================
# Post-processing: merge and split
# ======================================================================================

def tiled(*edges: float, texts: list[str] | None = None) -> list[Segment]:
    """Build contiguous segments from boundary times, for post-processing tests."""
    words = texts or [f"text {i}" for i in range(len(edges) - 1)]
    return [
        Segment(id=i, start=edges[i], end=edges[i + 1], text=words[i])
        for i in range(len(edges) - 1)
    ]


def test_short_segment_is_merged_into_its_neighbour() -> None:
    """A 0.4s fragment cannot be duration-fitted, so it is absorbed."""
    segments = tiled(0.0, 5.0, 5.4, 12.0, texts=["a", "b", "c"])

    merged, count = merge_short_segments(segments, min_segment_s=1.0)

    assert count == 1
    assert len(merged) == 2
    assert all(s.duration_s >= 1.0 for s in merged)
    # 'b' joins the shorter neighbour, which is 'a' (5.0s) rather than 'c' (6.6s).
    assert [s.text for s in merged] == ["a b", "c"]
    assert [(s.start, s.end) for s in merged] == [(0.0, 5.4), (5.4, 12.0)]


def test_merging_picks_the_shorter_neighbour() -> None:
    """Absorbing into the shorter side keeps segment durations even."""
    segments = tiled(0.0, 12.0, 12.5, 15.0, texts=["long", "tiny", "short"])

    merged, _ = merge_short_segments(segments, min_segment_s=1.0)

    # 'tiny' joins 'short' (2.5s), not 'long' (12s).
    assert [s.text for s in merged] == ["long", "tiny short"]


def test_merging_preserves_contiguity_and_total_duration() -> None:
    """Merging must never move the outer edges of the timeline."""
    segments = tiled(0.0, 0.5, 0.8, 1.1, 20.0)

    merged, _ = merge_short_segments(segments, min_segment_s=1.0)

    assert_timeline_is_sound(merged, 20.0)
    assert merged[0].start == 0.0 and merged[-1].end == 20.0


def test_a_lone_short_segment_is_left_alone() -> None:
    """With no neighbour there is nothing to merge into; it is kept and warned about."""
    segments = tiled(0.0, 0.5)

    merged, count = merge_short_segments(segments, min_segment_s=1.0)

    assert count == 0 and len(merged) == 1


def test_long_segment_splits_at_a_sentence_boundary() -> None:
    """An over-long segment is cut between sentences, not mid-sentence."""
    text = "First sentence here. Second sentence here. Third sentence here."
    segments = [Segment(id=0, start=0.0, end=24.0, text=text)]

    split, count = split_long_segments(segments, [], max_segment_s=15.0)

    assert count >= 1
    assert all(s.duration_s <= 15.0 for s in split)
    assert all(s.text.strip() for s in split)
    # No word is lost or duplicated across the split.
    assert " ".join(s.text for s in split).split() == text.split()


def test_split_snaps_to_a_real_silence_when_one_is_close() -> None:
    """When a detected pause sits near the sentence boundary, the emitted time is the pause."""
    text = "First sentence here. Second sentence here."
    segments = [Segment(id=0, start=0.0, end=20.0, text=text)]
    silences = [Silence(9.8, 10.2)]

    split, _ = split_long_segments(segments, silences, max_segment_s=15.0)

    assert len(split) == 2
    assert split[0].end == pytest.approx(10.0)
    assert split[1].boundary_source == "sentence+silence"


def test_split_falls_back_to_an_estimate_and_says_so() -> None:
    """With no nearby pause the time is interpolated from character position, and labelled."""
    text = "First sentence here. Second sentence here."
    segments = [Segment(id=0, start=0.0, end=20.0, text=text)]

    split, _ = split_long_segments(segments, [], max_segment_s=15.0)

    assert split[1].boundary_source == "sentence+estimated"


def test_split_preserves_the_timeline() -> None:
    """Splitting keeps segments tiling exactly; the pieces meet where the parent was cut."""
    text = "One. Two. Three. Four. Five. Six."
    segments = [Segment(id=0, start=5.0, end=45.0, text=text)]

    split, _ = split_long_segments(segments, [], max_segment_s=15.0)

    assert split[0].start == 5.0 and split[-1].end == 45.0
    for left, right in zip(split, split[1:]):
        assert left.end == right.start


def test_unsplittable_segment_is_kept_rather_than_mangled() -> None:
    """A long segment with no boundary to cut on is left whole and warned about, not dropped."""
    segments = [Segment(id=0, start=0.0, end=40.0, text="Uninterrupted")]

    split, count = split_long_segments(segments, [], max_segment_s=15.0)

    assert count == 0 and len(split) == 1 and split[0].text == "Uninterrupted"


def test_post_process_merges_before_it_splits() -> None:
    """Running split first would create pieces the merge pass immediately glued back."""
    segments = tiled(0.0, 0.6, 30.0, texts=["Hi.", "One. Two. Three. Four. Five."])

    result, report = post_process(segments, [], min_segment_s=1.0, max_segment_s=15.0,
                                  total_duration_s=30.0)

    assert report["count_before"] == 2
    assert report["merges_applied"] == 1
    assert report["splits_applied"] >= 1
    assert report["still_under_min"] == 0
    assert report["still_over_max"] == 0
    assert_timeline_is_sound(result, 30.0)


def test_post_process_never_loses_or_duplicates_a_word() -> None:
    """Text is conserved across merge and split; sync depends on every word surviving."""
    segments = tiled(0.0, 0.5, 26.0, texts=["Alpha bravo.", "Charlie. Delta. Echo. Foxtrot."])
    original = " ".join(s.text for s in segments).split()

    result, _ = post_process(segments, [], min_segment_s=1.0, max_segment_s=15.0,
                             total_duration_s=26.0)

    assert " ".join(s.text for s in result).split() == original


# ======================================================================================
# Seam verification
# ======================================================================================

def test_verify_seams_reports_a_clean_contiguous_boundary() -> None:
    """A seam cut inside a detected pause reports contiguous, in-silence, with real margin."""
    chunks = make_chunks(25.270219, 36.107062)
    segments = tiled(0.0, 25.270219, 36.107062, texts=["Really? Yeah.", "How did you do that?"])
    silences = [Silence(25.062313, 25.478125)]

    seams = verify_seams(chunks, segments, silences)

    assert len(seams) == 1
    assert seams[0]["contiguous"] is True
    assert seams[0]["gap_or_overlap_s"] == 0.0
    assert seams[0]["inside_detected_silence"] is True
    assert seams[0]["margin_to_speech_s"] == pytest.approx(0.207906, abs=1e-5)
    assert seams[0]["text_before_seam"] == "Really? Yeah."
    assert seams[0]["text_after_seam"] == "How did you do that?"


def test_verify_seams_flags_a_hard_cut_outside_any_silence() -> None:
    """A boundary at the 30s cap with no pause around it is reported, not hidden."""
    chunks = make_chunks(30.0, 50.0, sources=["hard", "eof"])
    segments = tiled(0.0, 30.0, 50.0)

    seams = verify_seams(chunks, segments, [])

    assert seams[0]["cut_source"] == "hard"
    assert seams[0]["inside_detected_silence"] is False
    assert seams[0]["margin_to_speech_s"] is None
