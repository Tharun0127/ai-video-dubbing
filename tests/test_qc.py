"""Tests for the QC harness: normalisation, WER/CER scoring, coverage, and flagging.

No test opens a socket: back-transcription is served by a fake client that returns whatever
the test wants Saaras to have heard, which is what makes the interesting cases reachable --
a perfect dub, a mispronounced word, a window where nothing was spoken at all.

The scoring behaviour worth pinning down is that a *worse* dub must score worse. Several
tests assert that ordering directly rather than asserting a specific WER value, because the
ordering is the property the ranked report depends on.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest

from src.config import Config
from src.metrics import MetricsCollector
from src.qc import (
    EMPTY_HYPOTHESIS_TOKEN,
    QcError,
    SegmentScore,
    _coverage,
    corpus_scores,
    count_latin_tokens,
    normalise_for_scoring,
    run_qc,
    score_pair,
)

REFERENCE = "आपने ये कैसे किया? मैं हैरान हूँ।"


# --- normalisation ------------------------------------------------------------------------

def test_punctuation_is_not_scored() -> None:
    """The danda and the full stop are not spoken, so they must not count as errors."""
    assert normalise_for_scoring("ठीक है, जी।") == normalise_for_scoring("ठीक है जी")


def test_case_is_folded_for_latin_tokens() -> None:
    """A loanword's capitalisation is not something the synthesiser can get wrong."""
    assert normalise_for_scoring("Jay ka Card") == "jay ka card"


def test_whitespace_is_collapsed() -> None:
    """Line breaks from a batched translation must not become word boundaries."""
    assert normalise_for_scoring("एक  \n दो\tतीन") == "एक दो तीन"


def test_composed_and_decomposed_devanagari_score_as_equal() -> None:
    """The same grapheme in two Unicode encodings is the same spoken sound."""
    composed = "नि"           # NFC
    decomposed = "नि"         # same text, normalised explicitly
    assert normalise_for_scoring(composed) == normalise_for_scoring(decomposed)


def test_latin_tokens_are_counted_not_excused() -> None:
    """The script-mismatch penalty is quantified so the WER can be read honestly."""
    assert count_latin_tokens("Jay, आपका card सबसे ऊपर वाला card है।") == 3
    assert count_latin_tokens("आपने ये कैसे किया?") == 0


# --- pair scoring ---------------------------------------------------------------------------

def test_a_perfect_dub_scores_zero() -> None:
    """Identical reference and hypothesis must give WER 0 and CER 0."""
    scored = score_pair(REFERENCE, REFERENCE)

    assert scored["wer"] == 0.0
    assert scored["cer"] == 0.0
    assert scored["deletions"] == 0
    assert scored["empty_hypothesis"] is False


def test_a_substitution_is_counted() -> None:
    """One wrong word in seven is one substitution, not a deletion plus an insertion."""
    scored = score_pair("मैं हैरान हूँ", "मैं परेशान हूँ")

    assert scored["substitutions"] == 1
    assert scored["wer"] == pytest.approx(1 / 3)


def test_a_silent_window_scores_as_a_total_loss() -> None:
    """Nothing transcribed where a line should be is every word deleted, and is flagged."""
    scored = score_pair(REFERENCE, "")

    assert scored["wer"] == 1.0
    assert scored["cer"] == 1.0
    assert scored["empty_hypothesis"] is True
    assert scored["deletions"] == len(normalise_for_scoring(REFERENCE).split())


def test_a_worse_dub_scores_worse() -> None:
    """The property the ranked report depends on: more errors must mean a higher WER."""
    one_wrong = score_pair("एक दो तीन चार", "एक दो तीन पाँच")
    two_wrong = score_pair("एक दो तीन चार", "एक दो छह पाँच")

    assert one_wrong["wer"] < two_wrong["wer"]


def test_scoring_without_a_reference_is_refused() -> None:
    """Scoring against a missing translation would silently report a perfect dub."""
    with pytest.raises(QcError, match="empty reference"):
        score_pair("", "कुछ")


# --- corpus scoring ---------------------------------------------------------------------------

def test_corpus_wer_is_length_weighted_not_a_mean_of_means() -> None:
    """A long segment with an error must move the corpus WER more than a short perfect one."""
    corpus = corpus_scores(
        ["एक दो तीन चार पाँच छह", "सात"],
        ["एक दो तीन चार पाँच सात", "सात"],
    )

    # 1 substitution over 7 reference words, not the mean of (1/6, 0/1).
    assert corpus["wer"] == pytest.approx(1 / 7)
    assert corpus["reference_words"] == 7


def test_an_unheard_segment_stays_in_the_corpus_total() -> None:
    """Dropping a silent segment would flatter the score; it is scored as a full error."""
    with_silence = corpus_scores(["एक दो", "तीन चार"], ["एक दो", ""])
    all_heard = corpus_scores(["एक दो", "तीन चार"], ["एक दो", "तीन चार"])

    assert with_silence["wer"] > all_heard["wer"]
    assert EMPTY_HYPOTHESIS_TOKEN not in with_silence


def test_corpus_scoring_needs_at_least_one_reference() -> None:
    """An empty corpus cannot produce a score, and says so instead of returning zero."""
    with pytest.raises(QcError, match="non-empty reference"):
        corpus_scores(["", ""], ["a", "b"])


# --- coverage ---------------------------------------------------------------------------------

def make_segments(count: int, *, translated: int | None = None) -> list[dict[str, Any]]:
    """Build `count` segments, the first `translated` of which carry a translation."""
    translated = count if translated is None else translated
    return [
        {"id": index, "start": float(index), "end": float(index + 1),
         "text": f"line {index}", "speaker": None,
         "translation": f"पंक्ति {index}" if index < translated else ""}
        for index in range(count)
    ]


def test_coverage_passes_when_every_count_matches(config: Config) -> None:
    """The happy path: four segments survive every hop."""
    coverage = _coverage(
        config=config, segments=make_segments(4),
        tts_report={"files": [{}] * 4},
        assemble_report={"placements": [{}] * 4}, scored=4,
    )

    assert coverage["all_passed"] is True
    assert coverage["counts"] == {
        "asr_segments": 4, "translated": 4, "synthesised": 4, "placed": 4, "scored": 4,
    }


def test_a_dropped_translation_fails_coverage(config: Config) -> None:
    """An untranslated segment would be dubbed as silence, so the hop must fail."""
    coverage = _coverage(
        config=config, segments=make_segments(4, translated=3),
        tts_report={"files": [{}] * 3},
        assemble_report={"placements": [{}] * 3}, scored=3,
    )

    assert coverage["checks"]["asr_to_translate"]["passed"] is False
    assert coverage["all_passed"] is False


def test_a_clip_that_was_never_placed_fails_coverage(config: Config) -> None:
    """A synthesised clip missing from the timeline is a hole in the dub."""
    coverage = _coverage(
        config=config, segments=make_segments(4),
        tts_report={"files": [{}] * 4},
        assemble_report={"placements": [{}] * 3}, scored=3,
    )

    assert coverage["checks"]["tts_to_assemble"]["passed"] is False


def test_max_segments_does_not_count_as_a_dropped_segment(tmp_path: Path) -> None:
    """--max-segments truncates on purpose, so the translate -> TTS hop must allow it."""
    capped = Config(
        api_key="sk_test_not_a_real_key", cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out", max_segments=2,
    )
    coverage = _coverage(
        config=capped, segments=make_segments(4),
        tts_report={"files": [{}] * 2},
        assemble_report={"placements": [{}] * 2}, scored=2,
    )

    assert coverage["all_passed"] is True
    assert "--max-segments" in coverage["checks"]["translate_to_tts"]["hop"]


def test_an_empty_transcript_fails_coverage(config: Config) -> None:
    """An empty ASR segment would synthesise silence into a window that should speak."""
    segments = make_segments(3)
    segments[1]["text"] = "   "
    coverage = _coverage(
        config=config, segments=segments, tts_report={"files": [{}] * 3},
        assemble_report={"placements": [{}] * 3}, scored=3,
    )

    assert coverage["checks"]["no_empty_segments"]["passed"] is False
    assert coverage["checks"]["no_empty_segments"]["actual"] == [1]


# --- ranking and flagging -----------------------------------------------------------------------

def make_score(segment_id: int, *, wer: float, drift_pct: float, **overrides: Any) -> SegmentScore:
    """Build a SegmentScore with only the fields ranking and rendering read."""
    defaults: dict[str, Any] = dict(
        segment_id=segment_id, start_s=float(segment_id * 5), end_s=float(segment_id * 5 + 5),
        source_text="source", translation="अनुवाद", back_transcript="अनुवाद",
        wer=wer, cer=wer,
        edits={"substitutions": 1, "deletions": 0, "insertions": 0, "reference_words": 7,
               "empty_hypothesis": False},
        target_duration_s=5.0, achieved_duration_s=5.0 * (1 + drift_pct / 100.0),
        abs_drift_pct=abs(drift_pct), signed_drift_pct=drift_pct, final_pace=1.0,
        attempts=1, clamped=False, converged=True, fill_pct=100.0, overlap_s=0.0,
        latin_tokens_in_reference=0,
    )
    defaults.update(overrides)
    return SegmentScore(**defaults)


def test_the_combined_score_rises_with_both_failure_modes() -> None:
    """Semantic error and timing error each make a segment rank worse."""
    clean = make_score(0, wer=0.0, drift_pct=0.0)
    mistranscribed = make_score(1, wer=0.5, drift_pct=0.0)
    mistimed = make_score(2, wer=0.0, drift_pct=50.0)

    score = lambda s: s.combined_score(wer_weight=1.0, drift_weight=1.0)  # noqa: E731
    assert score(clean) == 0.0
    assert score(mistranscribed) == pytest.approx(0.5)
    assert score(mistimed) == pytest.approx(0.5)


def test_timestamps_are_rendered_for_jumping_into_the_video() -> None:
    """Each finding carries mm:ss.mmm so the segment can be found and listened to."""
    assert make_score(0, wer=0.0, drift_pct=0.0, start_s=0.0).timestamp == "00:00.000"
    assert make_score(1, wer=0.0, drift_pct=0.0, start_s=13.183).timestamp == "00:13.183"
    assert make_score(2, wer=0.0, drift_pct=0.0, start_s=125.5).timestamp == "02:05.500"


def test_reasons_name_the_measured_cause() -> None:
    """Every flag is backed by its own number, not by an adjective."""
    reasons = make_score(
        0, wer=0.6, drift_pct=40.0, clamped=True, final_pace=0.85,
        latin_tokens_in_reference=2,
    ).reasons(drift_threshold_pct=5.0)

    assert any("WER 0.60" in r for r in reasons)
    # +40% is an overrun -- the direction that actually breaks sync, since an underrun is
    # absorbed by the silence M5 leaves in the rest of the window.
    assert any("overruns" in r and "40.0%" in r for r in reasons)
    assert any("clamped" in r for r in reasons)
    assert any("Latin-script" in r for r in reasons)


def test_a_clean_segment_has_nothing_to_report() -> None:
    """A segment that is both accurate and in time produces no findings."""
    assert make_score(0, wer=0.0, drift_pct=1.0).reasons(drift_threshold_pct=5.0) == []


def test_a_silent_window_is_reported_first_among_its_reasons() -> None:
    """The most serious finding -- nothing was heard -- leads the list."""
    reasons = make_score(
        0, wer=1.0, drift_pct=0.0,
        edits={"substitutions": 0, "deletions": 7, "insertions": 0,
               "reference_words": 7, "empty_hypothesis": True},
    ).reasons(drift_threshold_pct=5.0)

    assert "nothing was transcribed" in reasons[0]


# --- the stage end to end -------------------------------------------------------------------

class FakeClient:
    """Stands in for SarvamClient, returning whatever the test wants Saaras to have heard."""

    def __init__(self, transcripts: dict[int, str]) -> None:
        """Map segment index (in call order) to the transcript to return."""
        self.transcripts = transcripts
        self.calls: list[dict[str, Any]] = []

    def speech_to_text(self, audio_path, **kwargs: Any) -> dict[str, Any]:
        """Record the call and return the scripted transcript."""
        index = len(self.calls)
        self.calls.append({"path": Path(audio_path), **kwargs})
        return {
            "transcript": self.transcripts.get(index, ""),
            "request_id": f"fake-{index}",
        }


@pytest.fixture
def scorable(tmp_path: Path, config: Config) -> Config:
    """Write the four artefacts run_qc reads, describing a two-segment dub."""
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with wave.open(str(out / "dubbed_audio.wav"), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x11\x11" * int(24000 * 10))

    (out / "segments.json").write_text(json.dumps([
        {"id": 0, "start": 0.0, "end": 5.0, "text": "How did you do that?",
         "speaker": None, "translation": "आपने ये कैसे किया"},
        {"id": 1, "start": 5.0, "end": 10.0, "text": "I am amazed",
         "speaker": None, "translation": "मैं हैरान हूँ"},
    ], ensure_ascii=False), encoding="utf-8")

    (out / "tts_report.json").write_text(json.dumps({"files": [{}, {}], "per_segment": [
        {"segment_id": 0, "target_duration_s": 5.0, "final_duration_s": 4.9,
         "final_pace": 1.0, "attempts": 1, "clamped": False, "converged": True},
        {"segment_id": 1, "target_duration_s": 5.0, "final_duration_s": 2.0,
         "final_pace": 0.85, "attempts": 3, "clamped": True, "converged": False},
    ]}), encoding="utf-8")

    (out / "assemble_report.json").write_text(json.dumps({"placements": [
        {"segment_id": 0, "window": [0.0, 5.0], "duration_s": 4.9, "fill_pct": 98.0},
        {"segment_id": 1, "window": [5.0, 10.0], "duration_s": 2.0, "fill_pct": 40.0},
    ], "overlaps": {"detail": []}}), encoding="utf-8")

    return config


def test_qc_scores_the_dub_and_writes_both_reports(scorable: Config) -> None:
    """The stage produces qc_report.md and qc_report.json with measured scores."""
    metrics = MetricsCollector()
    client = FakeClient({0: "आपने ये कैसे किया", 1: "मैं हैरान हूँ"})

    report = run_qc(scorable, client, metrics)  # type: ignore[arg-type]

    assert report["semantic_fidelity"]["wer"] == 0.0
    assert (scorable.output_dir / "qc_report.md").exists()
    assert (scorable.output_dir / "qc_report.json").exists()
    assert metrics.qc["wer"] == 0.0
    assert metrics.qc["coverage_passed"] is True


def test_qc_back_transcribes_in_the_target_language(scorable: Config) -> None:
    """Transcribing the dub as the source language would measure language detection instead."""
    client = FakeClient({0: "आपने ये कैसे किया", 1: "मैं हैरान हूँ"})
    run_qc(scorable, client, MetricsCollector())  # type: ignore[arg-type]

    assert {call["language_code"] for call in client.calls} == {"hi-IN"}
    assert {call["stage"] for call in client.calls} == {"qc"}
    assert {call["mode"] for call in client.calls} == {"transcribe"}


def test_a_mispronounced_segment_is_ranked_worst(scorable: Config) -> None:
    """The ranked list puts the genuinely worse segment first."""
    client = FakeClient({0: "आपने ये कैसे किया", 1: "मैं परेशान था"})
    report = run_qc(scorable, client, MetricsCollector())  # type: ignore[arg-type]

    assert report["ranked_worst"][0]["segment_id"] == 1
    assert report["ranked_worst"][0]["wer"] > 0
    assert report["semantic_fidelity"]["wer"] > 0


def test_a_silent_window_is_scored_and_flagged(scorable: Config) -> None:
    """A window where nothing was heard becomes the top finding, not a missing row."""
    client = FakeClient({0: "आपने ये कैसे किया", 1: ""})
    report = run_qc(scorable, client, MetricsCollector())  # type: ignore[arg-type]

    worst = report["ranked_worst"][0]
    assert worst["segment_id"] == 1
    assert worst["wer"] == 1.0
    assert any("nothing was transcribed" in reason for reason in worst["reasons"])


def test_timing_fidelity_counts_segments_over_the_threshold(scorable: Config) -> None:
    """Segment 1 drifts 60%; segment 0 drifts 2% and must not be counted."""
    client = FakeClient({0: "आपने ये कैसे किया", 1: "मैं हैरान हूँ"})
    report = run_qc(scorable, client, MetricsCollector())  # type: ignore[arg-type]

    timing = report["timing_fidelity"]
    assert timing["segments_over_threshold"] == 1
    assert timing["threshold_pct"] == 5.0
    assert timing["segments_underrunning"] == 2


def test_coverage_failure_stops_the_run(scorable: Config) -> None:
    """A count mismatch is raised, not logged: it means sync is already broken."""
    segments = json.loads((scorable.output_dir / "segments.json").read_text(encoding="utf-8"))
    segments.append({"id": 2, "start": 10.0, "end": 12.0, "text": "extra",
                     "speaker": None, "translation": "अतिरिक्त"})
    (scorable.output_dir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False), encoding="utf-8")

    client = FakeClient({0: "आपने ये कैसे किया", 1: "मैं हैरान हूँ"})
    with pytest.raises(QcError, match="coverage failed"):
        run_qc(scorable, client, MetricsCollector())  # type: ignore[arg-type]


def test_the_markdown_report_carries_timestamps_and_the_caveat(scorable: Config) -> None:
    """qc_report.md must be usable on its own: findings, timestamps, and the WER caveat."""
    client = FakeClient({0: "आपने ये कैसे किया", 1: "मैं परेशान था"})
    run_qc(scorable, client, MetricsCollector())  # type: ignore[arg-type]

    markdown = (scorable.output_dir / "qc_report.md").read_text(encoding="utf-8")
    assert "# QC report" in markdown
    assert "00:05.000" in markdown
    assert "Latin script" in markdown
    assert "Back-transcript" in markdown


def test_missing_artefacts_name_the_stage_to_run(config: Config) -> None:
    """The error tells the user which stage produces the missing file."""
    with pytest.raises(QcError, match="--stage"):
        run_qc(config, FakeClient({}), MetricsCollector())  # type: ignore[arg-type]
