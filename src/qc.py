"""
QC stage: score the pipeline's own output and flag the segments that are genuinely worst.

Input:  output/dubbed_audio.wav (M5), output/segments.json (M3 translations),
        output/tts_report.json (M4 fit records), output/assemble_report.json (M5 placements).
Output: output/qc_report.md (ranked findings, each with a timestamp to jump to) and
        output/qc_report.json, plus the qc block of output/metrics.json.

WHAT IS BEING SCORED, AND AGAINST WHAT

1. **Semantic fidelity.** The dubbed track is sliced at each segment's window and sent back
   through Saaras in the *target* language, then compared with the stage-3 translation that
   was supposed to be spoken there. Reference = what we asked Bulbul to say; hypothesis =
   what Saaras hears in the finished dub. A high WER means the synthesiser mispronounced
   something, the fit loop pushed the pace past intelligibility, or assembly put the clip in
   the wrong place -- all three are failures a listener would notice, and all three are
   invisible to the earlier stages, which only ever see their own output.

   The dub is re-transcribed *after assembly*, not clip by clip, precisely so that assembly
   bugs (a clip placed at the wrong offset, a clip mixed into its neighbour, a clip dropped)
   show up in this number instead of passing unnoticed.

2. **Timing fidelity.** Per-segment target vs achieved duration from M4, reported as
   mean/p50/p95 and a count over the 5% threshold the project brief sets.

3. **Coverage.** Segment counts at every hop -- ASR -> translate -> TTS -> assemble -> QC.
   An off-by-one here silently destroys sync, so the counts are asserted, not logged.

A CAVEAT THIS REPORT STATES RATHER THAN HIDES
mayura:v1 keeps English loanwords in Latin script ("card", "shuffle"), while Saaras
transcribes the spoken result in Devanagari. Those tokens therefore count as substitutions
no matter how well they were pronounced. The number of such tokens is measured per segment
and reported alongside the WER, so the inflation is visible and quantified instead of being
argued away.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jiwer

from .audio import (
    extract_audio_16k_mono,
    probe_duration_s,
    write_wav_slice,
)
from .config import (
    ASR_MAX_AUDIO_S,
    DUBBED_AUDIO_NAME,
    QC_DRIFT_WEIGHT,
    QC_WER_WEIGHT,
    Config,
)
from .metrics import MetricsCollector, percentile
from .sarvam_client import SarvamClient
from .stages.assemble import REPORT_NAME as ASSEMBLE_REPORT_NAME
from .stages.tts import REPORT_NAME as TTS_REPORT_NAME

logger = logging.getLogger(__name__)

REPORT_JSON_NAME = "qc_report.json"
REPORT_MD_NAME = "qc_report.md"
QC_AUDIO_DIR_NAME = "qc_slices"
QC_WAV_NAME = "dubbed_16k_mono.wav"

#: Punctuation stripped before scoring, including the Devanagari danda. Punctuation is not
#: spoken, so leaving it in would charge the synthesiser for the transcriber's full stops.
_PUNCT_RE = re.compile(r"[.,!?;:\"'`()\[\]{}<>/\\|~@#$%^&*_+=—–\-।॥]")
_WHITESPACE_RE = re.compile(r"\s+")

#: A token counts as Latin-script if any of its letters are; used to quantify the
#: script-mismatch penalty rather than to excuse it.
_LATIN_RE = re.compile(r"[A-Za-z]")

#: Stand-in for a window where nothing was transcribed. jiwer rejects an empty hypothesis
#: inside a corpus, and dropping the segment instead would quietly remove the worst case
#: from the total, so it is scored as one token that cannot match any reference word.
EMPTY_HYPOTHESIS_TOKEN = "qc-empty-hypothesis"


class QcError(RuntimeError):
    """Raised when the pipeline's output cannot be scored."""


# --------------------------------------------------------------------------------------
# Text normalisation and scoring
# --------------------------------------------------------------------------------------

def normalise_for_scoring(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace before WER/CER scoring."""
    # NFC first: Devanagari can encode the same grapheme as a precomposed character or as a
    # base plus a combining mark, and the two forms would otherwise score as different.
    normalised = unicodedata.normalize("NFC", text)
    normalised = _PUNCT_RE.sub(" ", normalised.lower())
    return _WHITESPACE_RE.sub(" ", normalised).strip()


def count_latin_tokens(text: str) -> int:
    """Count whitespace tokens containing Latin letters -- the ones the ASR renders in Devanagari."""
    return sum(1 for token in normalise_for_scoring(text).split() if _LATIN_RE.search(token))


def score_pair(reference: str, hypothesis: str) -> dict[str, Any]:
    """Compute WER and CER for one reference/hypothesis pair, with the raw edit counts."""
    ref = normalise_for_scoring(reference)
    hyp = normalise_for_scoring(hypothesis)

    if not ref:
        raise QcError("cannot score against an empty reference; the translation is missing")

    if not hyp:
        # Nothing was transcribed where a line should be: every reference word was deleted.
        # jiwer handles this, but stating it explicitly keeps the failure legible.
        words = len(ref.split())
        return {
            "wer": 1.0, "cer": 1.0,
            "hits": 0, "substitutions": 0, "deletions": words, "insertions": 0,
            "reference_words": words, "hypothesis_words": 0,
            "empty_hypothesis": True,
        }

    words = jiwer.process_words(ref, hyp)
    characters = jiwer.process_characters(ref, hyp)
    return {
        "wer": float(words.wer),
        "cer": float(characters.cer),
        "hits": int(words.hits),
        "substitutions": int(words.substitutions),
        "deletions": int(words.deletions),
        "insertions": int(words.insertions),
        "reference_words": len(ref.split()),
        "hypothesis_words": len(hyp.split()),
        "empty_hypothesis": False,
    }


def corpus_scores(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float]:
    """Aggregate WER/CER over every segment at once -- weighted by length, not a mean of means."""
    refs = [normalise_for_scoring(r) for r in references]
    hyps = [normalise_for_scoring(h) for h in hypotheses]

    kept = [(r, h) for r, h in zip(refs, hyps) if r]
    if not kept:
        raise QcError("no segment has a non-empty reference translation to score against")

    # jiwer rejects an empty hypothesis inside a list, so an unheard segment is scored as a
    # single unmatchable token; that keeps it in the corpus total as a full error instead of
    # dropping it, which would flatter the result.
    ref_list = [r for r, _ in kept]
    hyp_list = [h if h else EMPTY_HYPOTHESIS_TOKEN for _, h in kept]

    words = jiwer.process_words(ref_list, hyp_list)
    characters = jiwer.process_characters(ref_list, hyp_list)
    return {
        "wer": float(words.wer),
        "cer": float(characters.cer),
        "hits": int(words.hits),
        "substitutions": int(words.substitutions),
        "deletions": int(words.deletions),
        "insertions": int(words.insertions),
        "reference_words": sum(len(r.split()) for r in ref_list),
    }


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------

@dataclass
class SegmentScore:
    """Everything measured about one segment of the finished dub."""

    segment_id: int
    start_s: float
    end_s: float
    source_text: str
    translation: str
    back_transcript: str
    wer: float
    cer: float
    edits: dict[str, Any]
    target_duration_s: float
    achieved_duration_s: float
    abs_drift_pct: float
    signed_drift_pct: float
    final_pace: float
    attempts: int
    clamped: bool
    converged: bool
    fill_pct: float
    overlap_s: float
    latin_tokens_in_reference: int

    @property
    def timestamp(self) -> str:
        """The segment's start as mm:ss.mmm, so it can be found in the video immediately."""
        minutes, seconds = divmod(self.start_s, 60.0)
        return f"{int(minutes):02d}:{seconds:06.3f}"

    def combined_score(self, *, wer_weight: float, drift_weight: float) -> float:
        """Rank score combining semantic and timing error; higher is worse."""
        # WER and drift are different units, so they are combined explicitly with stated
        # weights rather than by an implicit convention hidden in a sort key.
        return wer_weight * self.wer + drift_weight * (self.abs_drift_pct / 100.0)

    def reasons(self, *, drift_threshold_pct: float) -> list[str]:
        """Plain-language reasons this segment was flagged, each backed by its own number."""
        found: list[str] = []
        if self.edits.get("empty_hypothesis"):
            found.append("nothing was transcribed in this window -- the dub may be silent here")
        if self.wer >= 0.5:
            found.append(f"WER {self.wer:.2f}: most words were not heard as written")
        elif self.wer > 0.0:
            found.append(f"WER {self.wer:.2f} ({self.edits['substitutions']} substitution(s), "
                         f"{self.edits['deletions']} deletion(s), "
                         f"{self.edits['insertions']} insertion(s))")
        if self.abs_drift_pct > drift_threshold_pct:
            direction = "overruns" if self.signed_drift_pct > 0 else "underruns"
            found.append(f"timing {direction} its window by {self.abs_drift_pct:.1f}% "
                         f"({self.achieved_duration_s:.2f}s of {self.target_duration_s:.2f}s)")
        if self.clamped:
            found.append(f"pace clamped at {self.final_pace:.2f}: the window cannot hold this "
                         f"line naturally")
        if self.overlap_s > 0:
            found.append(f"overlaps the next segment by {self.overlap_s:.3f}s")
        if self.latin_tokens_in_reference:
            found.append(f"{self.latin_tokens_in_reference} Latin-script token(s) in the "
                         f"reference are transcribed in Devanagari and count as errors "
                         f"regardless of pronunciation")
        return found

    def to_dict(self, *, wer_weight: float, drift_weight: float,
                drift_threshold_pct: float) -> dict[str, Any]:
        """Serialise the full per-segment score."""
        return {
            "segment_id": self.segment_id,
            "timestamp": self.timestamp,
            "window": [round(self.start_s, 3), round(self.end_s, 3)],
            "source_text": self.source_text,
            "translation": self.translation,
            "back_transcript": self.back_transcript,
            "wer": round(self.wer, 6),
            "cer": round(self.cer, 6),
            "edits": self.edits,
            "latin_tokens_in_reference": self.latin_tokens_in_reference,
            "target_duration_s": round(self.target_duration_s, 6),
            "achieved_duration_s": round(self.achieved_duration_s, 6),
            "abs_drift_pct": round(self.abs_drift_pct, 4),
            "signed_drift_pct": round(self.signed_drift_pct, 4),
            "final_pace": round(self.final_pace, 4),
            "attempts": self.attempts,
            "clamped": self.clamped,
            "converged": self.converged,
            "fill_pct": round(self.fill_pct, 4),
            "overlap_with_next_s": round(self.overlap_s, 6),
            "combined_score": round(
                self.combined_score(wer_weight=wer_weight, drift_weight=drift_weight), 6),
            "reasons": self.reasons(drift_threshold_pct=drift_threshold_pct),
        }


# --------------------------------------------------------------------------------------
# Back-transcription
# --------------------------------------------------------------------------------------

def back_transcribe_window(
    wav_16k_path: Path,
    slice_path: Path,
    start_s: float,
    end_s: float,
    config: Config,
    client: SarvamClient,
) -> tuple[str, list[str]]:
    """Transcribe one window of the dubbed track in the target language; returns (text, ids).

    Windows longer than the 30 s REST cap are split into equal sub-cap pieces and their
    transcripts joined. M2 keeps segments under --max-segment-s (15 s by default), so this
    path only opens if that limit was raised.
    """
    duration_s = end_s - start_s
    if duration_s <= 0:
        raise QcError(f"cannot transcribe a window of {duration_s:.3f}s")

    pieces = 1
    if duration_s > ASR_MAX_AUDIO_S:
        pieces = int(duration_s // ASR_MAX_AUDIO_S) + 1
        logger.warning(
            "QC window %.3f-%.3fs is %.3fs, over the %.0fs REST cap; splitting into %d "
            "piece(s) at fixed offsets (these cuts are not silence-aligned)",
            start_s, end_s, duration_s, ASR_MAX_AUDIO_S, pieces,
        )

    texts: list[str] = []
    request_ids: list[str] = []
    step = duration_s / pieces

    for piece in range(pieces):
        piece_start = start_s + piece * step
        piece_end = min(end_s, piece_start + step)
        piece_path = (
            slice_path if pieces == 1
            else slice_path.with_name(f"{slice_path.stem}_p{piece:02d}{slice_path.suffix}")
        )
        sliced_s = write_wav_slice(wav_16k_path, piece_path, piece_start, piece_end)

        payload = client.speech_to_text(
            piece_path,
            model=config.asr_model,
            mode="transcribe",
            # The dub is in the TARGET language; transcribing it as the source would
            # measure Saaras's language detection, not the dub's fidelity.
            language_code=config.target_lang,
            audio_duration_s=sliced_s,
            stage="qc",
        )
        texts.append((payload.get("transcript") or "").strip())
        if payload.get("request_id"):
            request_ids.append(str(payload["request_id"]))

    return " ".join(t for t in texts if t).strip(), request_ids


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_qc(
    config: Config,
    client: SarvamClient,
    metrics: MetricsCollector,
) -> dict[str, Any]:
    """Back-transcribe and score the finished dub; writes qc_report.md, qc_report.json, metrics."""
    segments_path = config.output_dir / "segments.json"
    tts_report_path = config.output_dir / TTS_REPORT_NAME
    assemble_report_path = config.output_dir / ASSEMBLE_REPORT_NAME
    dubbed_path = config.output_dir / DUBBED_AUDIO_NAME

    for required, stage in (
        (segments_path, "translate"), (tts_report_path, "tts"),
        (assemble_report_path, "assemble"), (dubbed_path, "assemble"),
    ):
        if not required.exists():
            raise QcError(
                f"{required} not found. Run the {stage} stage first "
                f"(python -m src.pipeline --input <video> --stage {stage})."
            )

    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    tts_report = json.loads(tts_report_path.read_text(encoding="utf-8"))
    assemble_report = json.loads(assemble_report_path.read_text(encoding="utf-8"))

    fits = {int(f["segment_id"]): f for f in tts_report.get("per_segment", [])}
    placements = {int(p["segment_id"]): p for p in assemble_report.get("placements", [])}
    overlap_by_id = {
        int(o["segment_ids"][0]): float(o["overlap_s"])
        for o in assemble_report.get("overlaps", {}).get("detail", [])
    }

    scored_ids = sorted(placements)
    if not scored_ids:
        raise QcError(f"{assemble_report_path} lists no placed clips to score")

    # --- prepare the audio Saaras wants -------------------------------------------------
    wav_16k = config.output_dir / QC_WAV_NAME
    extract_audio_16k_mono(dubbed_path, wav_16k)
    dub_duration_s = probe_duration_s(wav_16k)
    slice_dir = config.output_dir / QC_AUDIO_DIR_NAME
    slice_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "QC: back-transcribing %d window(s) of %s (%.3fs) as %s",
        len(scored_ids), dubbed_path.name, dub_duration_s, config.target_lang,
    )

    # --- score every segment ------------------------------------------------------------
    by_id = {int(s["id"]): s for s in segments}
    scores: list[SegmentScore] = []
    request_ids: list[str] = []

    for segment_id in scored_ids:
        segment = by_id.get(segment_id)
        if segment is None:
            raise QcError(
                f"segment {segment_id} was placed on the timeline but is absent from "
                f"{segments_path}; the stages disagree about what was dubbed"
            )
        translation = (segment.get("translation") or "").strip()
        if not translation:
            raise QcError(
                f"segment {segment_id} has no translation to score the dub against; "
                f"re-run --stage translate"
            )

        placement = placements[segment_id]
        window_start = float(placement["window"][0])
        window_end = min(float(placement["window"][1]), dub_duration_s)

        transcript, ids = back_transcribe_window(
            wav_16k, slice_dir / f"qc_seg_{segment_id:03d}.wav",
            window_start, window_end, config, client,
        )
        request_ids.extend(ids)

        edits = score_pair(translation, transcript)
        fit = fits.get(segment_id, {})
        target = float(fit.get("target_duration_s") or (window_end - window_start))
        achieved = float(fit.get("final_duration_s") or placement["duration_s"])
        signed = ((achieved - target) / target * 100.0) if target > 0 else 0.0

        score = SegmentScore(
            segment_id=segment_id,
            start_s=window_start,
            end_s=float(placement["window"][1]),
            source_text=(segment.get("text") or "").strip(),
            translation=translation,
            back_transcript=transcript,
            wer=edits["wer"],
            cer=edits["cer"],
            edits=edits,
            target_duration_s=target,
            achieved_duration_s=achieved,
            abs_drift_pct=abs(signed),
            signed_drift_pct=signed,
            final_pace=float(fit.get("final_pace") or 1.0),
            attempts=int(fit.get("attempts") or 0),
            clamped=bool(fit.get("clamped")),
            converged=bool(fit.get("converged")),
            fill_pct=float(placement.get("fill_pct") or 0.0),
            overlap_s=overlap_by_id.get(segment_id, 0.0),
            latin_tokens_in_reference=count_latin_tokens(translation),
        )
        scores.append(score)
        logger.info(
            "  segment %d [%s]: WER %.3f  CER %.3f  drift %+.1f%%  (%d ref word(s))",
            segment_id, score.timestamp, score.wer, score.cer, score.signed_drift_pct,
            edits["reference_words"],
        )

    # --- aggregate -----------------------------------------------------------------------
    corpus = corpus_scores([s.translation for s in scores], [s.back_transcript for s in scores])
    drifts = [s.abs_drift_pct for s in scores]
    timing = {
        "mean_abs_drift_pct": round(sum(drifts) / len(drifts), 4),
        "p50_abs_drift_pct": round(percentile(drifts, 50), 4),
        "p95_abs_drift_pct": round(percentile(drifts, 95), 4),
        "max_abs_drift_pct": round(max(drifts), 4),
        "threshold_pct": config.qc_drift_threshold_pct,
        "segments_over_threshold": sum(
            1 for d in drifts if d > config.qc_drift_threshold_pct),
        "segments_overrunning": sum(1 for s in scores if s.signed_drift_pct > 0),
        "segments_underrunning": sum(1 for s in scores if s.signed_drift_pct < 0),
        "small_sample": len(scores) < 20,
    }

    coverage = _coverage(
        config=config, segments=segments, tts_report=tts_report,
        assemble_report=assemble_report, scored=len(scores),
    )

    ranked = sorted(
        scores,
        key=lambda s: s.combined_score(
            wer_weight=QC_WER_WEIGHT, drift_weight=QC_DRIFT_WEIGHT),
        reverse=True,
    )
    flagged = [
        s for s in ranked
        if s.wer > 0 or s.abs_drift_pct > config.qc_drift_threshold_pct or s.overlap_s > 0
    ]

    report = {
        "dubbed_audio_path": str(dubbed_path),
        "back_transcription": {
            "model": config.asr_model,
            "mode": "transcribe",
            "language_code": config.target_lang,
            "windows_transcribed": len(scores),
            "request_ids": request_ids,
            "wav_16k_path": str(wav_16k),
        },
        "semantic_fidelity": {
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in corpus.items()},
            "scoring": (
                "reference = the stage-3 translation; hypothesis = Saaras transcribing the "
                "assembled dub in the target language. Both are NFC-normalised, lowercased, "
                "and stripped of punctuation before scoring."
            ),
            "latin_tokens_in_references": sum(s.latin_tokens_in_reference for s in scores),
            "latin_token_caveat": (
                "mayura:v1 leaves English loanwords in Latin script; Saaras transcribes them "
                "in Devanagari. Those tokens count as substitutions however well they were "
                "pronounced, so the WER above is an upper bound on the real error."
            ),
        },
        "timing_fidelity": timing,
        "coverage": coverage,
        "flagging": {
            "wer_weight": QC_WER_WEIGHT,
            "drift_weight": QC_DRIFT_WEIGHT,
            "drift_threshold_pct": config.qc_drift_threshold_pct,
            "flagged_segments": len(flagged),
            "top_n_reported": min(config.qc_flag_top_n, len(ranked)),
        },
        "segments": [
            s.to_dict(wer_weight=QC_WER_WEIGHT, drift_weight=QC_DRIFT_WEIGHT,
                      drift_threshold_pct=config.qc_drift_threshold_pct)
            for s in scores
        ],
        "ranked_worst": [
            s.to_dict(wer_weight=QC_WER_WEIGHT, drift_weight=QC_DRIFT_WEIGHT,
                      drift_threshold_pct=config.qc_drift_threshold_pct)
            for s in ranked[:config.qc_flag_top_n]
        ],
    }

    json_path = config.output_dir / REPORT_JSON_NAME
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = config.output_dir / REPORT_MD_NAME
    md_path.write_text(
        _render_markdown(report, ranked[:config.qc_flag_top_n], scores, config),
        encoding="utf-8",
    )

    metrics.qc = {
        "wer": round(corpus["wer"], 6),
        "cer": round(corpus["cer"], 6),
        "flagged_segments": len(flagged),
        "segments_scored": len(scores),
        "mean_abs_drift_pct": timing["mean_abs_drift_pct"],
        "p95_abs_drift_pct": timing["p95_abs_drift_pct"],
        "segments_over_drift_threshold": timing["segments_over_threshold"],
        "coverage_passed": coverage["all_passed"],
    }

    failures = [name for name, check in coverage["checks"].items() if not check["passed"]]
    if failures:
        raise QcError(
            f"coverage failed at {len(failures)} hop(s): {failures}. Segment counts must "
            f"match at every stage or sync is already broken. Detail in {json_path}."
        )

    logger.info("QC: corpus WER %.4f, CER %.4f over %d segment(s); %d flagged",
                corpus["wer"], corpus["cer"], len(scores), len(flagged))
    logger.info("QC: wrote %s and %s", md_path, json_path)
    return report


def _coverage(
    *,
    config: Config,
    segments: Sequence[dict[str, Any]],
    tts_report: dict[str, Any],
    assemble_report: dict[str, Any],
    scored: int,
) -> dict[str, Any]:
    """Assert the segment count survives every hop; a mismatch is a silent sync bug."""
    asr_count = len(segments)
    translated = sum(1 for s in segments if (s.get("translation") or "").strip())
    synthesised = len(tts_report.get("files", []))
    placed = len(assemble_report.get("placements", []))

    checks: dict[str, dict[str, Any]] = {
        "asr_to_translate": {
            "passed": translated == asr_count,
            "expected": asr_count, "actual": translated,
            "hop": "every ASR segment must carry a translation",
        },
        "tts_to_assemble": {
            "passed": placed == synthesised,
            "expected": synthesised, "actual": placed,
            "hop": "every synthesised clip must be placed on the timeline",
        },
        "assemble_to_qc": {
            "passed": scored == placed,
            "expected": placed, "actual": scored,
            "hop": "every placed clip must be scored",
        },
    }

    # --max-segments deliberately truncates the run, so the translate -> TTS hop is only
    # required to match when the whole clip was processed.
    expected_synth = min(config.max_segments, translated) if config.max_segments else translated
    checks["translate_to_tts"] = {
        "passed": synthesised == expected_synth,
        "expected": expected_synth, "actual": synthesised,
        "hop": "every translated segment must be synthesised"
              + (f" (capped by --max-segments {config.max_segments})"
                 if config.max_segments else ""),
    }

    empty_text = [int(s["id"]) for s in segments if not (s.get("text") or "").strip()]
    checks["no_empty_segments"] = {
        "passed": not empty_text,
        "expected": "no empty transcripts", "actual": empty_text or "none",
        "hop": "an empty segment would synthesise silence into its window",
    }

    for name, check in checks.items():
        logger.info("  coverage %-20s %s (expected %s, got %s)",
                    name, "PASS" if check["passed"] else "FAIL",
                    check["expected"], check["actual"])

    return {
        "counts": {
            "asr_segments": asr_count,
            "translated": translated,
            "synthesised": synthesised,
            "placed": placed,
            "scored": scored,
        },
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks.values()),
    }


def _render_markdown(
    report: dict[str, Any],
    worst: Sequence[SegmentScore],
    scores: Sequence[SegmentScore],
    config: Config,
) -> str:
    """Render qc_report.md: the headline scores, then the ranked worst segments with timestamps."""
    semantic = report["semantic_fidelity"]
    timing = report["timing_fidelity"]
    coverage = report["coverage"]

    lines: list[str] = [
        "# QC report",
        "",
        f"Scored `{report['dubbed_audio_path']}` against the stage-3 translations. "
        f"Every number below is measured from this run.",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Segments scored | {len(scores)} |",
        f"| Corpus WER | {semantic['wer']:.4f} |",
        f"| Corpus CER | {semantic['cer']:.4f} |",
        f"| Mean absolute timing drift | {timing['mean_abs_drift_pct']:.2f}% |",
        f"| p95 absolute timing drift | {timing['p95_abs_drift_pct']:.2f}% |",
        f"| Segments over the {timing['threshold_pct']:.0f}% drift threshold | "
        f"{timing['segments_over_threshold']} of {len(scores)} |",
        f"| Segments flagged | {report['flagging']['flagged_segments']} |",
        f"| Coverage | {'PASS' if coverage['all_passed'] else 'FAIL'} |",
        "",
    ]

    if timing.get("small_sample"):
        lines += [
            f"> **Small sample.** {len(scores)} segments cannot support a meaningful p95. "
            f"It is reported because the brief asks for it, not because it is robust.",
            "",
        ]

    lines += [
        "## How semantic fidelity is measured",
        "",
        semantic["scoring"],
        "",
        f"**Caveat, quantified:** {semantic['latin_token_caveat']} "
        f"{semantic['latin_tokens_in_references']} of "
        f"{semantic['reference_words']} reference words are Latin-script.",
        "",
        "## Coverage",
        "",
        "| Hop | Expected | Actual | Result |",
        "| --- | --- | --- | --- |",
    ]
    for name, check in coverage["checks"].items():
        lines.append(
            f"| `{name}` | {check['expected']} | {check['actual']} | "
            f"{'PASS' if check['passed'] else '**FAIL**'} |"
        )

    lines += [
        "",
        "## Worst segments, ranked",
        "",
        f"Ranked by `{QC_WER_WEIGHT} x WER + {QC_DRIFT_WEIGHT} x (absolute drift / 100)`. "
        f"Jump to the timestamp in the video and listen.",
        "",
        "| Rank | Timestamp | Segment | WER | CER | Drift | Pace | Score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, score in enumerate(worst, start=1):
        lines.append(
            f"| {rank} | `{score.timestamp}` | {score.segment_id} | {score.wer:.3f} | "
            f"{score.cer:.3f} | {score.signed_drift_pct:+.1f}% | {score.final_pace:.2f} | "
            f"{score.combined_score(wer_weight=QC_WER_WEIGHT, drift_weight=QC_DRIFT_WEIGHT):.3f} |"
        )

    lines.append("")
    for rank, score in enumerate(worst, start=1):
        lines += [
            f"### {rank}. Segment {score.segment_id} at `{score.timestamp}` "
            f"({score.start_s:.3f}s - {score.end_s:.3f}s)",
            "",
            f"- **Source:** {score.source_text}",
            f"- **Translation (what we asked Bulbul to say):** {score.translation}",
            f"- **Back-transcript (what Saaras hears in the dub):** "
            f"{score.back_transcript or '_nothing transcribed_'}",
            f"- **WER {score.wer:.3f} / CER {score.cer:.3f}** "
            f"({score.edits['substitutions']}S {score.edits['deletions']}D "
            f"{score.edits['insertions']}I over {score.edits['reference_words']} words)",
            f"- **Timing:** {score.achieved_duration_s:.3f}s spoken into a "
            f"{score.target_duration_s:.3f}s window ({score.signed_drift_pct:+.1f}%), "
            f"pace {score.final_pace:.2f} after {score.attempts} attempt(s)"
            + (", **clamped**" if score.clamped else ""),
            "",
            "Why it is flagged:",
            "",
        ]
        for reason in score.reasons(drift_threshold_pct=config.qc_drift_threshold_pct):
            lines.append(f"- {reason}")
        lines.append("")

    lines += [
        "## Every segment",
        "",
        "| Segment | Timestamp | WER | CER | Target | Achieved | Drift | Fill | Pace |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for score in scores:
        lines.append(
            f"| {score.segment_id} | `{score.timestamp}` | {score.wer:.3f} | {score.cer:.3f} | "
            f"{score.target_duration_s:.2f}s | {score.achieved_duration_s:.2f}s | "
            f"{score.signed_drift_pct:+.1f}% | {score.fill_pct:.0f}% | {score.final_pace:.2f} |"
        )

    lines.append("")
    return "\n".join(lines)
