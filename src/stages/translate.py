"""
Translate stage: per-segment en-IN -> hi-IN via Sarvam's POST /translate.

Input:  output/segments.json from M2, as [{id, start, end, text, speaker?}].
Output: the same file extended in place to
        [{id, start, end, text, translation, speaker, expansion}], plus
        output/translate_report.json carrying the batching and expansion measurements.

Three things this stage has to get right, in order of how badly they break the dub:

1. **Mapping.** An off-by-one between segments and translations silently destroys A/V
   sync several stages later, where it is nearly impossible to diagnose. Every invariant
   -- count in == count out, ids unchanged, timestamps unchanged, no empty translation --
   is asserted and raises `MappingError`. Nothing here degrades to a warning.

2. **Context.** `/translate` has no context parameter: the confirmed request schema is
   `{input, source_language_code, target_language_code, speaker_gender, mode, model,
   output_script, numerals_format}` and nothing else (docs/api-notes.md). Discourse
   context can therefore only be carried *inside* `input`. This module does that by
   sending a window of neighbouring segments in the same request as numbered lines and
   discarding the neighbours' translations. "Remember it." and "That's amazing." are
   pronoun-bearing fragments whose Hindi rendering depends on what preceded them, so the
   window is the difference between a correct dub and a plausible-sounding wrong one.

3. **Batching.** Several segments share one request, again as numbered lines. This is a
   wall-clock optimisation, *not* a cost one: /translate is billed per input character,
   so batching adds the marker and context characters rather than saving money (measured
   both ways in docs/m3-results.md). Because packing many segments into one reply means
   trusting the model to preserve line structure, every batch reply is parsed strictly and
   a batch that does not round-trip cleanly falls back to one request per segment. The
   fallback is logged and counted, never silent.

Wire protocol for a multi-line request::

    [1] Okay, Jay, pick a card.
    [2] Remember it. Right, Jay, shuffle the cards.

The reply must carry markers [1]..[N] exactly once each, in order, each with non-empty
text. A single-line request is sent bare, with no markers, because there is nothing to
disambiguate and the markers would only add cost and parse risk.

Example output segment (measured, samples/test_clip.mp4)::

    {"id": 0, "start": 0.0, "end": 5.475, "text": "Okay, Jay, pick a card.",
     "translation": "ओके, जे, एक कार्ड चुनो।", "speaker": null,
     "expansion": {"char_ratio": ..., "syllable_ratio_est": ...}}
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..config import (
    TRANSLATE_BATCH_FILL,
    TRANSLATE_MODEL_MAX_CHARS,
    Config,
)
from ..metrics import MetricsCollector
from ..sarvam_client import SarvamClient

logger = logging.getLogger(__name__)

#: Name of the file this stage reads and rewrites.
SEGMENTS_NAME = "segments.json"
REPORT_NAME = "translate_report.json"

#: Line marker for multi-segment requests, and the pattern that must round-trip it.
#:
#: Scanned anywhere in the reply rather than anchored to the start of a line, because
#: mayura:v1 returns every line joined onto one (measured 2026-07-29, see
#: docs/api-notes.md). `\d` is Unicode-aware and `int()` accepts Devanagari digits, which
#: matters because `numerals_format=native` rewrites the markers themselves as [१][२][३].
_MARKER_TEMPLATE = "[{n}] {text}"
_MARKER_RE = re.compile(r"[\[\(]\s*(\d+)\s*[\]\)]")

#: Rough cost of one marker in characters, used only when planning how much fits.
_MARKER_OVERHEAD_CHARS = 6


class TranslateError(RuntimeError):
    """Raised when the translate stage cannot proceed safely."""


class MappingError(TranslateError):
    """Raised when segments and translations stop corresponding one-to-one.

    This is deliberately fatal. A dropped or duplicated translation shifts every
    downstream segment's audio onto the wrong timestamp, and the failure only becomes
    visible as an unexplained loss of lip sync in the finished video.
    """


# --------------------------------------------------------------------------------------
# 1. Text measurement -- expansion ratio and syllable estimation
# --------------------------------------------------------------------------------------

#: Devanagari code points, used by the akshara counter below.
_DEV_VIRAMA = "्"
_DEV_NUKTA = "़"
_DEV_INDEPENDENT_VOWELS = set(range(0x0904, 0x0915)) | {0x0960, 0x0961}
_DEV_MATRAS = (
    set(range(0x093A, 0x0940 + 1))
    | set(range(0x0941, 0x094C + 1))
    | {0x094E, 0x094F, 0x0955, 0x0956, 0x0957, 0x0962, 0x0963}
)
_DEV_CONSONANTS = (
    set(range(0x0915, 0x0939 + 1))
    | set(range(0x0958, 0x095F + 1))
    | set(range(0x0978, 0x097F + 1))
)

_EN_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")

#: Characters that hold a word together. Python's `\w` is not usable here: the Devanagari
#: virama and anusvara are Mn (non-spacing marks) and therefore not alphanumeric, so `\w+`
#: splits "कार्ड" into "कार" + "ड" and the akshara counter then scores it 2 instead of 1.
_JOINERS = frozenset("'’‌‍")


def _tokenize_words(text: str) -> list[str]:
    """Split text into words, keeping combining marks and intra-word apostrophes attached."""
    words: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalpha() or unicodedata.category(char) in {"Mn", "Mc"} or char in _JOINERS:
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    # A leading/trailing joiner is punctuation, not part of the word ("'quoted'").
    return [w for w in (word.strip("".join(_JOINERS)) for word in words) if w]


def _syllables_in_english_word(word: str) -> int:
    """Estimate the syllables in one Latin-script word from its vowel groups."""
    lowered = word.lower()
    count = len(_EN_VOWEL_GROUP_RE.findall(lowered))
    # "card" -> 1, but "care" would count 2 without this: a final 'e' after a
    # consonant is almost always silent in English.
    if count > 1 and lowered.endswith("e") and not lowered.endswith(("le", "ee", "ye")):
        count -= 1
    # Past-tense "-ed" is only its own syllable after t or d: "amazed" is 2, "wanted" 3.
    elif count > 1 and lowered.endswith("ed") and not lowered.endswith(("ted", "ded")):
        count -= 1
    return max(1, count)


def _devanagari_aksharas(word: str) -> list[bool]:
    """Split a Devanagari word into aksharas; True where the vowel is written explicitly.

    False means the akshara carries only the *inherent* schwa, which Hindi frequently
    does not pronounce -- see `_apply_schwa_deletion`.
    """
    codes = [ord(ch) for ch in word]
    units: list[bool] = []
    position, length = 0, len(codes)

    while position < length:
        code = codes[position]
        if code in _DEV_INDEPENDENT_VOWELS:
            units.append(True)
            position += 1
            continue
        if code not in _DEV_CONSONANTS:
            position += 1
            continue

        # Look past a nukta to find what actually follows the consonant.
        after = position + 1
        while after < length and codes[after] == ord(_DEV_NUKTA):
            after += 1

        if after < length and codes[after] == ord(_DEV_VIRAMA):
            # Part of a conjunct: no vowel of its own, so the following consonant is what
            # opens the akshara. "कार्ड" is ka + (rD), two units, not three.
            position = after + 1
            continue
        if after < length and codes[after] in _DEV_MATRAS:
            units.append(True)
            position = after + 1
            continue
        units.append(False)  # inherent schwa
        position = after

    return units


def _apply_schwa_deletion(units: list[bool]) -> int:
    """Count pronounced syllables after Hindi schwa deletion; `units` is from the parser.

    Hindi drops the inherent schwa in two positions, and skipping either makes every
    Hindi word score high, which would inflate the expansion ratio M4 is planned from:

      * word-finally -- "राम" is *raam*, not *raa-ma*;
      * medially between two vowel-bearing aksharas (the classic V-C-C-V rule) --
        "तुमने" is *tum-ne*, not *tu-ma-ne*, and "आपका" is *aap-kaa*, not *aa-pa-kaa*.

    Deletion runs right to left so that two adjacent schwas can never both be dropped:
    in "कमल" the final one goes, which then protects the medial one and leaves *ka-mal*.
    """
    if not units:
        return 0

    #: None marks a deleted schwa; True/False are explicit vowel / surviving schwa.
    state: list[bool | None] = list(units)
    if len(state) > 1 and state[-1] is False:
        state[-1] = None

    for index in range(len(state) - 2, 0, -1):
        if state[index] is False and state[index - 1] is not None and state[index + 1] is not None:
            state[index] = None

    return sum(1 for unit in state if unit is not None)


def _syllables_in_devanagari_word(word: str) -> int:
    """Estimate the pronounced syllables in one Devanagari word."""
    return max(1, _apply_schwa_deletion(_devanagari_aksharas(word)))


def count_syllables_english(text: str) -> int:
    """Estimate English syllables by counting vowel groups, minus silent word-final 'e'."""
    return sum(_syllables_in_english_word(w) for w in _tokenize_words(text))


def count_syllables_devanagari(text: str) -> int:
    """Estimate Hindi syllables as aksharas, applying word-final schwa deletion."""
    return sum(_syllables_in_devanagari_word(w) for w in _tokenize_words(text))


def count_syllables(text: str, language_code: str) -> int | None:
    """Estimate syllables word by word, dispatching on each word's script.

    Dispatching per word rather than per string is what makes the code-mixed modes
    measurable: "Jay, आपका card top card है" is half Latin and half Devanagari, and a
    whole-string script check would simply refuse it and report nothing.
    Returns None only when the text contains a script neither estimator covers.
    """
    if not text.strip():
        return 0

    total = 0
    unsupported = 0
    for word in _tokenize_words(text):
        script = _word_script(word)
        if script == "latin":
            total += _syllables_in_english_word(word)
        elif script == "devanagari":
            total += _syllables_in_devanagari_word(word)
        else:
            unsupported += 1

    if unsupported:
        # Refuse to report a number that silently omits part of the text.
        logger.debug("no syllable estimator for %d word(s) in %r (%s)",
                     unsupported, text[:40], language_code)
        return None
    return total


def _word_script(word: str) -> str:
    """Classify one word as 'latin', 'devanagari', or 'other' by its majority script."""
    letters = [ch for ch in word if ch.isalpha()]
    if not letters:
        return "other"
    names = [unicodedata.name(ch, "") for ch in letters]
    devanagari = sum(1 for n in names if n.startswith("DEVANAGARI"))
    latin = sum(1 for n in names if n.startswith("LATIN"))
    if devanagari >= latin and devanagari > 0:
        return "devanagari"
    if latin > 0:
        return "latin"
    return "other"


@dataclass(frozen=True)
class Expansion:
    """How much longer the translation is than its source, per segment."""

    source_chars: int
    target_chars: int
    source_syllables_est: int | None
    target_syllables_est: int | None
    #: Segment duration from M2, used to express the M4 implication in real time.
    duration_s: float

    @property
    def char_ratio(self) -> float:
        """Translated characters divided by source characters."""
        return self.target_chars / self.source_chars if self.source_chars else 0.0

    @property
    def syllable_ratio_est(self) -> float | None:
        """Estimated translated syllables divided by estimated source syllables."""
        if not self.source_syllables_est or self.target_syllables_est is None:
            return None
        return self.target_syllables_est / self.source_syllables_est

    @property
    def max_speedup_needed(self) -> float | None:
        """Speed-up M4 would need if the segment were wall-to-wall speech (upper bound).

        Under the stated assumption that Bulbul speaks Hindi at the same syllable rate
        the source speaker used for English, a segment carrying `syllable_ratio` times
        more syllables needs to be spoken that many times faster to occupy the same
        window. Any pause inside the segment reduces this proportionally, so it is an
        upper bound, not a prediction of the pace M4 will actually settle on.
        """
        return self.syllable_ratio_est

    def to_dict(self) -> dict[str, Any]:
        """Serialise for segments.json and the report."""
        from ..metrics import NOT_MEASURED

        syllable_ratio = self.syllable_ratio_est
        return {
            "source_chars": self.source_chars,
            "target_chars": self.target_chars,
            "char_ratio": round(self.char_ratio, 4),
            "source_syllables_est": (
                self.source_syllables_est if self.source_syllables_est is not None
                else NOT_MEASURED
            ),
            "target_syllables_est": (
                self.target_syllables_est if self.target_syllables_est is not None
                else NOT_MEASURED
            ),
            "syllable_ratio_est": (
                round(syllable_ratio, 4) if syllable_ratio is not None else NOT_MEASURED
            ),
            "source_syllables_per_s": (
                round(self.source_syllables_est / self.duration_s, 4)
                if self.source_syllables_est is not None and self.duration_s > 0
                else NOT_MEASURED
            ),
            "max_speedup_needed_est": (
                round(self.max_speedup_needed, 4)
                if self.max_speedup_needed is not None else NOT_MEASURED
            ),
        }


def measure_expansion(
    source_text: str,
    target_text: str,
    *,
    source_lang: str,
    target_lang: str,
    duration_s: float,
) -> Expansion:
    """Measure the source->target size change for one segment."""
    return Expansion(
        source_chars=len(source_text),
        target_chars=len(target_text),
        source_syllables_est=count_syllables(source_text, source_lang),
        target_syllables_est=count_syllables(target_text, target_lang),
        duration_s=duration_s,
    )


# --------------------------------------------------------------------------------------
# 2. Batch planning
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Batch:
    """One /translate request: the segments it answers for, plus context-only neighbours."""

    index: int
    #: Segment ids whose translations are kept. Never empty.
    payload_ids: list[int]
    #: Neighbouring segment ids sent for discourse context; their translations are dropped.
    context_ids: list[int]

    @property
    def line_ids(self) -> list[int]:
        """Every segment id in this request, in timeline order."""
        return sorted(self.payload_ids + self.context_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the report."""
        return {
            "index": self.index,
            "payload_ids": list(self.payload_ids),
            "context_ids": list(self.context_ids),
            "lines": len(self.line_ids),
        }


def plan_batches(
    segments: Sequence[dict[str, Any]],
    *,
    max_chars: int,
    max_segments: int,
    context_segments: int,
    fill: float = TRANSLATE_BATCH_FILL,
) -> list[Batch]:
    """Pack segments into requests under the character cap, each with a context window."""
    if not segments:
        return []
    if max_segments < 1:
        raise TranslateError(f"max_segments must be >= 1, got {max_segments}")

    budget = int(max_chars * fill)
    ids = [int(s["id"]) for s in segments]
    text_by_id = {int(s["id"]): str(s["text"]) for s in segments}

    groups: list[list[int]] = []
    current: list[int] = []
    used = 0

    for segment_id in ids:
        cost = len(text_by_id[segment_id]) + _MARKER_OVERHEAD_CHARS + 1
        if len(text_by_id[segment_id]) > max_chars:
            raise TranslateError(
                f"segment {segment_id} is {len(text_by_id[segment_id])} chars, over the "
                f"{max_chars}-char cap for this model even on its own; M2 should have "
                f"split it"
            )
        # A segment that cannot join the current group opens a new one. A group is never
        # left empty, so a single oversized-for-the-budget segment still gets its own
        # request rather than being dropped.
        if current and (used + cost > budget or len(current) >= max_segments):
            groups.append(current)
            current, used = [], 0
        current.append(segment_id)
        used += cost

    if current:
        groups.append(current)

    position = {segment_id: index for index, segment_id in enumerate(ids)}
    batches: list[Batch] = []

    for index, group in enumerate(groups):
        context: list[int] = []
        if context_segments > 0:
            first, last = position[group[0]], position[group[-1]]
            before = ids[max(0, first - context_segments):first]
            after = ids[last + 1:last + 1 + context_segments]
            # Context is only worth paying for at a batch seam; inside a batch the
            # neighbours are already in the same request.
            context = [i for i in before + after if i not in group]
            context = _trim_context_to_budget(context, group, text_by_id, budget)
        batches.append(Batch(index=index, payload_ids=list(group), context_ids=context))

    return batches


def _trim_context_to_budget(
    context: list[int],
    group: list[int],
    text_by_id: dict[int, str],
    budget: int,
) -> list[int]:
    """Drop the furthest context lines until the request fits inside the character budget."""
    used = sum(len(text_by_id[i]) + _MARKER_OVERHEAD_CHARS + 1 for i in group)
    kept: list[int] = []
    # Nearest neighbours first: the segment immediately before a fragment carries its
    # antecedent, the one three back rarely does.
    for segment_id in sorted(context, key=lambda i: min(abs(i - g) for g in group)):
        cost = len(text_by_id[segment_id]) + _MARKER_OVERHEAD_CHARS + 1
        if used + cost > budget:
            continue
        kept.append(segment_id)
        used += cost
    return sorted(kept)


# --------------------------------------------------------------------------------------
# 3. Wire protocol -- render and strictly parse the numbered-line format
# --------------------------------------------------------------------------------------

def render_batch(texts: Sequence[str]) -> str:
    """Render one request body: bare text for a single line, numbered markers otherwise."""
    if not texts:
        raise TranslateError("cannot render an empty batch")
    if len(texts) == 1:
        return texts[0].strip()
    return "\n".join(
        _MARKER_TEMPLATE.format(n=n, text=text.strip()) for n, text in enumerate(texts, 1)
    )


def parse_batch(reply: str, expected: int) -> list[str] | None:
    """Parse a numbered reply into exactly `expected` texts; None if it does not round-trip."""
    if expected < 1:
        raise TranslateError(f"expected must be >= 1, got {expected}")
    if expected == 1:
        text = reply.strip()
        return [text] if text else None

    markers = list(_MARKER_RE.finditer(reply))
    numbers = [int(m.group(1)) for m in markers]
    if numbers != list(range(1, expected + 1)):
        logger.warning(
            "batch reply did not round-trip: expected markers 1..%d, got %s",
            expected, numbers or "none",
        )
        return None

    texts: list[str] = []
    for position, marker in enumerate(markers):
        stop = markers[position + 1].start() if position + 1 < len(markers) else len(reply)
        # Collapse whitespace: a segment is one utterance, so any newline the model
        # introduced inside it is formatting, not content.
        texts.append(" ".join(reply[marker.end():stop].split()))

    if any(not text for text in texts):
        logger.warning("batch reply contained an empty line at position(s) %s",
                       [i + 1 for i, t in enumerate(texts) if not t])
        return None
    return texts


# --------------------------------------------------------------------------------------
# 4. Translation
# --------------------------------------------------------------------------------------

@dataclass
class TranslateStats:
    """Counters describing how the translations were actually obtained."""

    batches_planned: int = 0
    batches_parsed: int = 0
    batches_fell_back: int = 0
    requests_issued: int = 0
    single_line_requests: int = 0
    context_lines_sent: int = 0
    request_chars: int = 0
    payload_chars: int = 0
    fallback_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the report."""
        return {
            "batches_planned": self.batches_planned,
            "batches_parsed_cleanly": self.batches_parsed,
            "batches_fell_back_to_per_segment": self.batches_fell_back,
            "translate_requests_issued": self.requests_issued,
            "single_line_requests": self.single_line_requests,
            "context_lines_sent": self.context_lines_sent,
            "chars_sent_to_api": self.request_chars,
            "chars_of_actual_segment_text": self.payload_chars,
            "protocol_overhead_chars": self.request_chars - self.payload_chars,
            "fallback_reasons": list(self.fallback_reasons),
        }


def translate_segments(
    segments: Sequence[dict[str, Any]],
    config: Config,
    client: SarvamClient,
    *,
    model: str | None = None,
    mode: str | None = None,
) -> tuple[dict[int, str], list[Batch], TranslateStats]:
    """Translate every segment, batching where possible; returns {id: translation}."""
    resolved_model = model or config.translate_model
    resolved_mode = mode if mode is not None else config.translate_mode
    max_chars = TRANSLATE_MODEL_MAX_CHARS[resolved_model]

    text_by_id = {int(s["id"]): str(s["text"]) for s in segments}
    stats = TranslateStats()

    if config.translate_no_batch:
        batches = [
            Batch(index=i, payload_ids=[int(s["id"])], context_ids=[])
            for i, s in enumerate(segments)
        ]
        logger.info("--translate-no-batch: %d single-segment request(s)", len(batches))
    else:
        batches = plan_batches(
            segments,
            max_chars=max_chars,
            max_segments=config.translate_batch_segments,
            context_segments=config.translate_context_segments,
        )
    stats.batches_planned = len(batches)

    translations: dict[int, str] = {}

    for batch in batches:
        line_ids = batch.line_ids
        texts = [text_by_id[i] for i in line_ids]
        body = render_batch(texts)

        stats.context_lines_sent += len(batch.context_ids)
        logger.info(
            "batch %d: %d payload segment(s) %s + %d context %s (%d chars)",
            batch.index, len(batch.payload_ids), batch.payload_ids,
            len(batch.context_ids), batch.context_ids or "[]", len(body),
        )

        payload = client.translate(
            body,
            source_language_code=config.source_lang,
            target_language_code=config.target_lang,
            model=resolved_model,
            mode=resolved_mode,
            numerals_format="native",
        )
        stats.requests_issued += 1
        stats.request_chars += len(body)
        if len(line_ids) == 1:
            stats.single_line_requests += 1

        parsed = parse_batch(payload.get("translated_text", ""), len(line_ids))

        if parsed is None:
            # The model did not preserve the line structure. Rather than guess at an
            # alignment -- the exact class of bug that silently destroys sync -- redo this
            # batch one segment at a time, where the mapping is trivially correct.
            reason = (
                f"batch {batch.index} ({len(line_ids)} lines) did not round-trip; "
                f"retried as {len(batch.payload_ids)} single-segment request(s)"
            )
            logger.warning("%s", reason)
            stats.batches_fell_back += 1
            stats.fallback_reasons.append(reason)

            for segment_id in batch.payload_ids:
                single = text_by_id[segment_id]
                reply = client.translate(
                    single,
                    source_language_code=config.source_lang,
                    target_language_code=config.target_lang,
                    model=resolved_model,
                    mode=resolved_mode,
                    numerals_format="native",
                )
                stats.requests_issued += 1
                stats.single_line_requests += 1
                stats.request_chars += len(single)
                stats.payload_chars += len(single)
                translations[segment_id] = (reply.get("translated_text") or "").strip()
            continue

        stats.batches_parsed += 1
        for segment_id, translated in zip(line_ids, parsed):
            if segment_id in batch.payload_ids:
                translations[segment_id] = translated
                stats.payload_chars += len(text_by_id[segment_id])

    return translations, batches, stats


# --------------------------------------------------------------------------------------
# 5. Mapping assertions -- fatal by design
# --------------------------------------------------------------------------------------

def assert_mapping_is_sound(
    before: Sequence[dict[str, Any]],
    after: Sequence[dict[str, Any]],
    *,
    time_epsilon_s: float = 1e-6,
) -> dict[str, Any]:
    """Assert the translated segments correspond one-to-one with the input; returns evidence."""
    if len(before) != len(after):
        raise MappingError(
            f"segment count changed across the translate stage: {len(before)} in, "
            f"{len(after)} out. Every downstream timestamp would shift."
        )
    if not after:
        raise MappingError("no segments to translate; the ASR stage produced nothing")

    for index, (source, result) in enumerate(zip(before, after)):
        if source["id"] != result["id"]:
            raise MappingError(
                f"segment at position {index} changed id: {source['id']} -> {result['id']}"
            )
        for key in ("start", "end"):
            if abs(float(source[key]) - float(result[key])) > time_epsilon_s:
                raise MappingError(
                    f"segment {source['id']} {key} changed: {source[key]} -> {result[key]}; "
                    f"translation must never move a timestamp"
                )
        if str(source["text"]) != str(result["text"]):
            raise MappingError(
                f"segment {source['id']} source text changed across the stage"
            )
        translation = result.get("translation")
        if not isinstance(translation, str) or not translation.strip():
            raise MappingError(
                f"segment {source['id']} ({source['start']:.3f}-{source['end']:.3f}s, "
                f"text={source['text']!r}) has no translation"
            )

    ids = [r["id"] for r in after]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise MappingError(f"duplicate segment ids after translation: {duplicates}")
    if ids != sorted(ids):
        raise MappingError(f"segment ids are no longer in timeline order: {ids}")

    evidence = {
        "segments_in": len(before),
        "segments_out": len(after),
        "counts_match": True,
        "ids_unchanged": True,
        "timestamps_unchanged": True,
        "source_text_unchanged": True,
        "all_translations_non_empty": True,
        "ids": ids,
    }
    logger.info(
        "mapping assertions passed: %d in == %d out, ids and timestamps unchanged, "
        "%d non-empty translation(s)", len(before), len(after), len(after),
    )
    return evidence


# --------------------------------------------------------------------------------------
# 6. Orchestration
# --------------------------------------------------------------------------------------

def run_translate(
    config: Config,
    client: SarvamClient,
    metrics: MetricsCollector,
    *,
    segments_path: str | Path | None = None,
) -> dict[str, Any]:
    """Translate segments.json in place and write translate_report.json."""
    path = Path(segments_path) if segments_path else config.output_dir / SEGMENTS_NAME
    if not path.exists():
        raise TranslateError(
            f"{path} not found. Run the ASR stage first "
            f"(python -m src.pipeline --input <video> --stage asr)."
        )

    if config.source_lang == "unknown":
        raise TranslateError(
            "--source-lang unknown is the ASR auto-detect sentinel and /translate does not "
            "accept it. Re-run with an explicit source language (the ASR stage records the "
            "detected code as `language_code` in output/asr_report.json)."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise TranslateError(f"{path} does not contain a non-empty list of segments")

    # Snapshot the pre-translation state so the assertions compare against what was on
    # disk, not against an object this stage has already touched.
    before = [
        {"id": int(s["id"]), "start": float(s["start"]), "end": float(s["end"]),
         "text": str(s["text"])}
        for s in raw
    ]

    active = before
    if config.max_segments is not None and config.max_segments < len(before):
        active = before[:config.max_segments]
        logger.warning(
            "--max-segments %d: translating %d of %d segment(s)",
            config.max_segments, len(active), len(before),
        )

    logger.info(
        "translating %d segment(s) %s -> %s with %s (mode=%s)",
        len(active), config.source_lang, config.target_lang,
        config.translate_model, config.translate_mode,
    )

    translations, batches, stats = translate_segments(active, config, client)

    missing = [s["id"] for s in active if s["id"] not in translations]
    if missing:
        raise MappingError(
            f"{len(missing)} segment(s) came back without a translation: {missing}"
        )

    after: list[dict[str, Any]] = []
    expansions: dict[int, Expansion] = {}

    for original, source in zip(raw, before):
        segment_id = source["id"]
        record = dict(original)
        record["id"] = segment_id
        record["start"] = source["start"]
        record["end"] = source["end"]
        record["text"] = source["text"]

        if segment_id in translations:
            translation = translations[segment_id]
            expansion = measure_expansion(
                source["text"], translation,
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                duration_s=source["end"] - source["start"],
            )
            expansions[segment_id] = expansion
            record["translation"] = translation
            record["expansion"] = expansion.to_dict()
        else:
            # Only reachable under --max-segments, which deliberately stops short.
            record["translation"] = None
            record["expansion"] = None
        after.append(record)

    translated = [r for r in after if r.get("translation")]
    evidence = assert_mapping_is_sound(active, translated)
    evidence["segments_on_disk"] = len(after)
    evidence["segments_translated"] = len(translated)
    evidence["max_segments_cap"] = config.max_segments

    path.write_text(json.dumps(after, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = summarise_expansion(expansions)
    report = {
        "segments_path": str(path),
        "translate": {
            "model": config.translate_model,
            "mode": config.translate_mode,
            "source_language_code": config.source_lang,
            "target_language_code": config.target_lang,
            "numerals_format": "native",
            "max_input_chars": TRANSLATE_MODEL_MAX_CHARS[config.translate_model],
            "batching_enabled": not config.translate_no_batch,
            "max_batch_segments": config.translate_batch_segments,
            "context_segments_per_side": config.translate_context_segments,
        },
        "mapping_assertions": evidence,
        "batching": {**stats.to_dict(), "batches": [b.to_dict() for b in batches]},
        "expansion": {
            "per_segment": [
                {
                    "id": segment_id,
                    "start": next(s["start"] for s in before if s["id"] == segment_id),
                    "end": next(s["end"] for s in before if s["id"] == segment_id),
                    "source_text": next(s["text"] for s in before if s["id"] == segment_id),
                    "translation": translations[segment_id],
                    **expansions[segment_id].to_dict(),
                }
                for segment_id in sorted(expansions)
            ],
            "summary": summary,
        },
    }

    report_path = config.output_dir / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _log_expansion_table(report["expansion"]["per_segment"], summary)
    logger.info("wrote %s (%d translated) and %s", path, len(translated), report_path)

    metrics.segment_count = len(after)
    return report


def summarise_expansion(expansions: dict[int, Expansion]) -> dict[str, Any]:
    """Aggregate per-segment expansion into the numbers M4 needs before it is built."""
    from ..metrics import NOT_MEASURED, percentile

    if not expansions:
        return {"segments": 0, "mean_char_ratio": NOT_MEASURED}

    char_ratios = [e.char_ratio for e in expansions.values()]
    syllable_ratios = [
        e.syllable_ratio_est for e in expansions.values() if e.syllable_ratio_est is not None
    ]

    summary: dict[str, Any] = {
        "segments": len(expansions),
        "total_source_chars": sum(e.source_chars for e in expansions.values()),
        "total_target_chars": sum(e.target_chars for e in expansions.values()),
        "corpus_char_ratio": round(
            sum(e.target_chars for e in expansions.values())
            / max(1, sum(e.source_chars for e in expansions.values())), 4,
        ),
        "mean_char_ratio": round(sum(char_ratios) / len(char_ratios), 4),
        "min_char_ratio": round(min(char_ratios), 4),
        "max_char_ratio": round(max(char_ratios), 4),
        "p95_char_ratio": round(percentile(char_ratios, 95), 4),
    }

    if syllable_ratios:
        summary.update({
            "mean_syllable_ratio_est": round(sum(syllable_ratios) / len(syllable_ratios), 4),
            "min_syllable_ratio_est": round(min(syllable_ratios), 4),
            "max_syllable_ratio_est": round(max(syllable_ratios), 4),
            "p95_syllable_ratio_est": round(percentile(syllable_ratios, 95), 4),
            "segments_over_1_25_speedup_est": sum(1 for r in syllable_ratios if r > 1.25),
            "syllable_estimator": (
                "rule-based: English vowel groups with silent-final-e correction; "
                "Hindi aksharas with word-final schwa deletion. An estimate, not a "
                "phonetic measurement."
            ),
        })
    else:
        summary["mean_syllable_ratio_est"] = NOT_MEASURED

    return summary


def _log_expansion_table(rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Print the per-segment expansion table that M4's difficulty is predicted from."""
    logger.info("expansion ratios (translated vs source):")
    logger.info("  %3s %8s %8s %7s %8s %8s %7s", "id", "src_ch", "tgt_ch", "ratio",
                "src_syl", "tgt_syl", "syl_r")
    for row in rows:
        logger.info(
            "  %3d %8d %8d %7.3f %8s %8s %7s",
            row["id"], row["source_chars"], row["target_chars"], row["char_ratio"],
            row["source_syllables_est"], row["target_syllables_est"],
            row["syllable_ratio_est"],
        )
    logger.info("  mean char ratio: %s | mean syllable ratio: %s",
                summary.get("mean_char_ratio"), summary.get("mean_syllable_ratio_est"))
