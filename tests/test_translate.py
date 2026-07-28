"""Tests for the translate stage: mapping, batching, the wire protocol, and expansion.

The API is never called. Batch behaviour is exercised against a stub client scripted
with the exact reply shapes both models were *observed* to produce on 2026-07-29 --
sarvam-translate:v1 returning one marker per line, mayura:v1 collapsing every line into
one and rendering the markers in Devanagari digits. Those observations are recorded in
docs/api-notes.md and are what the parser has to survive.

The syllable cases below are hand-romanised so a reviewer can check them without running
anything; they are the evidence that the expansion ratio M4 is planned from is measured
on a counter that actually knows Hindi schwa deletion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.config import Config
from src.metrics import MetricsCollector
from src.sarvam_client import SarvamAPIError, SarvamClient
from src.stages.translate import (
    Batch,
    MappingError,
    TranslateError,
    assert_mapping_is_sound,
    count_syllables,
    measure_expansion,
    parse_batch,
    plan_batches,
    render_batch,
    run_translate,
    summarise_expansion,
    translate_segments,
)

from tests.test_sarvam_client import FakeResponse, build_client

# The four real segments this milestone was built and measured on.
SEGMENTS: list[dict[str, Any]] = [
    {"id": 0, "start": 0.0, "end": 5.475, "text": "Okay, Jay, pick a card.", "speaker": None},
    {"id": 1, "start": 5.475, "end": 13.183,
     "text": "Remember it. Right, Jay, shuffle the cards.", "speaker": None},
    {"id": 2, "start": 13.183, "end": 25.27,
     "text": "Jay, your card is the top card. No way. That's amazing. Really? Yeah.",
     "speaker": None},
    {"id": 3, "start": 25.27, "end": 36.107, "text": "How did you do that? I'm amazed.",
     "speaker": None},
]

#: The real mayura:v1 classic-colloquial reply to the four segments, verbatim.
REAL_MAYURA_REPLY = (
    "[१] ठीक है, Jay, एक card चुन लीजिए। [२] इसे याद रखिए। ठीक है, Jay, cards shuffle कीजिए। "
    "[३] Jay, आपका card सबसे ऊपर वाला card है। अरे नहीं। वाह, ये तो कमाल है। सच में? हाँ जी। "
    "[४] आपने ऐसा कैसे किया? मैं हैरान हूँ।"
)

#: The real sarvam-translate:v1 formal reply, which preserves one marker per line.
REAL_SARVAM_REPLY = (
    "[१] ठीक है, जय, एक कार्ड चुनो।\n"
    "[२] याद रखना। ठीक है, जय, ताश के पत्तों को मिला दो।\n"
    "[३] जय, तुम्हारा कार्ड सबसे अच्छा कार्ड है। कोई बात नहीं। यह अद्भुत है। सच में? हाँ।\n"
    "[४] तुमने यह कैसे किया? मैं हैरान हूँ।"
)


class StubClient:
    """A SarvamClient stand-in that replays scripted translated_text values."""

    def __init__(self, replies: list[str]) -> None:
        """Queue the replies this stub returns, in order."""
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def translate(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """Record the request and return the next scripted reply."""
        self.calls.append({"input": text, **kwargs})
        if not self._replies:
            raise AssertionError(f"unexpected extra translate call: {text!r}")
        return {"request_id": f"stub-{len(self.calls)}",
                "translated_text": self._replies.pop(0),
                "source_language_code": kwargs.get("source_language_code")}


# --- syllable estimation ----------------------------------------------------------------

@pytest.mark.parametrize("text,expected,romanisation", [
    ("card", 1, "card"),
    ("Okay, Jay, pick a card.", 6, "o-kay jay pick a card"),
    ("That's amazing.", 4, "thats a-ma-zing"),
    ("I'm amazed.", 3, "im a-mazed -- silent -ed"),
    ("wanted", 2, "want-ed -- -ed IS syllabic after t"),
    ("How did you do that? I'm amazed.", 8, ""),
    ("Remember it. Right, Jay, shuffle the cards.", 10, ""),
])
def test_english_syllable_estimates(text: str, expected: int, romanisation: str) -> None:
    """The English estimator handles silent final -e and non-syllabic past-tense -ed."""
    assert count_syllables(text, "en-IN") == expected, romanisation


@pytest.mark.parametrize("text,expected,romanisation", [
    # word-final schwa deletion
    ("राम", 1, "raam, not raa-ma"),
    ("जय", 1, "jay"),
    ("ठीक", 1, "Thiik"),
    ("एक", 1, "ek"),
    ("कमल", 2, "ka-mal -- the final schwa goes, the medial one survives"),
    # conjuncts collapse into one akshara
    ("कार्ड", 1, "kaarD"),
    ("पत्तों", 2, "pat-toN"),
    ("तुम्हारा", 3, "tum-haa-raa"),
    ("शीर्ष", 1, "shiirsh"),
    ("अद्भुत", 2, "ad-bhut"),
    # medial schwa deletion, the V-C-C-V rule
    ("तुमने", 2, "tum-ne, not tu-ma-ne"),
    ("आपका", 2, "aap-kaa, not aa-pa-kaa"),
    ("सबसे", 2, "sab-se"),
    ("रखना", 2, "rakh-naa"),
    ("ऊपर", 2, "uu-par -- no deletion: the next akshara has no vowel of its own"),
    ("रखिए", 3, "ra-khi-e"),
    # phrases
    ("ठीक है, जय, एक कार्ड चुनो।", 7, "Thiik hai jay ek kaarD chu-no"),
    ("तुमने यह कैसे किया? मैं हैरान हूँ।", 11,
     "tum-ne yah kai-se ki-yaa maiN hai-raan huuN"),
])
def test_hindi_syllable_estimates(text: str, expected: int, romanisation: str) -> None:
    """The Hindi estimator counts aksharas and applies both schwa-deletion rules."""
    assert count_syllables(text, "hi-IN") == expected, romanisation


def test_mixed_script_is_counted_word_by_word() -> None:
    """Code-mixed output must still be measurable, or the code-mixed modes cannot be compared."""
    # Jay(1) aap-kaa(2) card(1) top(1) card(1) hai(1)
    assert count_syllables("Jay, आपका card top card है", "hi-IN") == 7


def test_unsupported_script_reports_nothing_rather_than_a_wrong_number() -> None:
    """A script with no estimator returns None so metrics can flag it, not silently undercount."""
    assert count_syllables("これは日本語です", "ja-JP") is None


# --- expansion measurement --------------------------------------------------------------

def test_expansion_measures_both_ratios() -> None:
    """Character and syllable ratios are computed from the real segment 0 pair."""
    expansion = measure_expansion(
        "Okay, Jay, pick a card.", "ठीक है, Jay, एक card चुन लीजिए।",
        source_lang="en-IN", target_lang="hi-IN", duration_s=5.475,
    )
    assert expansion.source_chars == 23
    assert expansion.target_chars == 31
    assert expansion.char_ratio == pytest.approx(31 / 23)
    assert expansion.source_syllables_est == 6
    assert expansion.target_syllables_est == 9
    assert expansion.syllable_ratio_est == pytest.approx(1.5)


def test_character_ratio_understates_expansion_versus_syllables() -> None:
    """The measured finding M4 depends on: chars say ~1.2x, syllables say ~1.5x."""
    expansion = measure_expansion(
        "Okay, Jay, pick a card.", "ठीक है, Jay, एक card चुन लीजिए।",
        source_lang="en-IN", target_lang="hi-IN", duration_s=5.475,
    )
    assert expansion.syllable_ratio_est is not None
    assert expansion.syllable_ratio_est > expansion.char_ratio


def test_summary_flags_segments_over_the_perceptual_clamp() -> None:
    """Segments whose estimated speed-up exceeds 1.25 are counted for M4."""
    expansions = {
        0: measure_expansion("a b", "क कि की", source_lang="en-IN", target_lang="hi-IN",
                             duration_s=1.0),
    }
    summary = summarise_expansion(expansions)
    assert summary["segments"] == 1
    assert "segments_over_1_25_speedup_est" in summary


# --- batch planning ---------------------------------------------------------------------

def test_all_segments_fit_one_batch_when_the_text_is_short() -> None:
    """The measured case: 167 chars of source needs exactly one mayura:v1 request."""
    batches = plan_batches(SEGMENTS, max_chars=1000, max_segments=12, context_segments=2)
    assert len(batches) == 1
    assert batches[0].payload_ids == [0, 1, 2, 3]
    # Everything is already in the request, so there is nothing left to add as context.
    assert batches[0].context_ids == []


def test_character_cap_forces_more_batches() -> None:
    """A tight cap splits the work rather than sending an over-length request."""
    batches = plan_batches(SEGMENTS, max_chars=100, max_segments=12, context_segments=0)
    assert len(batches) > 1
    assert [i for b in batches for i in b.payload_ids] == [0, 1, 2, 3]


def test_max_segments_caps_batch_size() -> None:
    """The segment ceiling binds even when the character budget would allow more."""
    batches = plan_batches(SEGMENTS, max_chars=1000, max_segments=2, context_segments=0)
    assert [b.payload_ids for b in batches] == [[0, 1], [2, 3]]


def test_context_window_pulls_in_neighbours_at_a_batch_seam() -> None:
    """One segment per batch means every batch must carry its neighbours explicitly."""
    batches = plan_batches(SEGMENTS, max_chars=1000, max_segments=1, context_segments=1)
    assert [b.payload_ids for b in batches] == [[0], [1], [2], [3]]
    assert batches[0].context_ids == [1]
    assert batches[1].context_ids == [0, 2]
    assert batches[3].context_ids == [2]


def test_context_never_appears_in_two_roles_in_one_batch() -> None:
    """A segment that is already payload is not also billed as context."""
    for batch in plan_batches(SEGMENTS, max_chars=1000, max_segments=2, context_segments=2):
        assert not set(batch.payload_ids) & set(batch.context_ids)


def test_segment_longer_than_the_model_cap_fails_loudly() -> None:
    """M2 should have split it; translating it would return a silent 422."""
    with pytest.raises(TranslateError, match="over the"):
        plan_batches([{"id": 0, "text": "x" * 1200}], max_chars=1000, max_segments=12,
                     context_segments=0)


def test_no_batch_is_never_empty() -> None:
    """Every planned batch answers for at least one segment, even at a very tight cap."""
    # 90 sits just above the longest segment (69 chars), so the budget forces one segment
    # per request and the packing loop is pushed to its edge case.
    batches = plan_batches(SEGMENTS, max_chars=90, max_segments=12, context_segments=1)
    assert len(batches) == 4
    for batch in batches:
        assert batch.payload_ids


def test_context_is_dropped_rather_than_overflowing_the_budget() -> None:
    """Context is a nice-to-have; the character cap is not negotiable."""
    batches = plan_batches(SEGMENTS, max_chars=90, max_segments=12, context_segments=2)
    # Segment 2 alone is 69 chars and already fills the 72-char budget, so nothing else fits.
    crowded = next(b for b in batches if b.payload_ids == [2])
    assert crowded.context_ids == []


# --- wire protocol ------------------------------------------------------------------------

def test_single_segment_is_sent_bare() -> None:
    """One line needs no marker, so it costs nothing extra and cannot mis-parse."""
    assert render_batch(["Okay, Jay, pick a card."]) == "Okay, Jay, pick a card."


def test_multi_segment_request_is_numbered() -> None:
    """Markers are 1-based and one per line."""
    assert render_batch(["a", "b"]) == "[1] a\n[2] b"


def test_parses_the_real_sarvam_translate_reply() -> None:
    """sarvam-translate:v1 keeps one marker per line; all four lines must come back."""
    parsed = parse_batch(REAL_SARVAM_REPLY, 4)
    assert parsed is not None
    assert len(parsed) == 4
    assert parsed[0] == "ठीक है, जय, एक कार्ड चुनो।"
    assert parsed[3] == "तुमने यह कैसे किया? मैं हैरान हूँ।"


def test_parses_the_real_mayura_reply_despite_collapsed_lines() -> None:
    """mayura:v1 returns every segment on one line -- scanning for markers still recovers them."""
    assert REAL_MAYURA_REPLY.count("\n") == 0
    parsed = parse_batch(REAL_MAYURA_REPLY, 4)
    assert parsed is not None
    assert len(parsed) == 4
    assert parsed[0] == "ठीक है, Jay, एक card चुन लीजिए।"
    assert parsed[3] == "आपने ऐसा कैसे किया? मैं हैरान हूँ।"


def test_devanagari_digits_in_markers_are_understood() -> None:
    """numerals_format=native rewrites [1] as [१]; the parser must not care."""
    assert parse_batch("[१] एक\n[२] दो", 2) == ["एक", "दो"]


def test_a_dropped_marker_is_rejected_rather_than_guessed() -> None:
    """Observed for mayura:v1 in formal mode: marker 1 was missing from the reply."""
    assert parse_batch("पहला। [२] दूसरा [३] तीसरा [४] चौथा", 4) is None


def test_out_of_order_or_duplicated_markers_are_rejected() -> None:
    """Anything but the exact sequence 1..N is a mapping risk and must not be accepted."""
    assert parse_batch("[2] b [1] a", 2) is None
    assert parse_batch("[1] a [1] b", 2) is None


def test_an_empty_line_is_rejected() -> None:
    """A blank translation would pass the count check but produce a silent gap in the dub."""
    assert parse_batch("[1] a [2]   [3] c", 3) is None


def test_wrong_line_count_is_rejected() -> None:
    """Merging two segments into one line is exactly the off-by-one that breaks sync."""
    assert parse_batch("[1] a [2] b", 3) is None


# --- translation orchestration -------------------------------------------------------------

def test_batched_translation_maps_every_segment(config: Config) -> None:
    """The happy path: one request, four segments back, mapped by marker order."""
    client = StubClient([REAL_MAYURA_REPLY])
    translations, batches, stats = translate_segments(SEGMENTS, config, client)  # type: ignore[arg-type]

    assert len(client.calls) == 1
    assert sorted(translations) == [0, 1, 2, 3]
    assert translations[2].startswith("Jay, आपका card सबसे ऊपर वाला card है")
    assert stats.batches_planned == 1
    assert stats.batches_parsed == 1
    assert stats.batches_fell_back == 0
    assert len(batches) == 1


def test_a_batch_that_does_not_round_trip_falls_back_per_segment(config: Config) -> None:
    """A mangled reply must never be aligned by guesswork; it is redone one at a time."""
    mangled = "सब कुछ एक ही लाइन में, कोई marker नहीं।"
    client = StubClient([mangled, "एक", "दो", "तीन", "चार"])
    translations, _batches, stats = translate_segments(SEGMENTS, config, client)  # type: ignore[arg-type]

    assert stats.batches_fell_back == 1
    assert stats.batches_parsed == 0
    # One failed batch call plus one call per segment.
    assert len(client.calls) == 5
    assert translations == {0: "एक", 1: "दो", 2: "तीन", 3: "चार"}
    assert stats.fallback_reasons and "did not round-trip" in stats.fallback_reasons[0]


def test_no_batch_sends_one_request_per_segment(config: Config) -> None:
    """--translate-no-batch is the A/B control for the batching measurement."""
    no_batch = Config(**{**config.__dict__, "translate_no_batch": True})
    client = StubClient(["एक", "दो", "तीन", "चार"])
    translations, batches, stats = translate_segments(SEGMENTS, no_batch, client)  # type: ignore[arg-type]

    assert len(client.calls) == 4
    assert stats.single_line_requests == 4
    assert all(call["input"] in {s["text"] for s in SEGMENTS} for call in client.calls)
    assert len(batches) == 4
    assert len(translations) == 4


def test_context_segments_are_sent_but_their_translations_discarded(config: Config) -> None:
    """Context is paid for to improve the payload lines, not to produce output of its own."""
    windowed = Config(**{**config.__dict__, "translate_batch_segments": 1,
                         "translate_context_segments": 1})
    # Batch 0 sends segments 0 and 1; only 0 is payload.
    client = StubClient([
        "[1] शून्य [2] एक",
        "[1] शून्य [2] एक [3] दो",
        "[1] एक [2] दो [3] तीन",
        "[1] दो [2] तीन",
    ])
    translations, batches, stats = translate_segments(SEGMENTS, windowed, client)  # type: ignore[arg-type]

    assert batches[0].payload_ids == [0] and batches[0].context_ids == [1]
    assert translations[0] == "शून्य"
    assert stats.context_lines_sent == 6
    # Two segments went over the wire in the first request, but only one was kept.
    assert client.calls[0]["input"] == render_batch(
        [SEGMENTS[0]["text"], SEGMENTS[1]["text"]]
    )


# --- mapping assertions ---------------------------------------------------------------------

def _translated(**overrides: Any) -> list[dict[str, Any]]:
    """Build a well-formed translated segment list, with optional damage applied."""
    rows = [{**s, "translation": f"अनुवाद {s['id']}"} for s in SEGMENTS]
    for key, value in overrides.items():
        index, field = key.split("_", 1)
        rows[int(index)][field] = value
    return rows


def test_sound_mapping_passes_and_returns_evidence() -> None:
    """The good case records the evidence that goes into translate_report.json."""
    evidence = assert_mapping_is_sound(SEGMENTS, _translated())
    assert evidence["segments_in"] == evidence["segments_out"] == 4
    assert evidence["counts_match"] and evidence["all_translations_non_empty"]


def test_a_dropped_segment_is_fatal() -> None:
    """Count mismatch is the off-by-one that silently destroys sync downstream."""
    with pytest.raises(MappingError, match="segment count changed"):
        assert_mapping_is_sound(SEGMENTS, _translated()[:3])


def test_an_extra_segment_is_fatal() -> None:
    """More out than in is equally fatal, and equally invisible later."""
    rows = _translated()
    with pytest.raises(MappingError, match="segment count changed"):
        assert_mapping_is_sound(SEGMENTS, rows + [rows[-1]])


def test_a_changed_id_is_fatal() -> None:
    """Ids are the mapping; a renumber means the audio lands on the wrong timestamps."""
    with pytest.raises(MappingError, match="changed id"):
        assert_mapping_is_sound(SEGMENTS, _translated(**{"1_id": 99}))


def test_a_moved_timestamp_is_fatal() -> None:
    """Translation must never move a boundary -- M2 owns the timeline."""
    with pytest.raises(MappingError, match="start changed"):
        assert_mapping_is_sound(SEGMENTS, _translated(**{"2_start": 13.9}))


def test_a_rewritten_source_text_is_fatal() -> None:
    """The source text is the QC reference in M6; it must survive this stage untouched."""
    with pytest.raises(MappingError, match="source text changed"):
        assert_mapping_is_sound(SEGMENTS, _translated(**{"0_text": "something else"}))


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_an_empty_translation_is_fatal(empty: Any) -> None:
    """A blank translation would synthesise silence and desync everything after it."""
    with pytest.raises(MappingError, match="no translation"):
        assert_mapping_is_sound(SEGMENTS, _translated(**{"3_translation": empty}))


def test_duplicate_ids_are_fatal() -> None:
    """Two segments claiming the same id means one of them will be overwritten."""
    rows = _translated()
    rows[2]["id"] = 1
    rows[1]["id"] = 1
    with pytest.raises(MappingError, match="duplicate segment ids"):
        assert_mapping_is_sound([{**s, "id": 1 if s["id"] in (1, 2) else s["id"]}
                                 for s in SEGMENTS], rows)


# --- client-level validation ----------------------------------------------------------------

@pytest.fixture
def translate_response() -> dict[str, Any]:
    """The real mayura:v1 response captured on 2026-07-29."""
    path = Path(__file__).parent / "fixtures" / "translate_mayura_v1_classic_colloquial.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_translate_call_sends_the_documented_json_body(
    config: Config, translate_response: dict[str, Any]
) -> None:
    """The request body carries exactly the fields the confirmed schema defines."""
    client, session, _ = build_client(config, [FakeResponse(200, translate_response)])
    client.translate("Okay, Jay, pick a card.", source_language_code="en-IN",
                     target_language_code="hi-IN", model="mayura:v1",
                     mode="classic-colloquial", numerals_format="native")

    call = session.calls[0]
    assert call["url"] == "https://api.sarvam.ai/translate"
    assert call["json"] == {
        "model": "mayura:v1",
        "source_language_code": "en-IN",
        "target_language_code": "hi-IN",
        "mode": "classic-colloquial",
        "numerals_format": "native",
        "input": "Okay, Jay, pick a card.",
    }


def test_translation_is_billed_per_input_character(
    config: Config, translate_response: dict[str, Any]
) -> None:
    """Rs 20 per 10k input characters, from the published price list."""
    metrics = MetricsCollector()
    client, _, _ = build_client(config, [FakeResponse(200, translate_response)], metrics)
    client.translate("x" * 500, source_language_code="en-IN", target_language_code="hi-IN")

    call = metrics.api_calls[0]
    assert call.billable_units == 500
    assert call.billable_unit_name == "input_characters"
    assert call.cost_inr == pytest.approx(500 * 20.0 / 10_000)


def test_identical_translation_is_served_from_cache(
    config: Config, translate_response: dict[str, Any]
) -> None:
    """The M1 guarantee holds for /translate too: second call is free and offline."""
    from src.cache import DiskCache

    metrics = MetricsCollector()
    cache = DiskCache(config.cache_dir)
    first, session, _ = build_client(config, [FakeResponse(200, translate_response)],
                                     metrics, cache)
    first.translate("hello", source_language_code="en-IN", target_language_code="hi-IN")

    second, session2, _ = build_client(config, [], metrics, DiskCache(config.cache_dir))
    second.translate("hello", source_language_code="en-IN", target_language_code="hi-IN")

    assert len(session.calls) == 1 and len(session2.calls) == 0
    assert metrics.cache_hits == 1
    assert metrics.api_calls[1].cost_inr == 0.0


def test_a_different_mode_is_a_different_cache_key(
    config: Config, translate_response: dict[str, Any]
) -> None:
    """Register is part of the identity of the result; it must not reuse another mode's entry."""
    from src.cache import DiskCache

    cache = DiskCache(config.cache_dir)
    client, session, _ = build_client(
        config, [FakeResponse(200, translate_response)] * 2, cache=cache
    )
    for mode in ("formal", "classic-colloquial"):
        client.translate("hello", source_language_code="en-IN", target_language_code="hi-IN",
                         model="mayura:v1", mode=mode)
    assert len(session.calls) == 2


def test_over_length_input_is_rejected_before_the_network(config: Config) -> None:
    """mayura:v1 caps at 1000 chars; catch it locally rather than paying for a 422."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="1000"):
        client.translate("x" * 1001, source_language_code="en-IN",
                         target_language_code="hi-IN", model="mayura:v1")
    assert session.calls == []


def test_sarvam_translate_allows_the_larger_input(
    config: Config, translate_response: dict[str, Any]
) -> None:
    """The cap is per model: sarvam-translate:v1 accepts 2000 chars."""
    client, session, _ = build_client(config, [FakeResponse(200, translate_response)])
    client.translate("x" * 1500, source_language_code="en-IN", target_language_code="hi-IN",
                     model="sarvam-translate:v1", mode="formal")
    assert len(session.calls) == 1


def test_a_style_mode_the_model_ignores_is_rejected(config: Config) -> None:
    """sarvam-translate:v1 silently returns formal output for any mode, so guard it locally."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="not supported"):
        client.translate("hello", source_language_code="en-IN", target_language_code="hi-IN",
                         model="sarvam-translate:v1", mode="classic-colloquial")
    assert session.calls == []


def test_a_language_mayura_does_not_cover_is_rejected(config: Config) -> None:
    """mayura:v1 covers 11 languages; the other 12 must route to sarvam-translate:v1."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="not supported"):
        client.translate("hello", source_language_code="en-IN", target_language_code="sat-IN",
                         model="mayura:v1")
    assert session.calls == []


def test_empty_input_is_refused(config: Config) -> None:
    """Billing for whitespace is pure waste and yields nothing usable."""
    client, session, _ = build_client(config, [])
    with pytest.raises(SarvamAPIError, match="empty"):
        client.translate("   ", source_language_code="en-IN", target_language_code="hi-IN")
    assert session.calls == []


# --- end-to-end stage ------------------------------------------------------------------------

def test_run_translate_rewrites_segments_json_and_writes_the_report(
    config: Config, tmp_path: Path, translate_response: dict[str, Any]
) -> None:
    """The stage contract: segments.json gains translation + expansion, ids and times intact."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_path = output_dir / "segments.json"
    segments_path.write_text(json.dumps(SEGMENTS, ensure_ascii=False), encoding="utf-8")

    cfg = Config(**{**config.__dict__, "output_dir": output_dir})
    client, _session, _ = build_client(
        cfg, [FakeResponse(200, {**translate_response, "translated_text": REAL_MAYURA_REPLY})]
    )
    report = run_translate(cfg, client, MetricsCollector())

    written = json.loads(segments_path.read_text(encoding="utf-8"))
    assert len(written) == len(SEGMENTS)
    for source, result in zip(SEGMENTS, written):
        assert result["id"] == source["id"]
        assert result["start"] == source["start"]
        assert result["end"] == source["end"]
        assert result["text"] == source["text"]
        assert result["translation"].strip()
        assert result["expansion"]["char_ratio"] > 0

    assert report["mapping_assertions"]["counts_match"] is True
    assert (output_dir / "translate_report.json").exists()
    assert report["expansion"]["summary"]["segments"] == 4


def test_run_translate_fails_loudly_when_asr_output_is_missing(
    config: Config, tmp_path: Path
) -> None:
    """A missing segments.json names the stage to run, rather than producing an empty dub."""
    cfg = Config(**{**config.__dict__, "output_dir": tmp_path / "empty"})
    client, _, _ = build_client(cfg, [])
    with pytest.raises(TranslateError, match="--stage asr"):
        run_translate(cfg, client, MetricsCollector())
