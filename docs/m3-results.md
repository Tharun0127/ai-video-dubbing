# M3 — Per-segment translation (en-IN → hi-IN)

Every number here comes from a real run of this code on `samples/test_clip.mp4`
(36.107 s, 4 segments). Raw observations: `output/m3_register_probe.json`,
`output/translate_report.json`, `output/segments.json`.

Reproduce with:

```bash
python -m scripts.m3_probe                                   # register + batching probe
python -m src.pipeline --input samples/test_clip.mp4 --stage translate
```

Both are fully cached, so a repeat run issues zero network calls and costs ₹0.

---

## 1. What was built

`src/stages/translate.py`, plus `SarvamClient.translate()` as the only path to
`POST /translate`.

- Segments are packed into as few requests as the model's character cap allows, using a
  numbered-line protocol (`[1] …`), and every reply is parsed **strictly**.
- Neighbouring segments can be sent as context-only lines whose translations are discarded
  (`--translate-context-segments`). `/translate` has no context parameter, so this is the
  only way to give the model discourse context — confirmed against the live schema.
- Any reply that does not carry markers `1..N` exactly once, in order, with non-empty text
  is **rejected**, and that batch is redone one segment per request. Never realigned by
  guesswork.
- Mapping assertions are fatal, not warnings (§5).

---

## 2. Register — the investigation

### 2.1 Side-by-side, same four real segments

One request per segment, so the comparison measures register alone and is not confounded
by batching. Full data in `output/m3_register_probe.json`.

**Segment 0 — "Okay, Jay, pick a card."**

| model / mode | output |
| --- | --- |
| `sarvam-translate:v1` / formal | ठीक है, जय, एक कार्ड **चुनो**। |
| `mayura:v1` / formal | ठीक है, जय, एक कार्ड **चुनें**। |
| `mayura:v1` / modern-colloquial | Okay Jay, एक card pick **करो**। |
| `mayura:v1` / classic-colloquial | ठीक है, Jay, एक card चुन **लीजिए**। |
| `mayura:v1` / code-mixed | Okay Jay, एक card pick **कीजिए**। |

**Segment 1 — "Remember it. Right, Jay, shuffle the cards."**

| model / mode | output |
| --- | --- |
| `sarvam-translate:v1` / formal | याद रखना। ठीक है, जय, ताश के पत्तों को मिला दो। |
| `mayura:v1` / formal | याद रखिएगा। ठीक है, जय, कार्डों को फेर दो। |
| `mayura:v1` / modern-colloquial | याद रखिए। **Right**, Jay, cards shuffle कर दीजिए। |
| `mayura:v1` / classic-colloquial | **इसे** याद रखिए। ठीक है, Jay, cards shuffle कीजिए। |
| `mayura:v1` / code-mixed | याद रखिए। **Right**, Jay, cards shuffle कर दीजिए। |

**Segment 2 — "Jay, your card is the top card. No way. That's amazing. Really? Yeah."**

| model / mode | output |
| --- | --- |
| `sarvam-translate:v1` / formal | जय, तुम्हारा कार्ड **सबसे अच्छा कार्ड** है। **कोई बात नहीं।** यह अद्भुत है। सच में? हाँ। |
| `mayura:v1` / formal | **अच्छा**, आपका कार्ड **शीर्ष** कार्ड है। नहीं, ऐसा नहीं हो सकता। यह बहुत अच्छा है। सच में? हाँ। |
| `mayura:v1` / modern-colloquial | Jay, आपका card top card है। अरे नहीं। वाह, **amazing** है। सच में? हाँ। |
| `mayura:v1` / classic-colloquial | Jay, आपका card **सबसे ऊपर वाला** card है। अरे नहीं। वाह, **ये तो कमाल है**। सच में? हाँ जी। |
| `mayura:v1` / code-mixed | Jay, आपका card top card है। अरे नहीं। वाह, **amazing** है। सच में? हाँ। |

**Segment 3 — "How did you do that? I'm amazed."**

| model / mode | output |
| --- | --- |
| `sarvam-translate:v1` / formal | तुमने यह कैसे किया? मैं हैरान हूँ। |
| `mayura:v1` / formal | यह कैसे कर लिया आपने? मैं **चकित** हूँ। |
| `mayura:v1` / modern-colloquial | आपने यह कैसे किया? मैं **amazed** हूँ। |
| `mayura:v1` / classic-colloquial | आपने ऐसा कैसे किया? मैं हैरान हूँ। |
| `mayura:v1` / code-mixed | आपने यह कैसे किया? मैं **amazed** हूँ। |

### 2.2 What the samples show

**`sarvam-translate:v1` has two outright semantic errors on segment 2.**
"top card" → **सबसे अच्छा कार्ड** is *best card*, and this is a card trick whose entire
punchline is that the card is on **top** — the dub would stop making sense at the reveal.
"No way" → **कोई बात नहीं** is *never mind* / *no problem*; the line is astonishment, not
reassurance. This is not a register complaint, it is a mistranslation, and it is the
strongest single result in the comparison. Its register is otherwise fine: it holds
तुम-level address consistently, which actually matches the peer-to-peer source better than
anything mayura produced.

**`mayura:v1` / formal mixes levels of address within one segment.**
Segment 1 is "याद रखिएगा" (आप, deferential) followed by "फेर दो" (तुम, familiar) in the
same breath. It also drops the name "Jay" from segment 2 entirely, and reaches for literary
vocabulary — **शीर्ष** (apex) for "top", **चकित** for "amazed" — that no one says out loud.

**`modern-colloquial` and `code-mixed` are near-identical here** (they differ only on
segment 0: करो vs कीजिए) and both leave the emotionally-loaded words in English:
"वाह, **amazing** है", "मैं **amazed** हूँ". They also leave the discourse marker "Right"
untranslated, which reads as an artefact rather than a choice. Heavy Hinglish is realistic
for urban speech, but a dub that outsources every emphatic word to English has stopped
translating the part the viewer actually reacts to.

**`classic-colloquial` is the only variant that gets segment 2 completely right** —
सबसे ऊपर वाला card (topmost, and colloquially phrased), अरे नहीं (correct astonishment),
वाह, ये तो कमाल है (natural Hindi for "That's amazing"), हाँ जी. It also resolves the
pronoun in segment 1 explicitly (**इसे** याद रखिए — "remember *it*"), and it is the only
mayura mode that holds **one consistent level of address across all four segments**.

### 2.3 Recommendation — `mayura:v1` / `classic-colloquial`

Set as the default in `src/config.py`, overriding SPEC.md's Sarvam-Translate.

Ranked reasons:

1. **Semantic accuracy first.** It is the only variant with no mistranslation on segment 2.
   A wrong register sounds off; a wrong noun breaks the video.
2. **Consistent address.** A character switching between तुम and आप inside four seconds
   sounds broken in a way viewers notice even if they cannot name it. Only
   `classic-colloquial` avoids it.
3. **It translates the emotional content** instead of leaving it in English.
4. **Moderate, domain-appropriate code-mixing.** "card"/"shuffle" in English is how the
   domain is actually spoken; "amazing"/"amazed" in English is not.

**Known weakness of this choice, stated plainly:** its level of address (आप — लीजिए,
कीजिए) is more deferential than the casual peer banter of the source, where
`sarvam-translate:v1`'s तुम forms are a better match. Given a variant that is polite but
correct and a variant that is correctly-pitched but mistranslates the punchline, the
correct one wins.

**And it is the most expensive choice for M4** — see §3. `--translate-model` /
`--translate-mode` make this a one-flag change if M4 shows the pace clamp saturating.

**Constraint accepted with it:** `mayura:v1` caps at 1000 input chars (vs 2000) and covers
11 languages (vs 23). Fine for en→hi; a run targeting e.g. `sat-IN` must use
`--translate-model sarvam-translate:v1`, and the config rejects the bad combination at
startup rather than mid-run.

---

## 3. Expansion ratio — what M4 is walking into

### 3.1 Per segment, chosen variant (`mayura:v1` / `classic-colloquial`)

Persisted per segment in `output/segments.json` under `expansion`, and in full in
`output/translate_report.json`.

| id | window (s) | src chars | tgt chars | **char ratio** | src syl | tgt syl | **syllable ratio** | src syl/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000–5.475 | 23 | 31 | 1.348 | 6 | 9 | **1.500** | 1.096 |
| 1 | 5.475–13.183 | 43 | 47 | 1.093 | 10 | 15 | **1.500** | 1.297 |
| 2 | 13.183–25.270 | 69 | 83 | 1.203 | 16 | 26 | **1.625** | 1.324 |
| 3 | 25.270–36.107 | 32 | 33 | 1.031 | 8 | 11 | **1.375** | 0.738 |
| **mean** | | | | **1.169** | | | **1.500** | |
| **corpus total** | | 167 | 194 | **1.162** | 40 | 61 | **1.525** | |

### 3.2 The headline finding: character ratio badly understates the problem

**Characters say Hindi is ~17% longer. Syllables say it is ~50% longer.**

Character count is not comparable across scripts. Devanagari packs a consonant cluster plus
its vowel into one or two code points where Latin needs four or five, and this variant also
keeps English loanwords in Latin script, which deflates the character count further without
making the speech any shorter. Duration follows syllables, not code points.

Planning M4 off the character ratio would have predicted a comfortable ~1.17× and produced
a nasty surprise at synthesis time. The syllable estimate is the number to plan from.

### 3.3 All five variants, ranked by expansion

| model / mode | mean char ratio | **mean syllable ratio** | max syllable ratio |
| --- | --- | --- | --- |
| `mayura:v1` / modern-colloquial | 1.063 | **1.299** | 1.375 |
| `mayura:v1` / code-mixed | 1.085 | **1.341** | 1.500 |
| `sarvam-translate:v1` / formal | 1.111 | **1.345** | 1.438 |
| `mayura:v1` / formal | 1.130 | **1.398** | 1.625 |
| **`mayura:v1` / classic-colloquial** (chosen) | 1.177 | **1.531** | 1.625 |

The chosen variant is the worst of the five for expansion. That is the deliberate trade in
§2.3, and the flags to reverse it exist. The code-mixed modes are cheapest precisely
because they leave English words alone — English carries fewer syllables per idea here.

### 3.4 What this means for the M4 duration-fit loop

If Bulbul speaks Hindi at the same syllable rate the source speaker used for English, a
segment carrying 1.5× the syllables needs to be spoken 1.5× faster to fit the same window.
**Against SPEC.md's 1.25 perceptual clamp, all four segments would clamp, and the loop
would run out of headroom on every one of them.**

**That worst case probably does not apply to this clip, and the reason matters.** The
measured source syllable rate is **0.74–1.32 syllables/second** (last column of §3.1),
against the 4–6 syl/s typical of conversational speech. M2's segments tile the entire
timeline including pauses, so each window is mostly silence — segment 3 is a 10.8 s window
carrying only 8 source syllables. The speed-up actually required is
`syllable_ratio × (speech seconds ÷ window seconds)`. Dividing the measured rate by a
nominal 4–6 syl/s puts that second factor somewhere near 0.2–0.3, which would leave M4
generous headroom — but that is an **inference from an assumed speaking rate, not a
measurement**, and it is exactly the quantity M4 must measure rather than inherit.

`max_speedup_needed_est` in `segments.json` records the **upper bound** (the
syllable ratio itself, i.e. assuming a wall-to-wall-speech segment). It is deliberately an
upper bound and labelled as one: **the speech-vs-pause split per segment is not yet
measured** and is M4/M5 work. Concretely, what M4 should expect:

- On dense, wall-to-wall-speech material, expect the clamp to bind on most segments and
  the residual-drift path to carry real traffic. Build that path properly.
- On this clip, expect the opposite. **Do not tune the loop on this clip alone** — its
  slack is unrepresentative and would hide the clamping behaviour entirely.
- The stretch goal in SPEC.md (re-translate more concisely when clamping fails) is aimed at
  a ~1.5× syllable ratio, which is what was measured. It is likely to be needed on denser
  source material, and `--translate-mode modern-colloquial` (1.30) is a cheaper first lever.

### 3.5 How syllables are counted, and how far to trust it

Rule-based, not phonetic — labelled `*_est` everywhere and in `metrics`:

- **English:** vowel groups, minus silent word-final `-e` ("care" → 1), minus
  non-syllabic past-tense `-ed` ("amazed" → 2, but "wanted" → 2).
- **Hindi:** aksharas, with **both** schwa-deletion rules. Word-final (राम = *raam*, 1, not
  *raa-ma*) and medial V-C-C-V (तुमने = *tum-ne*, 2, not *tu-ma-ne*; आपका = *aap-kaa*, 2).
  Skipping medial deletion inflated every count and pushed the measured ratio from 1.53 to
  1.59 before it was fixed.
- **Mixed script** is counted word by word, so the code-mixed modes are measurable rather
  than skipped.

Validated against **25 hand-romanised cases** in `tests/test_translate.py`, chosen to cover
conjuncts (कार्ड = 1), both deletion rules, and the case where medial deletion must *not*
fire (ऊपर = *uu-par*, 2). Text in a script with no estimator returns `None` and serialises
as `"not yet measured"` rather than a wrong number.

---

## 4. Batching and context — measured, including where it did not pay off

### 4.1 Three configurations, same four segments

| run | requests | context lines | chars billed | of which payload | overhead | **cost** |
| --- | --- | --- | --- | --- | --- | --- |
| **default** (batched, ≤12/req) | **1** | 0 | 186 | 167 | 19 | **₹0.3720** |
| `--translate-batch-segments 1` (±2 context) | 4 | 10 | 679 | 167 | 512 | ₹1.3580 |
| `--translate-no-batch` (no context) | 4 | 0 | 167 | 167 | 0 | ₹0.3340 |

### 4.2 Batching saves wall-clock, not credits — correcting the brief

The brief asked to batch "to save both credits and wall-clock time". **It does not save
credits.** `/translate` is billed per input character, not per request, so batching four
segments adds the 19 characters of `[1] `-style markers and costs **11% more** than four
separate calls (₹0.3720 vs ₹0.3340). What it buys is round trips: **1 request instead of 4**.
Measured mean network latency per `/translate` call across the probe was **0.435 s** over 12
calls, so the saving is roughly 1.3 s on this clip and scales linearly with segment count.
(That is arithmetic over a measured per-call latency; a clean uncached end-to-end
wall-clock A/B was not run separately, to avoid re-spending credits.)

For a 3-minute video the ratio is what matters: batching turns ~40 serial round trips into
~4, which is the difference between a stage that takes 20 s and one that takes 2 s, for a
few paise.

### 4.3 Context is free when it is implicit, and expensive when it is not

The default run sent **zero explicit context lines** — all four segments fit in one
1000-char request, so every segment already had its neighbours in the same request. Context
only has to be paid for at a batch *seam*.

Forcing one payload segment per request (`--translate-batch-segments 1`) makes that cost
visible: 10 context lines, 512 characters of overhead, **4× the cost** (₹1.358 vs ₹0.334).

### 4.4 Honest result: on this clip, context changed almost nothing

Comparing all three runs segment by segment, **only segment 3 differed**, and only lexically:

- batched (context implicit): आपने **ये** कैसे किया? मैं हैरान हूँ।
- isolated (no context at all): आपने **ऐसा** कैसे किया? मैं हैरान हूँ।

Segments 0, 1 and 2 were **byte-identical** across all three configurations. In particular
"Remember it." became "**इसे** याद रखिए" — with the pronoun explicit — even with no context
whatsoever, so `mayura:v1` resolved it from the sentence alone.

The mechanism is built, tested and instrumented, but **this clip does not demonstrate that
it helps.** Four segments of simple dialogue is too little evidence either way; a longer
clip with cross-segment anaphora is what would settle it. It is not claimed as a win.

### 4.5 The batch protocol needed a real fix, not a hopeful one

The first implementation parsed markers anchored to the start of a line. Against the live
API that failed on 4 of 5 variants, for two distinct reasons (both now in
`docs/api-notes.md` and both covered by tests):

1. **`mayura:v1` returns every segment joined onto one line.** Only
   `sarvam-translate:v1` preserved one line per segment.
2. **`numerals_format: native` rewrites the markers themselves** — `[1]` came back as
   `[१]`. The model applies native numerals to the scaffolding, not just the content.

And one failure that cannot be parsed around: **`mayura:v1` / formal dropped marker `[१]`
entirely.** There is no safe realignment of a reply with a missing marker, so the stage
rejects it and re-issues that batch one segment per request. In the default configuration
this fallback did not fire (`batches_fell_back_to_per_segment: 0`), but it is the reason
batching is safe to run at all.

---

## 5. Mapping assertions — segment count before/after

All fatal (`MappingError`), never warnings. From `output/translate_report.json`:

```json
"mapping_assertions": {
  "segments_in": 4, "segments_out": 4,
  "counts_match": true, "ids_unchanged": true, "timestamps_unchanged": true,
  "source_text_unchanged": true, "all_translations_non_empty": true,
  "ids": [0, 1, 2, 3]
}
```

**Segment count: 4 in → 4 out.** Also asserted: ids unchanged and still in timeline order,
no duplicate ids, `start`/`end` bit-identical to the M2 values, source text unchanged (M6
needs it as the WER reference), and every translation non-empty after stripping.

Each is covered by a test that damages exactly one invariant and asserts the stage refuses:
dropped segment, extra segment, renumbered id, moved timestamp, rewritten source text,
empty/whitespace/null translation, duplicate ids.

Why fatal: an off-by-one here shifts every subsequent segment's audio onto the wrong
timestamp. It produces a file that plays, so nothing downstream fails — it surfaces as
unexplained loss of lip sync in the finished video, several stages and a lot of debugging
away from the cause.

---

## 6. Cost and cache

| | requests | network calls | cost |
| --- | --- | --- | --- |
| Register + batching probe (two runs) | 26 | 26 | ₹3.576 |
| Translate stage, first run | 1 | 1 | ₹0.372 |
| Translate stage, batching A/B runs | 8 | 6 | ₹0.948 |
| **Total M3 spend** | | | **₹4.896** |
| Translate stage, repeat run | 1 | **0** | **₹0.0000** |

The repeat run was executed under `--dry-run`, which raises `CacheMissError` on any miss —
so "0 network calls" is enforced, not merely observed:

```
translate en-IN->hi-IN mode=classic-colloquial (CACHE HIT, 186 chars in -> 213 chars out)
mapping assertions passed: 4 in == 4 out, ids and timestamps unchanged, 4 non-empty translation(s)
  api results     : 1 (1 cache hits, 0 network calls, hit rate 100%)
  cost this run   : Rs 0.0000
  cost avoided    : Rs 0.3720 (served from cache)
```

Cache key includes model and mode, so switching register is correctly a new key and never
serves another mode's result (tested).

---

## 7. Tests

`pytest`: **213 passed**, 72 of them new in `tests/test_translate.py`. No test opens a
socket; batch behaviour is exercised against the verbatim replies both models really
returned, including the malformed `mayura:v1` / formal one.

---

## 8. Limitations

- **The context window is unproven.** Built, tested, instrumented — but on this clip it
  changed one word in one segment (§4.4). Not claimed as a quality win.
- **Syllable counts are rule-based estimates**, not phonetic measurements (§3.5).
- **`max_speedup_needed_est` is an upper bound**, because the speech-vs-pause split within
  each segment is not yet measured (§3.4).
- **The register recommendation rests on four segments** of one speaker in one domain.
  It is the right call on this evidence; it is not a general claim about the models.
- **`mayura:v1` restricts the pipeline to 11 languages and 1000 chars/request.** Enforced
  at startup, and `--translate-model sarvam-translate:v1` lifts both at the cost of register
  control.
- **No end-to-end wall-clock A/B for batching** on uncached calls (§4.2); the saving is
  arithmetic over a measured per-call latency.
