# M6 — QC harness

Every number here comes from a real run on `samples/test_clip.mp4` (36.107 s, 4 segments).
Raw data: `output/qc_report.md`, `output/qc_report.json`, `output/metrics.json`,
`output/tts_variance_probe.json`.

```bash
python -m src.pipeline --input samples/test_clip.mp4 --stage qc
python -m scripts.tts_variance_probe --repeats 3
```

The QC stage made 4 real Saaras calls over 36.107 s of dubbed audio and cost **₹0.3167**.
Re-running it is a cache hit and costs ₹0.

---

## 1. Headline

| Metric | Measured |
| --- | --- |
| Corpus WER | **0.2381** (10 substitutions, 0 deletions, 0 insertions, 42 reference words) |
| Corpus CER | **0.2135** |
| Segments scored | 4 of 4 |
| Mean absolute timing drift | 60.81% |
| p95 absolute timing drift | 78.52% (n=4 — not a robust p95, reported because the brief asks) |
| Segments over the 5% drift threshold | 4 of 4 |
| Coverage | PASS at all 5 hops |
| Segments flagged | 4 |

---

## 2. What is scored, and against what

The dubbed track is sliced at each segment's window and sent back through Saaras **in the
target language** (`hi-IN`), then compared with the stage-3 translation that was supposed to
be spoken there. Reference = what we asked Bulbul to say. Hypothesis = what Saaras hears in
the finished dub.

**The dub is re-transcribed after assembly, not clip by clip.** That is deliberate: it means
an assembly bug — a clip at the wrong offset, a clip mixed into its neighbour, a clip
dropped — shows up in the WER instead of passing unnoticed. Scoring the individual clips
would only re-measure M4 and would call a broken timeline perfect.

Both sides are NFC-normalised, lowercased, and stripped of punctuation (including the
Devanagari danda) before scoring, because punctuation is not spoken.

---

## 3. Per segment — measured

| Seg | Time | WER | CER | S/D/I | Ref words |
| --- | --- | --- | --- | --- | --- |
| 0 | `00:00.000` | 0.4286 | 0.3214 | 3/0/0 | 7 |
| 1 | `00:05.475` | 0.3333 | 0.3488 | 3/0/0 | 9 |
| 2 | `00:13.183` | 0.2105 | 0.1842 | 4/0/0 | 19 |
| 3 | `00:25.270` | **0.0000** | **0.0000** | 0/0/0 | 7 |

Segment 3 came back through the full pipeline — translate, synthesise, fit, place, remux,
re-transcribe — **word for word identical**. That is the strongest single piece of evidence
that the timeline arithmetic is right.

---

## 4. What the 10 substitutions actually are

Zero deletions and zero insertions across the whole clip, which already rules out a dropped
or duplicated segment. Every error is a substitution, and all 10 were inspected by hand:

| Kind | Count | Example |
| --- | --- | --- |
| Latin-script loanword transcribed in Devanagari | **8** | `Jay` → `जय`, `card` → `कार्ड`, `shuffle` → `शफल` |
| Orthographic variant of the same sound | **1** | `हाँ` → `हां` (chandrabindu vs anusvara) |
| **Genuine content error** | **1** | `चुन लीजिए` (pick) → `सुन लीजिए` (listen) |

The 8 script mismatches are structural, not defects: `mayura:v1` in classic-colloquial keeps
English loanwords in Latin script (that was the M3 decision, and it is the right one for a
code-mixed register), while Saaras transcribes the spoken result in Devanagari. Those tokens
count as substitutions however perfectly they were pronounced. The QC report measures and
prints this count — 8 of 42 reference words — rather than arguing it away, so **0.2381 is an
upper bound on the error, and the genuine content error rate on this clip is 1 word in 42
(2.4%)**.

The one real error is worth keeping: `चुन` → `सुन` is a single-phoneme confusion (`ch` vs
`s`) on a short unstressed word. It is exactly the kind of defect this harness exists to
surface, and `qc_report.md` ranks that segment first.

---

## 5. Timing fidelity, and why every segment is flagged

All 4 segments exceed the 5% drift threshold, all in the **underrun** direction (mean signed
drift −60.8%, 0 segments overrunning). This is the M2 tiling consequence described in
`docs/m5-results.md` §5: segment windows cover a phrase *plus the pause after it*, so the
target duration is speech plus silence, and the synthesised line cannot fill it without an
unlistenable slow-down. The loop clamps at 0.85 and stops — correctly.

Underrun is the benign direction: M5 leaves the remainder silent, and the source had a pause
there anyway. Overrun is the direction that breaks sync, and **0 segments overrun**. The
metrics keep the two apart (`duration_fit.overrun_only`) rather than averaging two different
failure modes into one number.

---

## 6. Coverage — asserted, not logged

| Hop | Expected | Actual | Result |
| --- | --- | --- | --- |
| `asr_to_translate` | 4 | 4 | PASS |
| `translate_to_tts` | 4 | 4 | PASS |
| `tts_to_assemble` | 4 | 4 | PASS |
| `assemble_to_qc` | 4 | 4 | PASS |
| `no_empty_segments` | none | none | PASS |

A failure here raises and stops the run. An off-by-one between stages silently destroys
sync, and it is invisible in the audio until someone watches the whole video.

---

## 7. The finding that came out of building this: Bulbul is not duration-deterministic

The cold-cache end-to-end run and the warm-cache run disagreed about per-segment drift on
**byte-identical text with identical parameters** (verified: all 4 transcripts and all 4
translations matched exactly between runs). So the synthesiser was isolated and called three
times with one fixed input at `pace=1.0` (`scripts/tts_variance_probe.py`, fresh cache per
call so every call really hit the network):

| Call | Trimmed duration |
| --- | --- |
| 1 | 6.580 s |
| 2 | 5.260 s |
| 3 | 6.540 s |

**Spread 1.320 s = 21.55% of the mean (σ = 0.751 s). Responses were not byte-identical.**

This matters to the M4 design directly:

- **21.55% run-to-run variance is more than 4× the ±5% convergence band.** A pace computed
  from one measurement cannot be assumed to hold on the next call, so caching a pace per
  phrase and reusing it — an obvious-looking optimisation — would be unsound.
- **The pipeline is unaffected, because it ships the exact audio it measured.** The fit loop
  measures the WAV that came back and places *that same WAV* on the timeline; it never
  re-requests after measuring. Measure-then-ship is correct here; measure-then-re-request
  would not be.
- It also explains why the cold and warm runs report different drift improvements (11.47% vs
  2.88% reduction in mean absolute drift). Both are real measurements of real audio; the
  difference is the synthesiser, not the loop. With n=4 segments and 21% synthesis variance,
  **the drift-improvement figure on this clip is not statistically meaningful**, and the
  README says so rather than quoting the flattering one.

---

## 8. Tests

`tests/test_qc.py` (31 tests). Back-transcription is served by a fake client so the
interesting cases are reachable deterministically — a perfect dub, a mispronounced word, a
window where nothing was spoken. Several tests assert *ordering* (a worse dub must score
worse) rather than a specific WER value, because the ordering is the property the ranked
report depends on.
