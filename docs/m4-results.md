# M4 — TTS synthesis + closed-loop duration fitting

Every number here comes from a real run on `samples/test_clip.mp4` (36.107 s, 4 segments).
Raw data: `output/m4_pace_probe.json`, `output/tts_report.json`, `output/metrics.json`.

```bash
python -m scripts.m4_probe                                    # pace-semantics probe
python -m src.pipeline --input samples/test_clip.mp4 --stage tts
python -m src.pipeline --input samples/test_clip.mp4 --stage tts --no-fit
```

Both are fully cached: a repeat run issues zero network calls and costs ₹0.

---

## 1. `pace` semantics — confirmed by measurement, not by reading

**Higher `pace` produces shorter audio. Higher = faster. `pace *= ratio` has the correct
sign and does not need inverting.** Six-point sweep, durations measured from the returned
WAV frames — full table in `docs/api-notes.md`.

Two further findings from the same sweep that shaped the implementation:

**The response is not proportional, and it saturates.** Elasticity
`d(log duration)/d(log pace)` is −1.18 to −1.57 inside the perceptual clamp, then collapses
to **−0.23 between pace 1.25 and 1.50**. So `pace *= ratio` systematically *overshoots*
(it assumes −1.0), and the loop approaches by oscillating rather than from one side. It
still converges — overshoot with |elasticity| < 2 is a contraction — but it justifies the
3-attempt ceiling rather than expecting one-shot convergence.

It also gives SPEC.md's 0.85–1.25 clamp an independent justification beyond perception:
**above 1.25 the extra API range buys almost no duration change anyway**, so clamping
gives up far less than its width suggests.

**Padding is small but not negligible.** Leading/trailing silence measured 0.000–0.133 s
and does not scale with pace. On one clip that pad was **5.7% of the total — larger than
the ±5% convergence band**, so an untrimmed measurement could flip a segment's verdict on
its own. Trimming before measuring is load-bearing, not cosmetic.

---

## 2. What was built

`src/stages/tts.py`, plus `SarvamClient.text_to_speech()` as the only path to
`POST /text-to-speech`, and in-memory WAV helpers in `src/audio.py`.

- **Attempt 1 is always pace=1.0 and IS the unfitted baseline.** It is persisted
  separately (`seg_NNN_nofit.wav`) and reused for the before/after comparison — never
  re-synthesised. Both columns therefore cover identical segments and identical text, and
  the comparison costs zero extra credits.
- **Duration is measured from the returned audio**, via frame count ÷ frame rate through
  the stdlib `wave` module. The response reports no duration anyway, which settles it.
- **Silence is trimmed before measuring**, with the lead/trail offsets kept per segment
  because M5 must add the lead pad back when placing the clip.
- **`pace` is part of the cache key**, so every attempt is its own entry and re-running the
  whole loop is free.
- **The clamp is 0.85–1.25**, deliberately tighter than the API's 0.5–2.0.

### One deviation from the brief, forced by measurement

The brief implies shipping the final attempt. The stage **ships the attempt closest to
target instead**, because on the real clip a slower pace sometimes returned *shorter*
audio (§5). Shipping "the last attempt" would have shipped the worse clip on segment 0.

---

## 3. Before/after drift table — the headline result

Paired over the same 4 segments; "no-fit" is each segment's own pace=1.0 attempt.

| id | window | no-fit dur | drift | fitted dur | drift | final pace | attempts | converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5.475 s | 1.770 s | −67.7% | 1.770 s | −67.7% | 1.000 | 2 | no · **clamped** |
| 1 | 7.708 s | 3.390 s | −56.0% | 3.540 s | −54.1% | 0.850 | 2 | no · **clamped** |
| 2 | 12.087 s | 7.110 s | −41.2% | 7.123 s | −41.1% | 0.850 | 2 | no · **clamped** |
| 3 | 10.837 s | 1.560 s | −85.6% | 2.120 s | −80.4% | 0.850 | 2 | no · **clamped** |

| absolute drift | mean | p50 | p95 | max | >5% |
| --- | --- | --- | --- | --- | --- |
| **without fit** | 62.62% | 61.85% | 82.91% | 85.60% | 4/4 |
| **with fit** | 60.81% | 60.87% | 78.52% | 80.44% | 4/4 |

**n = 4 — small sample.** p50 and p95 are reported because SPEC.md asks for them, but at
n=4 a p95 is an interpolation between the top two values, not evidence. They are flagged
`"small_sample": true` in `metrics.json` itself.

**Clamp rate: 4/4 = 100%. Converged: 0/4 = 0%. Mean attempts 2.00, mean final pace 0.887.**

The fit loop improved 3 of 4 segments and worsened none, but the total improvement is
**1.80 percentage points (62.62% → 60.81%, a 2.9% relative reduction)**. That is a poor
result, and the reason is not the loop.

---

## 4. Why the clamp rate is 100%: the loop is being asked the wrong question

**Every segment underruns. None overruns.** Signed drift is −41% to −86%; the mean signed
drift is −60.8%. The synthesised Hindi is far *shorter* than its window, so the loop tries
to **slow speech down to fill silence**, immediately proposes paces of 0.14–0.59, and is
clamped to the 0.85 floor on every segment.

This is M3's prediction arriving from the opposite direction, and it is a property of the
segmentation, not of the synthesiser. M2's segments **tile the entire timeline including
pauses** — deliberately, so the pause is part of the fit loop's time budget. But that makes
the windows mostly silence: measured source syllable rate was 0.74–1.32 syl/s against 4–6
for ordinary speech. Segment 3 is a 10.8 s window holding roughly 2 s of speech.

**Slowing speech to 0.85 to fill a 10.8 s window with a 2 s line is not a dub anyone
wants.** The correct handling of an underrun is the one SPEC.md already specifies —
*"if actual < target_duration: pad with trailing silence"* — which costs nothing in sync
terms.

Counting only the direction that actually breaks sync:

| overrun-only | without fit | with fit |
| --- | --- | --- |
| mean overrun | **0.00%** | **0.00%** |
| segments overrunning | **0 / 4** | **0 / 4** |

**Measured in the terms that matter for sync, this clip is already perfect without any
fitting at all**, and the 8 synthesis calls the loop made could have been 4. Both framings
are now recorded in `metrics.json` (`with_fit`/`without_fit` and `overrun_only`), because
reporting only the symmetric number would overstate the problem and reporting only the
overrun number would hide the loop doing nothing useful.

### Recommendation for M5 (your call, not made unilaterally)

**Gate the correction on overrun only:** act when `ratio > fit_ratio_max`, accept any
underrun and let M5 pad it. On this clip that would mean 4 synthesis calls instead of 8,
byte-identical shipped audio, a clamp rate of 0%, and an honest headline number.

I did not implement it, because the brief specified the symmetric loop and this is a
design decision about what "fit" means, not a bug. It is a one-line change in
`fit_segment`.

**The loop itself is correct and is verified** — the mocked tests drive an overrunning
segment and show it corrects to exactly the target in one step. This clip simply never
presents that case. **Do not tune the loop on this clip**: it exercises only the underrun
path, which is the path that doesn't matter.

---

## 5. The other finding: `pace` is unreliable for small slow-downs

Moving pace 1.0 → 0.85 (asking for ~18% *longer* audio) on the four real segments:

| segment | pace 1.00 | pace 0.85 | change |
| --- | --- | --- | --- |
| 0 | 1.770 s | 1.680 s | **−5.1% — the wrong direction** |
| 1 | 3.390 s | 3.540 s | +4.4% |
| 2 | 7.110 s | 7.123 s | +0.2% |
| 3 | 1.560 s | 2.120 s | +35.9% |

The controlled sweep shows pace works in aggregate, but **per utterance the slow-down
response is erratic and can invert.** Segment 0 got shorter when asked to slow down;
segment 2 barely moved; segment 3 moved twice as much as requested.

Two conclusions:

1. **Ship the closest attempt, not the last one.** Already implemented (§2). Without it,
   segment 0 would have shipped the worse of its two clips.
2. **Do not build anything on "lower pace lengthens this line."** For M5's residual
   handling, silence padding is reliable and pace-based lengthening is not.

The compression direction (pace > 1.0), which is the one that matters for a real dub,
was not exercised by this clip at all and remains unverified on real material.

---

## 6. Per-segment record

Every field SPEC.md and the brief ask for, in `output/tts_report.json`:

```json
{
  "segment_id": 3,
  "target_duration_s": 10.837062,
  "baseline_duration_s": 1.56,
  "final_duration_s": 2.12,
  "final_pace": 0.85,
  "attempts": 2,
  "converged": false,
  "clamped": true,
  "residual_drift_pct": -80.4374,
  "baseline_drift_pct": -85.6049,
  "stop_reason": "pinned_at_clamp",
  "attempt_log": [ { "attempt": 1, "pace": 1.0, "duration_s": 1.56, "ratio": 0.144,
                     "audio_untrimmed_duration_s": 1.621, "audio_lead_trim_s": 0.01,
                     "audio_trail_trim_s": 0.051, "cache_hit": false }, ... ]
}
```

`stop_reason` distinguishes `converged`, `pinned_at_clamp`, `max_attempts`,
`pace_step_below_threshold`, and `fit_disabled`, so a segment's outcome never has to be
inferred from the numbers.

**Note on `pace_step_below_threshold`:** with the default ±5% band it is unreachable — the
smallest out-of-band ratio moves pace by at least 0.05 × 0.85 = 0.0425, above the 0.02
threshold. It becomes reachable when the band is tightened, which is when it earns its
keep, and the test exercises it that way.

---

## 7. Outputs

`output/audio_segments/` — 8 WAV files, 24 kHz mono 16-bit:

- `seg_NNN.wav` — the fitted clip to ship
- `seg_NNN_nofit.wav` — the pace=1.0 baseline

For segment 0 these are identical (attempt 1 was the closest), flagged
`"identical_to_baseline": true`. Each file entry also carries `lead_trim_s` / `trail_trim_s`
so M5 can restore the offsets rather than starting each clip early.

---

## 8. Cost and cache

| | network calls | cost |
| --- | --- | --- |
| Pace probe (6 paces) | 6 | ₹0.5940 |
| TTS stage, first run (4 segments × 2 attempts) | 8 | ₹1.1640 |
| **M4 total** | **14** | **₹1.7580** |
| TTS stage, repeat run | **0** | **₹0.0000** |
| `--no-fit` run | **0** | **₹0.0000** |

The repeat ran under `--dry-run`, which raises `CacheMissError` on any miss, so zero
network calls is enforced rather than observed.

**`--no-fit` issuing zero network calls is itself the proof that the baseline reuse is
sound.** Every pace=1.0 request it made was already in the cache from attempt 1 of the
fitted run — identical cache keys mean identical requests, so the "before" column really is
the same call the unfitted path would make.

Running total across M1–M4: roughly **₹8.3** of the ₹100 free tier.

---

## 9. Tests

`pytest`: **265 passed**, 51 new in `tests/test_tts.py`. Zero network calls.

The fit loop is driven by a synthetic synthesiser whose duration response to pace is
controlled by the test, which is what makes every branch reachable deterministically —
the real API cannot be asked to overrun by exactly 20% on demand. Covered:

- convergence in one attempt; convergence after one correction (overrun **and** underrun,
  so a sign error cannot pass)
- **the clamp path**, both directions, asserting the pace never leaves [0.85, 1.25] and
  that a pinned segment stops instead of spinning
- **the non-convergence path**, stopping at `fit_max_attempts` without clamping
- the sub-threshold-step guard, under a tightened band where it is reachable
- shipping the closest attempt when a later attempt is worse (the real segment-0 case)
- trimming: padding removed, offsets retained, all-silent audio flagged not deleted
- `--no-fit` making exactly one pace=1.0 call, and matching the fitted run's attempt 1
- cache: pace in the key, whole-loop replay free
- client validation: pace range, speaker, language, 2500-char cap, missing/corrupt audio

The synthetic model is duration = base / pace, i.e. exactly proportional. That is the
*ideal* the update rule assumes; the real API's measured non-linearity lives in
`docs/api-notes.md` rather than in a test, because a test encoding today's measured
elasticity would fail the day Sarvam retunes the model.

---

## 10. Limitations

- **The compression direction is untested on real material.** Every segment on this clip
  underran, so pace > 1.0 — the direction that matters for a real dub — was never
  exercised end to end. A dense, wall-to-wall-speech clip is needed.
- **n = 4.** p50/p95 are arithmetic, not evidence, and are flagged as such in the file.
- **100% clamp rate is an artefact of segmentation**, not a synthesiser limitation (§4).
- **Elasticity measured on one sentence.** The sign generalises; the magnitudes do not
  necessarily.
- **Speaker is fixed** (`shubh`) and unevaluated — voice selection and cloning are
  explicit v1 non-goals.
- **No listening test has been done.** Every quality claim here is about duration, not
  about how the audio sounds.
