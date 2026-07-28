# Milestone 2 — Demux + silence chunking + ASR: measured results

Every number here comes from a real run on `samples/test_clip.mp4` (2026-07-29).
Reproduce with:

```bash
make demux            # ffmpeg extraction
make asr              # chunk, transcribe, stitch, post-process
make verify-m2        # the six-check acceptance harness
```

---

## Input

| Property | Source | Extracted |
| --- | --- | --- |
| Container / codec | MP4 / **aac** | WAV / **pcm_s16le** |
| Sample rate | **44 100 Hz** | **16 000 Hz** |
| Channels | **2 (stereo)** | **1 (mono)** |
| Duration | 36.107 s | 36.107062 s |
| Video | h264 640x360 | untouched (`-vn`; M5 remuxes with `-c:v copy`) |

A genuine resample + downmix + re-encode, not a container passthrough.

---

## 1. Chunking — 2 chunks

At 36.107 s against the 30 s REST cap, one request is impossible, so this run exercises
the real multi-chunk stitching path.

| chunk | start | end | duration | boundary from |
| --- | --- | --- | --- | --- |
| 0 | 0.000000 s | 25.270219 s | **25.270250 s** | silence midpoint |
| 1 | 25.270219 s | 36.107062 s | **10.836813 s** | end of file |

Both under the 30 s cap. The boundary is the **midpoint of the detected pause
25.062313–25.478125 s** (0.415812 s long), leaving **0.207906 s of margin to speech on
either side**.

### Why -30 dB and not the -40 dB default

The clip's mean volume is **-26.0 dB**, so its noise floor sits above -40 dB. The planner
escalates only as far as it must, and records every attempt:

| level | threshold | min pause | silences found | hard cuts | accepted |
| --- | --- | --- | --- | --- | --- |
| 0 | -40.0 dB | 0.30 s | 1 (trailing only) | 1 | no |
| 1 | -35.0 dB | 0.30 s | 1 (trailing only) | 1 | no |
| 2 | **-30.0 dB** | 0.30 s | **4** | **0** | **yes** |

A WARNING is logged when this happens. `--silence-threshold-db` overrides the default.

### Why the cut is at 25.27 s and not nearer the middle

The planner balances (`target = remaining / chunks_left` = 18.05 s) rather than greedily
filling to 30 s. Of the four detected pauses, only the one at 25.270 s lies in the
admissible window [6.107 s, 30 s] — 5.475 s would strand a 30.6 s tail, and 34.786 s is
past the cap. So 25.270 s was the only legal choice, not a greedy one.

---

## 2. `output/segments.json`

```json
[
  { "id": 0, "start": 0.0,    "end": 5.475,  "text": "Okay, Jay, pick a card.",                                             "speaker": null },
  { "id": 1, "start": 5.475,  "end": 13.183, "text": "Remember it. Right, Jay, shuffle the cards.",                          "speaker": null },
  { "id": 2, "start": 13.183, "end": 25.27,  "text": "Jay, your card is the top card. No way. That's amazing. Really? Yeah.", "speaker": null },
  { "id": 3, "start": 25.27,  "end": 36.107, "text": "How did you do that? I'm amazed.",                                      "speaker": null }
]
```

Provenance (from `output/asr_report.json`) — how each segment's **start** was derived:

| id | span | duration | start boundary from | chunk |
| --- | --- | --- | --- | --- |
| 0 | 0.000 → 5.475 | 5.475 s | file start | 0 |
| 1 | 5.475 → 13.183 | 7.708 s | sentence boundary **snapped to a real pause** | 0 |
| 2 | 13.183 → 25.270 | 12.087 s | sentence boundary, time **estimated** | 0 |
| 3 | 25.270 → 36.107 | 10.837 s | chunk seam (silence midpoint) | 1 |

Only segment 2's boundary time is interpolated (no detected pause within 1.5 s of the
sentence end); it is labelled `sentence+estimated` rather than passed off as measured.

---

## 3. Seam verification — chunk 0 | chunk 1 at 25.270219 s

### Timing

```
gap_or_overlap_s   : 0.000000000
contiguous         : true
inside a silence   : true  (25.062313 – 25.478125 s, 0.415812 s long)
margin to speech   : 0.207906 s on each side
cut_source         : silence  (not a forced 30 s cap cut)
```

`segment[2].end == segment[3].start == 25.270219` exactly.

### No duplicated or dropped words — proved at the sample level

The text check:

```
chunk 0 ends   : "... No way. That's amazing. Really? Yeah."
chunk 1 begins : "How did you do that? I'm amazed."
```

A clean sentence boundary, nothing repeated, nothing missing.

But text inspection is weak evidence. The decisive check is that the chunk WAVs
**concatenate to a byte-identical copy of the demuxed source**:

| | bytes | frames | duration |
| --- | --- | --- | --- |
| source `audio_16k_mono.wav` | 1 155 426 | 577 713 | 36.107062 s |
| `chunk_000.wav` | 808 648 | 404 324 | 25.270250 s |
| `chunk_001.wav` | 346 778 | 173 389 | 10.836813 s |
| **chunks joined** | **1 155 426** | **577 713** | **36.107062 s** |

```
frames dropped or duplicated : 0
sha256 source                : 005025997cff6477dfd640e000802718f7ba9d97b67e380a0dead35567f288cf
sha256 chunks joined         : 005025997cff6477dfd640e000802718f7ba9d97b67e380a0dead35567f288cf
```

Every sample of the source was sent to the API exactly once. No word *could* be dropped or
duplicated at the seam, independent of what the transcript happens to say. This is why
slicing uses the stdlib `wave` module rather than an ffmpeg subprocess — deterministic,
sample-accurate output is what makes the claim checkable.

---

## 4. Post-processing — 2 segments before, 4 after

| stage | count |
| --- | --- |
| after stitching | **2** |
| after merging segments < 1.0 s | **2** (0 merges — none were short) |
| after splitting segments > 15.0 s | **4** (2 splits) |

Chunk 0's 25.270 s segment exceeded the 15 s limit and was split twice, at sentence
boundaries, into 5.475 s + 7.708 s + 12.087 s. Chunk 1's 10.837 s segment already fit.

Final state: **0 segments under 1.0 s, 0 segments over 15.0 s.** Mean segment duration
9.027 s.

The merge path found nothing to do on this clip — it is covered by unit tests instead
(`tests/test_asr_stitch.py`), not left unverified.

---

## 5. Cost and cache

| Run | Network calls | Cache hits | Cost |
| --- | --- | --- | --- |
| 1st (cold cache) | **2** | 0 | **₹0.3083** |
| 2nd (identical) | **0** | **2** | **₹0.0000** |
| 3rd (`--dry-run`, sockets blocked) | **0** | **2** | **₹0.0000** |

Cost check: `ceil(25.27) × 30/3600 + ceil(10.84) × 30/3600 = 26×30/3600 + 11×30/3600 =
₹0.216667 + ₹0.091667 = ₹0.308333`. Matches the measured ₹0.3083.

Cache avoided ₹0.3083 on each repeat run. Stage latency: **1.103 s cold → 0.160 s warm**
(mean network latency 0.475 s over 2 calls). Demux: 0.110 s.

Run 3 executes under a socket guard that raises on any outbound connection attempt, so
"zero network calls" is enforced rather than merely counted.

---

## 6. Acceptance harness

`python -m scripts.m2_verify` — 20 checks across 6 groups, **all passing**:
chunk plan, audio integrity, seams, segment timeline, post-processing bounds, cache.

`pytest` — **141 tests passing** (88 from M1, 53 added in M2), no network access.

---

## Known limitations (honest list)

1. **Intra-chunk timing is not measured.** The sync endpoint returns one span per chunk,
   so boundaries inside a chunk come from silence detection, and the one split with no
   nearby pause (segment 2) has an interpolated timestamp. Flagged as
   `sentence+estimated` in `asr_report.json`. Fix if needed: Batch API — the stitching
   code already handles multi-span input.
2. **The -40 dB default did not work on this clip** and was relaxed to -30 dB
   automatically. On noisier source material the ladder could relax further than is ideal;
   it stops at -25 dB and warns.
3. **A speaker who never pauses for over 30 s forces a mid-word cut.** Detected, warned,
   and recorded as `cut_source: "hard"` — not silently accepted. Did not occur here.
4. **`speaker` is always `null`.** Diarization is a v1 non-goal and is Batch-API-only.
5. **One test clip.** These numbers describe `samples/test_clip.mp4` only; the merge path
   and the hard-cut path have unit-test coverage but no real-audio measurement yet.
