# M5 — Assemble + mux

Every number here comes from a real run on `samples/test_clip.mp4` (36.107 s, 4 segments).
Raw data: `output/assemble_report.json`, `output/mux_report.json`, `output/metrics.json`.

```bash
python -m src.pipeline --input samples/test_clip.mp4 --stage assemble
python -m src.pipeline --input samples/test_clip.mp4 --stage mux
```

Neither stage makes an API call, so both cost ₹0 and run offline.

---

## 1. What was built

`src/stages/assemble.py` places the M4 clips on one silent timeline; `src/stages/mux.py`
remuxes that timeline onto the untouched video stream. The mixing primitives
(`silent_int16`, `apply_fade_int16`, `mix_int16`, `write_int16_wav`) live in `src/audio.py`
and use the stdlib `array` module — no numpy, no resampling, and no time-stretching
anywhere in the pipeline.

---

## 2. Placement — measured

| Segment | Window | Placed | Spoken | Fills | Lead trim restored |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.000–5.475 s | 0.030 s | 1.770 s | 32.3% | 0.030 s |
| 1 | 5.475–13.183 s | 5.595 s | 3.540 s | 45.9% | 0.120 s |
| 2 | 13.183–25.270 s | 13.183 s | 7.123 s | 58.9% | 0.000 s |
| 3 | 25.270–36.107 s | 25.280 s | 2.120 s | 19.6% | 0.010 s |

- **Track length:** 36.107083 s against a 36.107062 s source — a 21 µs difference, which is
  one sample at 24 kHz (the rounding in `seconds_to_frames`).
- **Speech:** 14.553 s of 36.107 s, i.e. **40.3% of the timeline is speech and 59.7% is
  silence.** That is not a defect of this stage; it is the M4 underrun, restated
  positionally. See §5.
- **Overlaps:** 0. **Samples clamped while mixing:** 0. **Frames dropped past the end:** 0.
- **Fades:** 360 frames (15 ms at 24 kHz) at each end of every clip.

## 3. Three decisions

**Lead trim is restored, not discarded.** M4 measured each clip with its silence trimmed,
because the synthesiser's padding does not scale with `pace` and would have biased every
correction the fit loop computed. That padding was still part of the clip's own onset, so
placement is at `start + lead_trim_s`. Trimming was a measurement device, not an edit. The
restored offsets are 0–120 ms.

**Overlaps are summed, not truncated.** Where two clips collide the samples are added with
clamping. An overlap that stays audible is one QC can flag; an overlap silently cut is a
missing half-sentence nobody sees. No clip is ever nudged later to make room — that would
convert one late line into a dub that slides progressively out of sync for the rest of the
video.

**The track is the length of the source, not the length of the speech.** So the mux gets an
audio stream matching the video frame for frame, and a dub that ends early cannot silently
shorten the output.

## 4. Mux — the video is provably untouched

`-c:v copy`, so the picture is copied bit for bit. Seven post-mux checks run against
`ffprobe` output and all seven passed:

| Check | Expected | Actual |
| --- | --- | --- |
| `video_codec_name_unchanged` | h264 | h264 |
| `video_width_unchanged` | 640 | 640 |
| `video_height_unchanged` | 360 | 360 |
| `video_pix_fmt_unchanged` | yuv420p | yuv420p |
| `video_frame_count_unchanged` | 1082 | 1082 |
| `audio_track_present` | one stream | aac 24000 Hz mono |
| `duration_matches_source` | 36.107029 s | 36.107083 s (54 µs drift) |

The source audio was `aac 44100 Hz stereo`; the output is `aac 24000 Hz mono`, which is the
dub. Output size 1.86 MB. A failed check raises rather than logging — a remux that silently
re-encoded the video would otherwise look exactly like a success.

**v1 replaces the audio track entirely.** Background music and effects are lost with the
original speech. Preserving them needs source separation (Demucs), a documented non-goal for
v1, and this is stated in `mux_report.json` rather than left for a listener to discover.

## 5. The honest finding: the windows are mostly silence

Every segment underruns its window, by 32.3% to 19.6% fill. The cause is upstream and
structural: **M2 segments tile the timeline**, so each window covers its phrase *plus the
pause that follows it*. On this clip — a card trick with long theatrical pauses — the pauses
dominate. The "target duration" the M4 fit loop is aiming at is therefore not the duration
of the source *speech*; it is the duration of speech plus silence.

Two consequences, both measured rather than argued:

1. The fit loop is asked to stretch a 1.8 s line to fill 5.5 s, which is a 3× slow-down. It
   clamps at 0.85 (correctly — the brief's perceptual limit) and cannot get close. Every
   segment ends up "clamped" and none "converged", which reads like a failing loop but is
   actually the loop refusing to produce unlistenable speech.
2. Sync is nonetheless preserved, because each line still *starts* at the timestamp its
   source phrase started, and the leftover time is silence — which is what the pause was in
   the source anyway.

The fix is not in M5. It is to give M4 a target derived from the **speech-active span**
inside each window (the silence spans are already detected in M2 and are already in
`asr_report.json`) rather than the full tiled window. That is the first item on the
limitations list in the README, and it is deliberately not being done here: it changes the
M2 contract that M3 and M4 were measured against, and it should be measured as its own
before/after rather than folded into this milestone.

## 6. Tests

`tests/test_assemble.py` (26 tests) and `tests/test_mux.py` (12 tests). The assemble tests
build WAVs with the stdlib and verify placement against the *samples* — silence before a
clip's offset, speech after — rather than trusting the report. The mux tests call real
ffmpeg on a 1-second generated clip, because the entire claim of that stage is that
`-c:v copy` passes the picture through unchanged and only a real remux can demonstrate it;
the failure paths (re-encode, dropped frame, truncation, missing audio) are tested against
`_verify` directly with synthetic probes.
