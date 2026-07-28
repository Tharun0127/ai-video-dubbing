PROJECT BRIEF

Build a production-quality offline (batch) video dubbing pipeline that takes an input video in one language and produces a dubbed video in a target Indian language, preserving the original speaker's timing. Built entirely on Sarvam AI's APIs.

This is a portfolio project targeting an ML Engineer (Dubbing) role at Sarvam AI, so the code must read like production engineering, not a notebook: typed, modular, cached, instrumented, and tested. The measured numbers this pipeline produces are the deliverable as much as the video is.

Default direction: English → Hindi (en-IN → hi-IN), configurable via CLI. This mirrors Sarvam's real use case of dubbing English technical lectures into Indian languages.

HARD RULES — read these before writing any code
Do not guess the Sarvam API. Before implementing any API call, fetch and read the live docs at https://docs.sarvam.ai. Confirm for each endpoint: exact URL, the auth header name, request body schema, response schema, and limits. If a doc page is ambiguous, write a tiny throwaway script that makes one real call and prints the raw response, then build against what you actually observed. Record confirmed API shapes in docs/api-notes.md as you go.
Never fabricate a number. Every metric in the README, in metrics.json, and in the QC report must come from an actual run of this code. No illustrative, estimated, or placeholder values anywhere — if a number isn't measured yet, leave the field empty and say "not yet measured". I will be asked to defend every number in an interview.
Caching is mandatory, not an optimisation. I'm on limited free API credits. A full run of a 3-minute video costs roughly ₹15–25 (ASR ₹30/hr, TTS ₹30/10k chars, translate ₹20/10k chars), and I'll run this dozens of times during development. Every API response must be cached to disk on first call and reused forever after. A second run over the same input with the same parameters must cost ₹0 and issue zero network calls. Log a cache hit/miss summary at the end of every run.
Checkpoint at every milestone. Stop after each milestone below, run it end to end, show me the real output and the real numbers, and wait for my confirmation before starting the next one. Do not build milestones 1–6 in one pass.
Ask before inventing scope. If a design decision isn't specified here, ask me rather than assuming.
Commit to GitHub after every milestone, with a real commit message. After M1, M2, ..., M7 completes:
Write a concise commit message describing what was built and what the measured results show
Push to a real GitHub repo (create one if you don't have it)
No placeholder commits; every message should tell the story of one completed piece
This is proof of iterative work and serious engineering, not a one-shot dump. Sarvam (and any job application) will read your GitHub history and care more about the granular commits than the final code — they show you think, test, measure, and iterate.
ARCHITECTURE
input.mp4
└─ 1. DEMUX ffmpeg → 16 kHz mono WAV (+ keep original video stream untouched)
└─ 2. ASR Sarvam Saaras v3 → segments [{start, end, text}] with timestamps
└─ 3. TRANSLATE Sarvam-Translate → per-segment target-language text
└─ 4. TTS + FIT Sarvam Bulbul v3 → per-segment audio, closed-loop duration fitting
└─ 5. ASSEMBLE place each segment at its original start time on a silent timeline
└─ 6. MUX ffmpeg → dubbed audio track remuxed onto the original video
└─ 7. QC back-transcribe the dub, score it, emit report + metrics
output/dubbed.mp4 + output/metrics.json + output/qc_report.md

Each stage is a separate module with a clean input/output contract, independently runnable and testable. Intermediate artifacts are written to disk so any stage can be resumed without re-running the prior ones.

THE CORE ENGINEERING PROBLEM (this is what the project is really about)

Translated speech is almost never the same length as the source. English "Let's get started" becomes a noticeably longer Hindi phrase. If you just synthesise it and drop it in, the dub drifts out of sync with the video within seconds and the whole thing falls apart.

The naive fix is post-hoc time-stretching the audio, which produces the chipmunk/robotic artifacts you hear in cheap dubbing. Do not do that. Instead implement closed-loop duration fitting using Bulbul's native pace parameter (supported range 0.5×–2.0×), so speech is generated at the right tempo rather than distorted after the fact:

target_duration = segment.end - segment.start
pace = 1.0

for attempt in 1..3:
audio = tts(text, pace=pace)
actual = duration(audio)
ratio = actual / target_duration

    if 0.95 <= ratio <= 1.05:
        break                                  # good enough, ship it

    proposed = pace * ratio
    pace     = clamp(proposed, 0.85, 1.25)     # perceptual limit, NOT the API limit

    if |proposed - pace| < 0.02:
        break                                  # converged or clamped, stop burning credits

# residual handling

if actual < target_duration: pad with trailing silence
if actual > target_duration: allow overflow only into the gap before the next segment;
otherwise accept the drift and log it as a QC finding

Two details that matter:

Clamp to 0.85–1.25, not to the API's 0.5–2.0. Beyond roughly ±25% speech stops sounding human. Being deliberately more conservative than the API allows is the correct engineering call, and being able to explain why is worth more in an interview than the code itself.
Verify pace semantics from the docs before relying on the formula — confirm whether a higher value means faster or slower speech, and invert the update rule if needed.

Instrument this stage above all others. Record, for every segment, the target duration, the achieved duration, the final pace, and the number of attempts. Then report mean and p95 absolute duration drift with the fit loop disabled vs. enabled. That before/after delta is the single most valuable number this project produces.

Optional stretch (only after milestone 6 works): when pace clamping still can't fit a segment, call Sarvam's LLM to re-translate that segment more concisely ("convey the same meaning in ~20% fewer syllables") and retry. This is exactly the kind of ML-integration judgment the role is about.

STAGE SPECS

1. Demux

ffmpeg to extract audio as 16 kHz mono PCM WAV. Keep the original video stream untouched for remuxing later. Probe and log duration, sample rate, and channel count.

2. ASR (Saaras v3)

Confirm from the docs, but expect: POST /speech-to-text, model saaras:v3, mode transcribe, with timestamps enabled.

Critical constraint: the REST endpoint accepts a maximum of 30 seconds of audio per request. A full video therefore cannot be sent in one call. Handle this with one of two approaches — pick one, and write a short note in docs/api-notes.md explaining the tradeoff:

(a) Chunked REST — split the audio on silence boundaries into sub-30s chunks (never mid-word), call the endpoint per chunk, then stitch the transcripts back together while correcting each chunk's timestamps by its global offset. More code, but synchronous and easy to debug.
(b) Batch API — the async batch endpoint accepts up to 2 hours per file. Fewer edge cases, but you must implement job submission plus polling.

Start with (a) for a short test clip. Output a normalised segments.json: [{id, start, end, text, speaker?}]. Merge segments shorter than ~1s into their neighbour, and split any segment longer than ~15s at a sentence boundary — very short segments make duration fitting unstable and very long ones make drift unrecoverable.

3. Translate (Sarvam-Translate)

Per segment, en-IN → hi-IN. Note the ~2000-character input cap. Batch multiple short segments into one request where the API allows it, to save both credits and wall time. Preserve the segment↔translation mapping exactly — an off-by-one here silently destroys sync, so assert the counts match.

Pass a small window of neighbouring segments as context if the API supports it; translating each segment in isolation loses pronouns and discourse markers.

4. TTS + duration fit (Bulbul v3)

bulbul:v3, target language hi-IN, a fixed speaker for v1, pace driven by the fit loop above. Note the ~2500-character-per-request cap. Implement the loop exactly as specified, and make it toggleable (--no-fit) so the before/after comparison can actually be measured.

5. Assemble

Build a silent audio track the exact length of the source, then place each synthesised segment at its original start timestamp. Detect and log any overlaps. Apply a 10–20 ms fade in/out on each segment to avoid clicks at the boundaries. Output one continuous dubbed WAV.

6. Mux

ffmpeg to combine the original video stream with the new dubbed audio track, copying the video codec (-c:v copy) so there's no re-encode and no quality loss. Output dubbed.mp4.

v1 replaces the audio track entirely. Preserving background music via source separation (Demucs) is a deliberate non-goal for v1 — see the stretch list.

7. QC harness

Score the pipeline's own output — an automated quality loop is explicitly part of the target role:

Semantic fidelity — back-transcribe dubbed.wav with Saaras (transcribe, hi-IN), then compute WER/CER (use jiwer) against the stage-3 translated text. High WER means TTS mispronounced or the ASR chunking corrupted something.
Timing fidelity — per-segment duration drift (target vs. achieved), reported as mean/p50/p95, plus a count of segments exceeding a 5% drift threshold.
Coverage — assert no dropped or empty segments between stages; the counts must match at every hop.
Flagging — emit a ranked list of the worst segments by combined score into qc_report.md, each with its timestamp so I can jump straight to it in the video and listen.
REPO STRUCTURE
sarvam-video-dubbing/
├── README.md # written LAST, from real measured numbers
├── SPEC.md # this document
├── .env.example # SARVAM_API_KEY=
├── requirements.txt
├── Makefile # make run / make qc / make clean-cache
├── src/
│ ├── config.py # dataclass config, env loading, language codes
│ ├── sarvam_client.py # ONE http client: auth, retries, backoff, rate limits, caching
│ ├── cache.py # sha256(stage+model+params+input) → disk
│ ├── audio.py # ffmpeg wrappers, duration probing, silence detection, fades
│ ├── stages/
│ │ ├── demux.py
│ │ ├── asr.py
│ │ ├── translate.py
│ │ ├── tts.py # includes the duration-fit loop
│ │ ├── assemble.py
│ │ └── mux.py
│ ├── qc.py
│ ├── metrics.py # stage timers, cost accounting, drift stats
│ └── pipeline.py # orchestrates stages, resumable
├── tests/ # pytest; mock the API, real ffmpeg on tiny fixtures
├── samples/ # 30–60s test clip, licence-clear
└── output/

sarvam_client.py is the most important file. Every API call goes through it. It owns auth, timeouts, retries with exponential backoff, rate-limit handling (429), the cache layer, per-call latency recording, and cost accounting. Get this right and the rest is straightforward.

CLI
bash
python -m src.pipeline \
 --input samples/lecture_30s.mp4 \
 --source-lang en-IN \
 --target-lang hi-IN \
 --output output/dubbed.mp4 \
 [--no-fit] # disable duration fitting, for the before/after comparison
[--max-segments N] # cap segments during development to save credits
[--dry-run] # run the full pipeline from cache only; fail loudly on any cache miss
[--stage asr] # run a single stage and stop
[--verbose]

--dry-run and --max-segments exist specifically so I can iterate on later stages without burning credits re-running earlier ones. Implement them early, in milestone 1, not as an afterthought.

METRICS TO EMIT (output/metrics.json)
jsonc
{
"input": { "path": "...", "duration_s": 0, "source_lang": "", "target_lang": "" },
"segments": { "count": 0, "mean_duration_s": 0 },
"latency_s": {
"demux": 0, "asr": 0, "translate": 0, "tts": 0, "assemble": 0, "mux": 0, "qc": 0,
"total_wall_clock": 0,
"realtime_factor": 0 // total wall clock ÷ video duration — the headline throughput number
},
"api": {
"calls": { "asr": 0, "translate": 0, "tts": 0 },
"cache_hits": 0, "cache_misses": 0,
"estimated_cost_inr": 0
},
"duration_fit": {
"enabled": true,
"mean_abs_drift_pct": 0, "p50_abs_drift_pct": 0, "p95_abs_drift_pct": 0,
"segments_over_5pct": 0,
"mean_attempts": 0, "mean_final_pace": 0,
"clamped_segments": 0
},
"qc": { "wer": 0, "cer": 0, "flagged_segments": 0 }
}
BUILD ORDER — stop and check in after each milestone

After each milestone, commit to GitHub with this message format:

git commit -m "M{N}: {what was built}

- Real measured outcome / numbers
- Any design decisions made
- Known limitations or TODOs for next milestone

Fixes: #1 (or reference your issue tracker)"

Example:

git commit -m "M1: Sarvam API client + caching layer

- Auth header confirmed for Saaras/Bulbul/Translate endpoints
- Cache layer working: second run of same ASR call costs ₹0 and issues 0 network calls
- Sarvam free tier working, currently at ₹0.50 of ₹100 credit
- TODO: M2 silence-boundary chunking for the 30s REST cap"

M1 — Skeleton + client. Repo structure, config, sarvam_client.py with auth/retry/cache, cache.py, metrics.py, CLI with all flags wired. Prove it with one real ASR call on a 10-second clip; then prove the identical call served from cache with zero network traffic. Done when: one real API round-trip succeeds and a repeat run is a confirmed cache hit. Commit: Push after verified cache hit works.

M2 — Demux + ASR. ffmpeg extraction, silence-boundary chunking under the 30s cap, timestamp stitching, normalised segments.json. Done when: a 30–60s clip yields correct segments with sane timestamps that line up with the video. Commit: After verified ASR output with correct segment boundaries and timestamps.

M3 — Translate. Per-segment translation with strict mapping assertions. Done when: every segment has a faithful translation and counts match exactly. Commit: After verified translation with segment↔translation count matching.

M4 — TTS + duration fit. The closed-loop fitting algorithm, fully instrumented, with --no-fit. Done when: per-segment audio exists and you can show me the drift table with and without fitting. Commit: Include the before/after duration drift metrics in the commit message.

M5 — Assemble + mux. Timeline placement, fades, remux. Done when: dubbed.mp4 plays, is in sync, and sounds natural. Commit: After verified video playback and sync with latency numbers.

M6 — QC harness. Back-transcription, WER/CER, drift stats, qc_report.md. Done when: the pipeline scores its own output and flags the genuinely worst segments. Commit: Include WER/CER scores and flagging results.

M7 — Package. README with a real architecture diagram, the real metrics table, an honest limitations section, and setup instructions verified from a clean clone. Include the before/after duration-drift comparison prominently — that's the headline result. Commit: Final polish, README, verified from fresh clone on another machine if possible.

TESTING

pytest, with the Sarvam API mocked (fixtures captured from real responses — keep them, they're also your API documentation). Real ffmpeg against tiny generated fixtures. Cover at minimum: the duration-fit loop convergence and clamping behaviour, chunk-boundary timestamp stitching, segment count preservation across every stage, and cache key correctness.

NON-GOALS FOR v1 (do not build these unless I say so)

Real-time/streaming, voice cloning, speaker diarization and multi-speaker voice assignment, background music preservation via Demucs, lip-sync, and any web UI. These are the documented stretch list — building them now will prevent v1 from shipping.
