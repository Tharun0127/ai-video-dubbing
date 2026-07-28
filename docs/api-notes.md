# Sarvam API notes — confirmed shapes

Source of truth for every API call in this repo. Nothing here is assumed: each entry is
either quoted from the live docs or observed in a real response, with the date it was
confirmed. Docs pages are fetched as clean markdown by appending `.md` to any
`docs.sarvam.ai` URL (e.g. `https://docs.sarvam.ai/api/getting-started/pricing.md`).

---

## Global

| Item | Value | Confirmed |
| --- | --- | --- |
| Base URL | `https://api.sarvam.ai` | 2026-07-28, docs + real call |
| Auth header | `api-subscription-key: <key>` | 2026-07-28, real call |
| Auth failure status | **403**, not 401 | 2026-07-28, observed |
| Docs index for LLMs | `https://docs.sarvam.ai/llms.txt` | 2026-07-28 |

### `/v1` is NOT a global prefix — observed failure

Putting `https://api.sarvam.ai/v1` in `SARVAM_BASE_URL` yields:

```json
{"error":{"message":"Not Found","code":"not_found_error",
          "request_id":"20260728_eef00e7f-a750-44ba-8533-6ffcc2b4ae82"}}
```

`/v1` is the prefix for the OpenAI-compatible **chat completions** endpoint only. Speech,
TTS, and translation are served from the root. `Config.__post_init__` now rejects a base
URL ending in `/v1` so this fails at startup with a readable message instead of a 404
mid-run.

### Auth failure shape — observed

A key pasted with its placeholder wrapper (`<sk_...>`) returns:

```json
{"error":{"message":"Invalid or missing authentication credentials",
          "code":"invalid_api_key_error","request_id":"20260728_6f043210-..."}}
```

HTTP 403. `Config.__post_init__` rejects bracket/quote-wrapped keys up front.

### Errors

Error bodies are always `{"error": {"message", "code", "request_id"}}`. `request_id` is
present on errors as well as successes — the client extracts it into `SarvamAPIError`
so a failed call can be quoted to support verbatim.

Documented error codes: `invalid_request_error`, `internal_server_error`,
`unprocessable_entity_error`, `insufficient_quota_error`, `invalid_api_key_error`,
`authentication_error`, `not_found_error`, `rate_limit_exceeded_error`.

Retry policy implemented in `sarvam_client.py`: retry on `{429, 500, 502, 503, 504}` with
exponential backoff + full jitter, honouring `Retry-After` when present; everything else
raises immediately. Docs explicitly say to treat 503 like 429.

---

## Speech-to-Text — `POST /speech-to-text`

`Content-Type: multipart/form-data`. Confirmed 2026-07-28 against the OpenAPI spec at
`https://docs.sarvam.ai/api-reference/speech-to-text/transcribe.md` **and** real calls.

### Request fields

| Field | Type | Notes |
| --- | --- | --- |
| `file` | binary | **required**. WAV, MP3, AAC, AIFF, OGG, OPUS, FLAC, MP4/M4A, AMR, WMA, WebM, PCM |
| `model` | enum | `saaras:v3` (default, recommended) or `saarika:v2.5` (legacy) |
| `mode` | enum | `transcribe` (default), `translate`, `verbatim`, `translit`, `codemix`. **saaras:v3 only** |
| `language_code` | enum | BCP-47, or `unknown` to auto-detect. 23 languages for saaras:v3 |
| `input_audio_codec` | enum | Only *required* for raw PCM (`pcm_s16le`, `pcm_l16`, `pcm_raw`), which must be 16 kHz |
| `with_timestamps` | bool | **Undocumented in the request schema but accepted** — see below |
| `with_diarization` | bool | **Rejected with HTTP 400 on this endpoint** — Batch API only |

### Response fields — observed

```json
{
  "request_id": "20260728_...",
  "transcript": "Almighty God, the same Solomon are forebears prescribed nearly a century and three quarters ago.",
  "language_code": "en-IN"
}
```

Without `with_timestamps`, the response has exactly three keys:
`["language_code", "request_id", "transcript"]`. The OpenAPI schema also defines
`timestamps`, `diarized_transcript`, and `language_probability` as optional/nullable.
`language_probability` is documented to return a value only when `language_code` is
omitted or set to `unknown`.

### `with_timestamps` — confirmed working, but too coarse to build on

Passing `with_timestamps=true` returns HTTP 200 with a fourth key. Observed on the
10-second clip:

```json
{"words": ["Almighty God, the same Solomon are forebears prescribed nearly a century and three quarters ago."],
 "start_time_seconds": [0.0],
 "end_time_seconds": [10.0]}
```

The `words` array contained **one entry spanning the entire 10-second chunk** — the whole
utterance, not per-word or per-sentence timings. So the sync endpoint gives no usable
intra-chunk timing.

**Consequence for M2:** segment boundaries cannot come from the sync endpoint's
timestamps. Either (a) derive boundaries locally from silence detection and treat each
chunk as one segment, or (b) use the Batch API, which documents real timestamps and
diarization. **Decided: (a), chunked REST** — see the M2 section below.

#### Re-confirmed on the M2 chunks, 2026-07-29

Both chunks of `samples/test_clip.mp4` were sent with `with_timestamps=true`. Both came
back with exactly one span covering the whole chunk, confirming the M1 observation is the
endpoint's steady-state behaviour and not an artefact of the 10 s clip:

| chunk | audio sent | spans returned | span extent | request_id |
| --- | --- | --- | --- | --- |
| 0 | 25.270 s | 1 | whole chunk | `20260728_cd422caf-81f1-4c01-8875-f13a366d9578` |
| 1 | 10.837 s | 1 | whole chunk | `20260728_999dcba3-8535-427f-9156-80589aad54ae` |

`local_spans_from_payload` labels this `api_single_span` and falls back to the chunk
extent. It also handles the multi-span shape, so Batch API word timings drop in without a
rewrite — and `tests/test_asr_stitch.py` exercises that path on synthetic spans today.

**Wire format gotcha:** `with_timestamps` is a multipart form field, so a Python `True`
serialises as the string `"True"`, which the API does not accept. `sarvam_client.py`
lowercases booleans to `"true"`/`"false"` on the way out.

### `with_diarization` on the sync endpoint

`with_timestamps=true` + `with_diarization=true` → **HTTP 400**. Consistent with the docs:
"Diarization is only available in Batch API with separate pricing."

### Limits

| Limit | Value | Source |
| --- | --- | --- |
| Max audio per request | **30 s** (HTTP 422 beyond it) | docs |
| Recommended sample rate | 16 kHz (8 kHz telephony also supported) | docs |
| Channels | mono; multi-channel is merged into one | docs |
| Preferred format | WAV 16-bit PCM | docs |

---

## Translation — `POST /translate`

`Content-Type: application/json`. Confirmed 2026-07-29 against
`https://docs.sarvam.ai/api-reference/text/translate-text.md`, the two model pages, and
real calls (`scripts/m3_probe.py`, raw response saved in `output/m3_register_probe.json`).

### Request fields

| Field | Type | Notes |
| --- | --- | --- |
| `input` | string | **required**. Char cap is **per model** — see below |
| `source_language_code` | enum | **required**. 23 codes, plus `auto` (mayura:v1 only) |
| `target_language_code` | enum | **required**. Same set, no `auto` |
| `model` | enum | `mayura:v1` or `sarvam-translate:v1` |
| `mode` | enum | `formal` (default), `modern-colloquial`, `classic-colloquial`, `code-mixed` |
| `speaker_gender` | enum | `Male` / `Female`. Documented as most relevant to `code-mixed` |
| `output_script` | enum \| null | `roman`, `fully-native`, `spoken-form-in-native`. **mayura:v1 only** |
| `numerals_format` | enum | `international` (default) or `native` |

There is **no context parameter and no batch/array parameter.** The request takes exactly
one `input` string. Both were checked for explicitly because SPEC.md asks for context
windows and batching; the consequence is that both have to be built on top of `input`
(numbered lines), which is what `src/stages/translate.py` does.

### Response fields — observed

```json
{
  "request_id": "20260728_f3dde1b6-3ca9-4645-ab2e-75cae23616e3",
  "translated_text": "ठीक है, जय, एक कार्ड चुनो।",
  "source_language_code": "en-IN"
}
```

Exactly three keys. No usage/token counts, so cost has to be derived from
`len(input)` locally, which is what `metrics.cost_translate_inr` does.

### Model capability matrix — this is the whole M3 design constraint

| | `sarvam-translate:v1` | `mayura:v1` |
| --- | --- | --- |
| Input cap | **2000 chars** | **1000 chars** |
| Languages | all 23 | **11** (10 Indian + English) |
| Style modes | **`formal` only** | all four |
| `output_script` | not supported | supported |

The two capabilities are disjoint in exactly the way that matters: the model with the
register control is the one with the smaller cap and fewer languages. Encoded in
`config.TRANSLATE_MODEL_MODES` / `TRANSLATE_MODEL_LANGUAGES` / `TRANSLATE_MODEL_MAX_CHARS`
and enforced in `SarvamClient.translate` *before* the request is built, because passing a
style mode to `sarvam-translate:v1` does **not** error — it silently returns formal
output, which would make a register comparison meaningless.

### Line-structure behaviour under batching — measured, and the two models differ

Four segments sent as one request in the form `[1] …\n[2] …\n[3] …\n[4] …`:

| model / mode | markers returned | newlines in reply |
| --- | --- | --- |
| `sarvam-translate:v1` / formal | 1,2,3,4 | 3 — one line per segment |
| `mayura:v1` / formal | **2,3,4 — marker 1 dropped** | 0 |
| `mayura:v1` / modern-colloquial | 1,2,3,4 | **0 — all on one line** |
| `mayura:v1` / classic-colloquial | 1,2,3,4 | **0** |
| `mayura:v1` / code-mixed | 1,2,3,4 | **0** |

Two consequences, both now covered by tests:

1. **A line-anchored parser does not work.** `mayura:v1` returns every segment joined onto
   a single line, so markers have to be scanned anywhere in the reply, not matched at the
   start of a line.
2. **`mayura:v1` can drop a marker outright** (observed in `formal` mode). There is no safe
   way to realign a reply with a missing marker, so the stage treats any reply that does
   not carry markers `1..N` exactly once, in order, as a failure and re-issues that batch
   one segment per request.

### `numerals_format: native` rewrites the protocol markers

With `numerals_format=native`, `[1] [2] [3] [4]` come back as `[१] [२] [३] [४]` — the
model applies native numerals to the scaffolding as well as to the content. Python's `\d`
is Unicode-aware and `int("१") == 1`, so the parser handles it; this is recorded because
it is exactly the kind of thing that silently breaks a hand-rolled `[0-9]` regex.

### M3 decision — mayura:v1 / classic-colloquial, against SPEC.md's default

SPEC.md names Sarvam-Translate. The measured comparison in `docs/m3-results.md` overrides
that for this source: on conversational speech `sarvam-translate:v1` rendered "top card"
as "सबसे अच्छा कार्ड" (*best* card — it destroys the trick's punchline) and "No way" as
"कोई बात नहीं" (*never mind*). `mayura:v1` in `classic-colloquial` got both right and was
the only variant holding one consistent level of address across all four segments.

The cost of that choice is expansion: `classic-colloquial` has the highest estimated
syllable ratio of the five variants (1.53 mean vs 1.30 for `modern-colloquial`).
`--translate-model` and `--translate-mode` make it a one-flag change if M4 shows the
1.25 pace clamp is being hit constantly.

---

## Pricing (INR)

From `https://docs.sarvam.ai/api/getting-started/pricing.md`, confirmed 2026-07-28.
Encoded in `src/config.py` and applied in `src/metrics.py`.

| Service | Price | Billing unit |
| --- | --- | --- |
| Speech to Text | ₹30 / hour | per second of audio, **rounded up per request** |
| Speech to Text + Diarization | ₹45 / hour | per second, rounded up |
| Speech to Text + Translate | ₹30 / hour | per second, rounded up |
| Sarvam Translate V1 | ₹20 / 10K chars | per character, rounded up |
| Bulbul v2 (TTS) | ₹15 / 10K chars | per character, rounded up |
| Bulbul v3 (TTS) | ₹30 / 10K chars | per character, rounded up (beta pricing) |
| Sarvam-105B | ₹4 in / ₹2.5 cached / ₹16 out | per 1M tokens |

Free tier: ₹100 of credits for new accounts.

**Worked example (measured):** the 10 s clip costs `ceil(10) × 30 / 3600 = ₹0.083333`.

---

## Rate limits

From `https://docs.sarvam.ai/api/getting-started/ratelimits.md`, confirmed 2026-07-28.

- Account-level plan limits: Starter 60 req/min, Pro 200, Business 1,000.
- Per-API concurrency limits also apply; for `bulbul:v3` the Starter limit is 30 concurrent.
- 429 and 503 should both be retried with exponential backoff.
- Bursts of connections can be rejected below the stated concurrency ceilings.

---

## M2 decision — chunked REST, and the tradeoff

**Chosen: (a) chunked REST.** Reasons, in order of weight:

1. **Synchronous and debuggable.** Every chunk is a self-contained request/response pair
   with its own `request_id`, cached under its own key. A failure names the chunk, and a
   re-run costs nothing. Batch would add submit-and-poll state to debug on top of the
   chunking logic that a 30 s cap forces on us anyway.
2. **The Batch API's main advantage does not pay off yet.** Its selling points are real
   timestamps and diarization. Diarization is an explicit v1 non-goal (SPEC.md), and the
   timestamps only matter if segment boundaries come from the API — which, for this
   pipeline, they deliberately do not (see below).
3. **Cost is identical.** STT is ₹30/hour billed per second either way. Chunking bills the
   same total audio; it does not bill per request.

**What we give up:** segment boundaries are only as good as local silence detection, and a
speaker who never pauses for >30 s would force a mid-word cut. The planner detects that
case, warns, and records `cut_source: "hard"` on the chunk rather than hiding it.

**Migration path if timing turns out to be the bottleneck at M6:** `stitch_chunk_transcripts`
already accepts multiple spans per chunk and offsets each one, so swapping in Batch
timestamps means changing only `local_spans_from_payload`.

### Silence threshold — measured, not assumed

`samples/test_clip.mp4` has a **mean volume of -26.0 dB** (ffmpeg `volumedetect`), so the
SPEC's -40 dB default finds only the trailing pause and no interior cut point at all:

| threshold | min pause | silences found | hard cuts |
| --- | --- | --- | --- |
| -40 dB | 0.30 s | 1 (trailing only) | 1 |
| -35 dB | 0.30 s | 1 (trailing only) | 1 |
| **-30 dB** | **0.30 s** | **4** | **0** ← accepted |

`plan_chunks_adaptive` keeps -40 dB as the default and escalates through a fixed ladder
only when the current level would force a hard cut, logging a WARNING and recording every
attempt in `asr_report.json`. Silently shipping a mid-word cut, or silently redefining the
documented default, would both be worse than saying which threshold actually ran.

---

## Not yet confirmed (do before implementing)

These will be filled in at the milestone that needs them, following the same rule:
fetch the docs, make one real call, record what came back.

- **Bulbul v3 TTS** (M4): endpoint path, speaker identifiers, the ~2500-char cap, output
  audio format/sample rate, and above all **the semantics of `pace`** — whether a value
  above 1.0 means faster or slower speech. The duration-fit update rule must be inverted
  if the sense is reversed, so this gets a dedicated real-call check.
- **Batch Speech-to-Text** (only if M6 shows timing is the bottleneck): job submission,
  polling, timestamp granularity, and whether its per-second price differs from the sync
  endpoint.

## Open questions for the project owner

_None outstanding. The M2 ASR-strategy question was answered: chunked REST — see the M2
decision section above._
