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
diarization. This is an open decision — see "Open questions" below.

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

## Not yet confirmed (do before implementing)

These will be filled in at the milestone that needs them, following the same rule:
fetch the docs, make one real call, record what came back.

- **Sarvam-Translate** (M3): endpoint path, request schema, the ~2000-char cap, whether a
  context window of neighbouring segments is supported, and whether several segments can
  be batched into one request.
- **Bulbul v3 TTS** (M4): endpoint path, speaker identifiers, the ~2500-char cap, output
  audio format/sample rate, and above all **the semantics of `pace`** — whether a value
  above 1.0 means faster or slower speech. The duration-fit update rule must be inverted
  if the sense is reversed, so this gets a dedicated real-call check.
- **Batch Speech-to-Text** (M2, if chosen): job submission, polling, timestamp granularity,
  and whether its per-second price differs from the sync endpoint.

## Open questions for the project owner

1. **M2 ASR strategy.** The sync endpoint's `with_timestamps` returns one span per chunk,
   so chunked-REST segment boundaries would be exactly the silence-split boundaries — the
   timing is only as good as the local silence detection. The Batch API documents real
   timestamps plus diarization. Chunked REST is simpler and synchronous; Batch needs
   submit-and-poll but gives genuinely better timing. Which do we build?
