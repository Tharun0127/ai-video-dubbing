# M1 results — Sarvam client + caching layer

Every number below was produced by `python -m scripts.m1_verify` on 2026-07-28 and is
reproducible with `make verify` (real call, ~₹0.083) or `make verify-free` (warm cache, ₹0).
The raw artifact is written to `output/m1_verification.json`, which is gitignored because
it is regenerated on every run; this file is the committed record.

## Test input

| Property | Value |
| --- | --- |
| File | `samples/jfk_10s_16k_mono.wav` |
| Duration | **10.000 s** (measured via the stdlib `wave` reader) |
| Format | PCM s16le, 16 kHz, mono, 320,078 bytes |
| Source | JFK inaugural address, Wikimedia Commons — public domain (see `samples/README.md`) |

## The two runs

| | Run 1 (cold cache) | Run 2 (warm cache) |
| --- | --- | --- |
| Cache status | **MISS — network call** | **HIT** |
| Latency | **0.5381 s** | **0.0066 s** |
| Cost | **₹0.083333** | **₹0.000000** |
| Network calls issued | **1** | **0** |
| HTTP status | 200 | — (no request made) |
| HTTP attempts | 1 (no retries needed) | 0 |
| Sarvam `request_id` | `20260728_30c4972c-5350-44a3-a967-21a31f700ee6` | — |

**Cache speedup: 82× faster. Cost avoided on run 2: ₹0.083333.**

Cost derivation, not estimation: STT is ₹30/hour billed per second and rounded up per
request, so `ceil(10.000) × 30 / 3600 = ₹0.083333`.

### How "zero network traffic" was proven

Run 2 executes inside a socket guard that replaces `socket.socket.connect` and
`socket.create_connection` with functions that raise `NetworkAccessError`. Any attempt to
open a connection would fail the run rather than be silently tolerated. Run 2 completed
successfully with the guard armed, and `metrics.cache_misses == 0`.

The cached payload was also compared byte-for-byte against the live response:
`payloads identical across runs: True`.

### Cache key

```
2098b43d20631c7f8c5e535a7cb5a95a95ef5b129f8dd8bda3b0427d06e8252d
```

= `sha256("asr" + "saaras:v3" + '{"language_code":"en-IN","mode":"transcribe","model":"saaras:v3"}' + sha256(audio bytes))`

Stored at `cache/asr/<key>.json`.

## Real API response

```json
{
  "request_id": "20260728_30c4972c-5350-44a3-a967-21a31f700ee6",
  "transcript": "Almighty God, the same Solomon are forebears prescribed nearly a century and three quarters ago.",
  "language_code": "en-IN"
}
```

The ground-truth line is *"…the same solemn oath our forebears prescribed nearly a century
and three quarters ago."* Saaras v3 misheard "solemn oath our" as "Solomon are" on this
1961 archival recording — roughly 3 word errors in 16 words. This is reported as observed,
not smoothed over; M6's QC harness will measure WER properly rather than by eye.

Captured verbatim into `tests/fixtures/asr_saaras_v3_transcribe.json`, where it doubles as
the mock used by the test suite.

## Acceptance checks

| Check | Result |
| --- | --- |
| Run 1 was a real network call | PASS |
| Run 2 was a cache hit | PASS |
| Run 2 issued zero network calls | PASS |
| Run 2 cost ₹0 | PASS |
| Cached payload identical to the live response | PASS |
| `--dry-run` fails loudly on an uncached key | PASS |
| Transcript non-empty | PASS |

`--dry-run` was verified against a deliberately uncached parameter combination
(`mode="verbatim"`), which raised:

```
CacheMissError: --dry-run requested but asr call is not cached
(endpoint=/speech-to-text model=saaras:v3 key=a098296134fc73c4).
Run once without --dry-run to populate the cache.
```

## Test suite

```
88 passed in 0.69s
```

Covers cache key correctness (ordering-independence, None-vs-absent equivalence, and that
stage/model/param/input changes each force a new key), cost arithmetic against the
published price list, retry and backoff behaviour on 429/503/network errors, `Retry-After`
handling, the `--dry-run` contract, secret redaction, and CLI flag wiring. No test opens a
socket.

## Credits spent building M1

Four successful 10-second transcriptions at ₹0.083333 each — **₹0.333 total**: three
cold verification runs plus one `with_timestamps` probe. Three further calls returned
errors and produced no transcription (404 from the `/v1` base URL, 403 from the
bracket-wrapped key, 400 from `with_diarization` on the sync endpoint); these are counted
here but the Sarvam dashboard is the authority on whether failed calls are billed.

## Bugs found and fixed while proving this

1. **`SARVAM_BASE_URL=https://api.sarvam.ai/v1` → HTTP 404.** `/v1` is the prefix for the
   OpenAI-compatible chat endpoint only. `Config` now rejects a `/v1` base URL at startup.
2. **API key stored as `<sk_...>` → HTTP 403.** `Config` now rejects bracket- or
   quote-wrapped keys with a message that names the fix.

Both were real failures hit during M1 and are now covered by tests in `tests/test_config.py`.

## Known limitations carried into M2

- `with_timestamps=true` works on the sync endpoint but returns one span covering the whole
  chunk, so it gives no usable intra-chunk timing. See `docs/api-notes.md` — this is an
  open design decision for M2 (chunked REST vs. the Batch API).
- Only the ASR endpoint is implemented on the client. Translate and TTS arrive in M3/M4,
  and each gets its own docs check plus a real-call confirmation before implementation.
- Every pipeline stage is still unimplemented; the CLI reports which milestone delivers each.
