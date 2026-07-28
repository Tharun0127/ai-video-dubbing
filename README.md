# Sarvam Video Dubbing Pipeline

Batch video dubbing (English → Hindi by default) built entirely on Sarvam AI's APIs, with
closed-loop duration fitting so the dub stays in sync with the original video.

**Status: Milestone 1 of 7 complete.** The full README — architecture diagram, measured
metrics table, and honest limitations — is written in M7 from real numbers. Until then,
per-milestone results live in `docs/`.

> Every number in this repo comes from an actual run. Nothing is estimated, interpolated,
> or illustrative. Unmeasured fields serialise as `"not yet measured"`, never as `0`.

## Milestone progress

| | Milestone | Status |
| --- | --- | --- |
| M1 | Skeleton, config, cache, metrics, Sarvam client, CLI | **Done** — [results](docs/m1-results.md) |
| M2 | Demux + silence-boundary chunking + ASR | Not started |
| M3 | Translation with strict segment mapping | Not started |
| M4 | TTS + closed-loop duration fitting | Not started |
| M5 | Assemble + mux | Not started |
| M6 | QC harness (WER/CER, drift, flagging) | Not started |
| M7 | Packaging + README | Not started |

## M1 headline result

One real Saaras v3 call on a 10-second clip, then the identical call again:

| | Run 1 (cold cache) | Run 2 (warm cache) |
| --- | --- | --- |
| Cache | MISS — network call | **HIT** |
| Latency | 0.5381 s | **0.0066 s** (82× faster) |
| Cost | ₹0.083333 | **₹0.000000** |
| Network calls | 1 | **0** — proven with a socket guard that raises on any connect |

Full detail, including the raw API response and the two real bugs found while proving it:
[`docs/m1-results.md`](docs/m1-results.md).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

cp .env.example .env            # then paste your key from https://dashboard.sarvam.ai
```

`ffmpeg` and `ffprobe` must be on `PATH`.

Store the key **bare** — `SARVAM_API_KEY=sk_xxx`, with no angle brackets and no quotes —
and leave `SARVAM_BASE_URL` at `https://api.sarvam.ai` with no `/v1` suffix. Both mistakes
now fail at startup with a message naming the fix, because both cost real debugging time.

## Usage

```bash
python -m src.pipeline \
  --input samples/lecture_30s.mp4 \
  --source-lang en-IN \
  --target-lang hi-IN \
  --output output/dubbed.mp4 \
  [--no-fit]          # disable duration fitting, for the before/after comparison
  [--max-segments N]  # cap segments during development to save credits
  [--dry-run]         # run entirely from cache; fail loudly on any cache miss
  [--stage asr]       # run a single stage and stop
  [--verbose]         # DEBUG logging, including API request/response bodies
```

Or via `make`:

```bash
make test          # pytest, no network access
make verify        # M1 acceptance: real call, then the same call from cache (~Rs 0.083)
make verify-free   # re-prove from the warm cache, Rs 0
make dry-run       # cache-only run
make clean-cache   # delete every cached response (the next run spends credits)
```

## Layout

```
src/
  config.py          typed config, .env loading, confirmed API constants and prices
  cache.py           sha256(stage+model+params+input) -> disk cache
  metrics.py         stage timers, cost accounting, drift stats, metrics.json
  sarvam_client.py   the ONE http client: auth, retries, backoff, cache, cost, latency
  audio.py           ffmpeg/ffprobe wrappers and duration probing
  pipeline.py        CLI entrypoint and stage orchestration
  stages/            demux, asr, translate, tts, assemble, mux (M2-M5)
tests/               pytest; API mocked from captured real responses, real ffmpeg on tiny clips
docs/
  api-notes.md       confirmed Sarvam API shapes -- the source of truth for every call
  m1-results.md      measured M1 outcomes
samples/             licence-clear test clips
```

## Design notes

**Caching is a requirement, not an optimisation.** The cache key is
`sha256(stage + model + canonical_params + sha256(input))`, so changing the audio, the
model, or any single parameter forces a fresh call, while a genuinely identical call is
free forever. `--dry-run` turns a cache miss into a hard failure so later milestones can be
developed without silently re-spending credits on earlier ones.

**One HTTP client.** `sarvam_client.py` owns auth, timeouts, retries with exponential
backoff and full jitter, `Retry-After` handling, 429/503 retry policy, the cache layer,
per-call latency, and cost accounting. No other module makes an HTTP request.

**Duration fitting will clamp to 0.85–1.25×, not the API's 0.5–2.0×** — beyond roughly ±25%
speech stops sounding human, so being deliberately more conservative than the API allows is
the right call. Implemented and measured in M4.

## Testing

```bash
pytest        # 88 tests
```

The Sarvam API is mocked from responses captured verbatim from real calls
(`tests/fixtures/`), which double as API documentation. `ffmpeg` runs only against tiny
generated fixtures.

## Licence and attribution

Test audio in `samples/` is public domain; see `samples/README.md` for the source and
extraction command.
