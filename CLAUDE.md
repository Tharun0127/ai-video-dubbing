# CLAUDE.md — Persistent Coding Standards for Sarvam Dubbing Pipeline

This file is the authoritative coding standard for this repo. Every Claude Code session loads this and adheres to it without re-explanation.

---

## Project Context
- **Goal:** Build a production-quality batch video dubbing pipeline (English → Hindi) using Sarvam AI APIs.
- **Portfolio aim:** ML Engineer (Dubbing) role at Sarvam AI.
- **Constraint:** No fabricated numbers. Every metric must be measured from real runs.
- **Deliverable:** A working pipeline + GitHub commit history showing iterative work + a web UI (week 2, optional).

---

## Coding Conventions

### File & Function Structure
- Every new function must have a one-sentence description + type hints on all parameters and return values.
- Example:
  ```python
  def chunk_audio_on_silence(
      wav_path: str, 
      max_chunk_duration_s: float = 30.0,
      silence_threshold_db: float = -40.0
  ) -> list[dict]:
      """Split audio on silence boundaries, preserving timestamps for stitching."""
      ...
  ```

- Every module (`asr.py`, `translate.py`, etc.) starts with a docstring explaining its role:
  ```python
  """
  ASR stage: transcribe audio using Sarvam Saaras v3.
  
  Handles the 30-second REST API cap by chunking on silence boundaries.
  Stitches chunk transcripts together with corrected timestamps.
  Output: segments.json with [{id, start, end, text, speaker?}].
  """
  ```

### Error Handling
- No silent failures. Every API call must include retry logic with exponential backoff (see `sarvam_client.py`).
- All exceptions must be logged with context (which stage, which input, which API call).
- On fatal error, print the full error trace AND the last successful checkpoint so the user can resume.

### Logging
- Use Python's `logging` module, not `print()`.
- Log levels: `DEBUG` for API request/response bodies, `INFO` for stage progress, `WARNING` for degraded behavior (e.g., high duration drift), `ERROR` for failures.
- Every run must emit a summary log at the end showing cache hits/misses, total cost, and wall-clock time.

### Testing
- Every module has a `test_<module>.py` in `tests/`.
- Mock the Sarvam API (store real responses as JSON fixtures, reuse them).
- Real `ffmpeg` only on tiny generated clips (~5 seconds).
- Run `pytest` before every commit.

### Caching
- **All API responses are cached.** Cache key = `sha256(stage + model + params + input_hash)`.
- A second run over the same input with identical parameters must cost ₹0 and issue 0 network calls.
- Log every cache hit/miss at the end of a run in `metrics.json`.

### Metrics & Measurement
- Every stage records its latency (wall-clock time from start to finish).
- Every TTS segment records: target duration, achieved duration, final pace, number of fit attempts.
- Every API call records: endpoint, model, cost (inferred from pricing), latency.
- No metric is estimated, interpolated, or guessed. If it's not measured, leave the field empty and flag "not yet measured".

---

## Branching & Git Strategy

### Main Branches
- `main` — production-ready code. Only merge after a full end-to-end run succeeds.
- `dev` — integration branch. Daily work happens here.

### Feature Branches
- One per milestone: `feat/m1-client`, `feat/m2-asr`, `feat/m3-translate`, etc.
- Branch naming: `feat/<milestone>-<short-name>` or `fix/<issue-number>`.

### Commit Message Format
Every commit must follow this template:

```
<Milestone>: <what was built>

- Measured outcome / real numbers
- Design decisions made
- Known limitations or TODOs

Fixes: #<issue-number> (if applicable)
```

**Example:**
```
M2: Demux + silence-boundary chunking for ASR

- Tested on 60s lecture clip: 12 segments generated
- Silence threshold tuned to -40dB; max chunk 30s
- Segment counts match across ASR stages (12→12)
- TODO: M3 translation per-segment mapping

Fixes: #2
```

### Push Discipline
- After every milestone completes and is tested, push to `dev`.
- Once the full week is done, merge `dev` → `main`.
- Do NOT push to `main` until end-to-end runs successfully.

---

## API Integration Standards

### Before Implementing Any API Call
1. Fetch the live Sarvam docs at `https://docs.sarvam.ai`.
2. Confirm: exact endpoint URL, auth header name, request body schema, response schema, rate limits.
3. Write a tiny throwaway script that makes one real call and prints the raw JSON response.
4. Record the confirmed API shape in `docs/api-notes.md` (this file is your source of truth).
5. Only then implement in the actual pipeline.

### Sarvam Client (`sarvam_client.py`)
- One global HTTP client that owns: auth, retries, backoff, rate limits, caching, cost accounting, latency recording.
- Every API call goes through this client; never make raw HTTP requests elsewhere.
- On 429 (rate limit), back off exponentially. On 5xx, retry up to 3 times with backoff.
- Log the actual API response on DEBUG level (for post-mortem debugging).

---

## QC & Validation Before Commit

### Before Every Commit
1. Run `pytest` — all tests pass.
2. Run the full pipeline on a 30–60s test clip.
3. Verify `metrics.json` has no empty fields (unless explicitly "not yet measured").
4. Verify cache hit/miss summary is logged.
5. Spot-check the output video or QC report for sanity.

### Before Milestone Checkpoint
1. Show real output (segments, audio, metrics).
2. Walk through the measured numbers with me (latency, cost, accuracy).
3. Get explicit sign-off before moving to the next milestone.

---

## Documentation

### Per-Stage
Each stage module (`asr.py`, `tts.py`, etc.) must have:
- Module docstring (role, input, output).
- A comment explaining any non-obvious algorithm (e.g., the duration-fit loop clamping).
- Example input/output in the docstring or a `tests/fixtures/` file.

### Main README (Written in M7)
- Architecture diagram (ASCII or Mermaid).
- Real metrics table from a full run.
- Setup instructions (verified from a fresh clone).
- Limitations section (no hiding failures; be honest).
- GitHub commit history linked (recruiter can click through and see the progression).

### `docs/api-notes.md`
- Accumulated API confirmations as you build each stage.
- This file is the reference if you need to re-implement or debug an API call.

---

## CI/CD Hygiene (Week 2, Optional)

- GitHub Actions workflow (`.github/workflows/test.yml`) that runs `pytest` on every push to `dev`.
- Workflow also runs a 30s test clip end-to-end and reports cost + latency.
- This proves the pipeline still works even after you stop active development.

---

## Code Review Checklist (Before Final PR to Main)

- [ ] All tests pass (`pytest`).
- [ ] No `TODO` or `FIXME` comments left in production code (move them to GitHub issues).
- [ ] All metrics in `metrics.json` are measured (not estimated).
- [ ] Commit history tells a coherent story (7 milestones, 7 commits, each with real outcomes).
- [ ] README is complete and accurate (setup instructions verified from a fresh clone).
- [ ] No API keys or secrets in code or logs (use `.env` and `.gitignore`).
- [ ] Duration-fit loop is fully instrumented and the before/after comparison is in the README.

---

## When Claude Code Gets Stuck

If Claude Code encounters ambiguity or a decision point not covered here:
1. Ask me (the user) for clarification rather than guessing.
2. Log the question in an issue or a comment in the code.
3. Do not proceed with a guess; it will cost credits and time.

**Golden rule:** Assume I would rather have you ask than have you ship the wrong thing.
