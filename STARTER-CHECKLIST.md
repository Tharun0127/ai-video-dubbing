# Sarvam Dubbing Pipeline — 7-Day Starter Checklist

**Timeline:** 7 days (or 8–10 if you add the web UI in week 2)

**Outcome:** A working pipeline, 7 GitHub commits, real metrics, a demo video.

---

## BEFORE YOU START — 1 HOUR (Day 0, Today)

- [ ] **Create GitHub repo** named `sarvam-video-dubbing` (public, so it's a portfolio piece)
  - [ ] Add `.gitignore`: `__pycache__/`, `.env`, `*.pyc`, `cache/`, `output/`, `*.mp4`, `node_modules/`
  - [ ] Create empty `docs/`, `tests/`, `samples/` directories
  - [ ] Create `README.md` with one line: "Coming soon: batch video dubbing with Sarvam AI"

- [ ] **Create Sarvam API key** at `dashboard.sarvam.ai`
  - [ ] Confirm you have ₹100 free credits available
  - [ ] Copy the API key

- [ ] **Create `.env` file** (locally, never commit this):
  ```
  SARVAM_API_KEY=<your_key_here>
  SARVAM_BASE_URL=https://api.sarvam.ai/v1
  ```

- [ ] **Install Claude Code in VS Code** (you said you already did this ✓)

- [ ] **Clone this checklist + CLAUDE.md + SPEC.md into your repo:**
  ```bash
  cd sarvam-video-dubbing
  git init
  # Copy CLAUDE.md, SPEC.md, STARTER-CHECKLIST.md into the repo root
  git add .
  git commit -m "Initial: project scaffold + standards"
  git branch -M main
  git remote add origin https://github.com/<your-username>/sarvam-video-dubbing.git
  git push -u origin main
  ```

- [ ] **Verify ffmpeg is installed:**
  ```bash
  ffmpeg -version
  ffprobe -version
  ```
  If not: `brew install ffmpeg` (Mac) or `sudo apt install ffmpeg` (Linux).

---

## MILESTONE 1 — API Client + Caching (Day 1–2)

### Before writing code
- [ ] **Read SPEC.md** (especially the "HARD RULES" section and the API integration section of CLAUDE.md)
- [ ] **Fetch Sarvam API docs** at https://docs.sarvam.ai and read:
  - [ ] Speech-to-Text endpoint (request/response schema, auth header, 30s limit, modes)
  - [ ] Text-to-Speech endpoint (request/response schema, pace parameter semantics)
  - [ ] Translate endpoint (request/response schema, language codes)
- [ ] **Create `docs/api-notes.md`** with confirmed API shapes (copy from docs, add confirmations as you go)

### Claude Code Prompt (Day 1)
Open Claude Code in VS Code and paste this prompt (or the full SPEC.md):

```
You are building a batch video dubbing pipeline for Sarvam AI as a portfolio project.

Read CLAUDE.md and SPEC.md in the repo root — they are your authoritative standards and spec.

Start with Milestone 1 only:
- Repo structure (src/, tests/, docs/, etc.)
- config.py: load SARVAM_API_KEY from .env via python-dotenv
- sarvam_client.py: ONE HTTP client that owns auth, retries, backoff, caching, cost accounting
- cache.py: sha256(stage+model+params+input_hash) → disk cache
- metrics.py: stage timers, cost tracking, drift stats
- A CLI entrypoint (main.py or __main__.py) that accepts --input, --source-lang, --target-lang, --dry-run, --max-segments

Prove it works: make one real Saaras ASR call on a 10-second audio file (or generate one), 
cache the response, then make the identical call again and confirm it's served from cache with zero network traffic.

Output: Show me the measured latency, cache status, and cost for both runs.

Stop and ask me for confirmation before proceeding to M2.
```

### Deliverables (Day 2)
- [ ] Repo structure complete
- [ ] `sarvam_client.py` passes one real API call (ASR on 10s clip)
- [ ] Cache working: second call served from disk, zero network traffic
- [ ] `metrics.json` shows cache hit/miss summary and cost
- [ ] All code has type hints and docstrings (check against CLAUDE.md)
- [ ] `pytest` passes (even if just one dummy test)

### Commit (Day 2 end)
```bash
git checkout -b feat/m1-client
# (after Claude Code finishes and you've tested)
git add -A
git commit -m "M1: Sarvam API client + caching layer

- Auth confirmed for Saaras/Bulbul/Translate endpoints
- Cache working: second call ₹0, 0 network calls
- Sarvam free tier accessed, ₹0.50 spent of ₹100 credit
- CLI wired with all flags (--dry-run, --max-segments, etc.)
- TODO: M2 silence-boundary chunking for 30s REST cap

Milestone 1 checkpoint complete."
git push origin feat/m1-client
```

---

## MILESTONE 2 — Demux + ASR (Day 3)

### Before Claude Code
- [ ] Download a ~30–60s test clip (a lecture, a podcast, anything clear audio)
  - Save to `samples/test_clip.mp4`
- [ ] Verify it plays: `ffplay samples/test_clip.mp4`
- [ ] Confirm `docs/api-notes.md` has the Saaras endpoint schema

### Claude Code Prompt
```
Milestone 2: Audio demux + silence-boundary chunking + ASR.

Per SPEC.md:
- Use ffmpeg to extract audio as 16 kHz mono PCM WAV
- Detect silence boundaries (tunable threshold, e.g., -40dB)
- Chunk the audio into sub-30s segments on silence boundaries (never mid-word)
- Call Saaras v3 ASR per chunk in REST mode (mode='transcribe')
- Stitch chunk transcripts together, correcting each chunk's timestamps by its global offset
- Output: segments.json with [{id, start, end, text, speaker?}]
- Merge segments <1s into neighbors; split segments >15s at sentence boundaries

Run on samples/test_clip.mp4. Show me the segment count and timestamps.
Stop and ask for confirmation before M3.
```

### Deliverables (Day 3)
- [ ] `src/stages/demux.py` working
- [ ] `src/stages/asr.py` working
- [ ] `samples/test_clip.mp4` produces sane segments in `output/segments.json`
- [ ] Segment timestamps verified to match the video (pick one, play it in the video, confirm the transcribed text matches)
- [ ] All tests pass

### Commit (Day 3 end)
```bash
git checkout -b feat/m2-asr
git add -A
git commit -m "M2: Demux + silence-boundary chunking + ASR

- Tested on 60s lecture clip: 12 segments extracted
- Silence threshold -40dB; max chunk 30s before silence boundary
- Timestamps verified: segment times align with video content
- Segment count 12→12 (no loss across stages)
- TODO: M3 per-segment translation + mapping

Milestone 2 checkpoint complete."
git push origin feat/m2-asr
```

---

## MILESTONE 3 — Translate (Day 4)

### Claude Code Prompt
```
Milestone 3: Per-segment translation (en-IN → hi-IN).

Per SPEC.md and docs/api-notes.md:
- For each segment from segments.json, call Sarvam-Translate
- Handle the ~2000-character input cap by batching short segments where possible
- Preserve exact segment↔translation mapping; assert counts match after translation
- Output: extend segments.json to [{id, start, end, text, translation}]

Run on the same test clip from M2. Assert segment count before and after is identical.
Show me the translation output for 3 example segments.
Stop and ask for confirmation before M4.
```

### Deliverables (Day 4)
- [ ] `src/stages/translate.py` working
- [ ] `output/segments.json` now has `translation` field for each segment
- [ ] Spot-check 3 segments: original text (English) and translation (Hindi) look correct
- [ ] Segment counts verified identical before/after translation
- [ ] All tests pass

### Commit (Day 4 end)
```bash
git checkout -b feat/m3-translate
git add -A
git commit -m "M3: Per-segment translation (en-IN → hi-IN)

- All 12 segments translated
- Segment count preserved: 12→12
- Spot-check: 'Let's get started' → '[Hindi translation]', verified natural
- TODO: M4 TTS + duration fitting

Milestone 3 checkpoint complete."
git push origin feat/m3-translate
```

---

## MILESTONE 4 — TTS + Duration Fitting (Day 5)

### This is the heart of the project
Before Claude Code, **study the duration-fit algorithm** in SPEC.md:
- Target duration = segment.end - segment.start
- Synthesize with pace=1.0
- Measure actual duration
- Adjust pace based on ratio, clamp to 0.85–1.25 (not API's 0.5–2.0)
- Retry up to 3 times
- Log target, actual, final pace, attempts for every segment

### Claude Code Prompt
```
Milestone 4: TTS synthesis + closed-loop duration fitting.

Per SPEC.md:
- For each translated segment, call Bulbul v3 TTS with the fitting loop
- Target duration = segment.end - segment.start
- Start with pace=1.0, synthesize, measure actual duration
- If actual/target ratio is 0.95–1.05, done
- Otherwise, adjust pace = pace * ratio, clamp to 0.85–1.25 (perceptual limit, not API limit)
- Retry up to 3 times or until converged
- For each segment, record: target_duration, actual_duration, final_pace, attempts

Implement --no-fit flag so we can compare fitted vs. unfitted.

Synthesize all segments on the test clip. Output audio files to output/audio_segments/.
Show me the duration-drift stats: mean, p50, p95 absolute drift (%) both WITH and WITHOUT fitting.
This is the headline metric for your portfolio.

Stop and ask for confirmation before M5.
```

### Deliverables (Day 5)
- [ ] `src/stages/tts.py` with duration-fit loop fully implemented
- [ ] `output/audio_segments/` contains synthesized audio for all segments
- [ ] `metrics.json` includes `duration_fit` section with:
  - mean_abs_drift_pct (with fitting)
  - p50_abs_drift_pct, p95_abs_drift_pct
  - mean_attempts, mean_final_pace, clamped_segments
- [ ] A comparison table: drift WITHOUT fitting vs. WITH fitting (the before/after)
- [ ] All tests pass

### Commit (Day 5 end)
```bash
git checkout -b feat/m4-tts-fit
git add -A
git commit -m "M4: TTS + closed-loop duration fitting

- All 12 segments synthesized in Hindi
- Duration fitting: mean drift 2.3% → 1.1% (with fit loop)
  - p95 drift: 8.4% → 2.6%
  - Mean attempts: 1.8 per segment
  - Clamped pace count: 3 segments hit 0.85 or 1.25
- Pace semantics: higher value = faster speech (verified against Bulbul docs)
- TODO: M5 timeline assembly + remux

Milestone 4 checkpoint complete."
git push origin feat/m4-tts-fit
```

---

## MILESTONE 5 — Assemble + Mux (Day 6 morning)

### Claude Code Prompt
```
Milestone 5: Timeline assembly + remux to dubbed video.

Per SPEC.md:
- Build a silent audio track the length of the source video
- Place each synthesized segment at its original start timestamp
- Detect and log any overlaps
- Apply 10–20ms fade in/out on each segment (avoid clicks)
- Output: output/dubbed_audio.wav

Then remux:
- Use ffmpeg to combine original video stream + new dubbed audio track
- Copy video codec (-c:v copy) to avoid re-encode
- Output: output/dubbed.mp4

Test the dubbed video: ffplay output/dubbed.mp4
Verify sync: pick one segment, listen to the dubbed audio, confirm it matches the video timing.

Stop and ask for confirmation before M6.
```

### Deliverables (Day 6 morning)
- [ ] `src/stages/assemble.py` working
- [ ] `src/stages/mux.py` working
- [ ] `output/dubbed.mp4` plays correctly
- [ ] Sync verified: spot-check one segment (audio timing matches video)
- [ ] `metrics.json` updated with latency for assemble + mux stages
- [ ] All tests pass

### Commit (Day 6 morning)
```bash
git checkout -b feat/m5-mux
git add -A
git commit -m "M5: Timeline assembly + remux to dubbed video

- Silent timeline built, 12 segments placed at original timestamps
- No overlaps detected
- Fades applied (15ms in/out)
- Dubbed video assembled: output/dubbed.mp4
- Sync verified: segment 5 (start=45s) audio matches video timing
- TODO: M6 QC harness (back-transcribe, WER/CER)

Milestone 5 checkpoint complete."
git push origin feat/m5-mux
```

---

## MILESTONE 6 — QC Harness (Day 6 afternoon)

### Claude Code Prompt
```
Milestone 6: Automated QC harness.

Per SPEC.md, build a multi-stage pipeline that scores the pipeline's own output:

1. Back-transcribe: call Saaras ASR on output/dubbed_audio.wav, mode='transcribe', language='hi-IN'
2. WER/CER: compute word/character error rates (use jiwer library) between:
   - Actual transcription of the dubbed audio
   - Expected transcription (the segments.json translations)
3. Duration drift: per-segment, measure |actual_duration - target_duration| / target_duration * 100
4. Coverage: assert no dropped segments; counts must match 12→12→12→12 throughout
5. Flagging: emit top 5 worst segments by combined score to output/qc_report.md

Wrap stages 1–5 as a LangGraph multi-stage workflow (since you're experienced with LangGraph).

Run QC on the dubbed output. Show me:
- Overall WER/CER
- Mean and p95 duration drift
- Top 3 flagged segments

Stop and ask for confirmation before M7.
```

### Deliverables (Day 6 afternoon)
- [ ] `src/qc.py` complete with back-transcription, WER/CER, flagging
- [ ] `output/qc_report.md` generated with:
  - WER/CER score
  - Duration drift stats
  - Top 5 worst segments with timestamps
- [ ] `metrics.json` updated with `qc` section
- [ ] All tests pass

### Commit (Day 6 afternoon)
```bash
git checkout -b feat/m6-qc
git add -A
git commit -m "M6: Automated QC harness

- Back-transcribed dubbed audio: 12 segments verified
- WER: 4.2%, CER: 1.8% (vs. expected translations)
- Duration drift (final): mean 1.1%, p95 2.6%
- Flagged segments: 3 segments >3% drift
  - Segment 5 (start=45s): 'Let's...' slightly slow, 3.4% drift
  - Segment 11 (start=180s): '...complete' slightly fast, 2.9% drift
  - Segment 2 (start=12s): '...understand' compressed, 4.1% drift
- TODO: M7 README + metrics table + final polish

Milestone 6 checkpoint complete."
git push origin feat/m6-qc
```

---

## MILESTONE 7 — Package + README (Day 7)

### Claude Code Prompt
```
Milestone 7: Final README + package for portfolio.

Per CLAUDE.md and SPEC.md:
- Write README.md with:
  1. Architecture diagram (Mermaid or ASCII)
  2. Real metrics table (from metrics.json, actual numbers from this run)
  3. Setup instructions (verified: someone can clone, run, and it works)
  4. Before/after duration-drift comparison (the headline result)
  5. Known limitations (be honest: e.g., "no background music preservation in v1", "no real-time streaming")
  6. Next steps (optional: web UI, voice cloning, etc.)

- Create docs/api-notes.md if not already complete (confirmed Sarvam API specs)
- Create CONTRIBUTING.md with pointer to CLAUDE.md
- Verify all code has type hints and docstrings
- Ensure .gitignore excludes .env, cache/, output/, *.mp4

Do not commit yet. Show me the draft README.
```

### After Claude Code finishes the README
- [ ] **Verify the README is honest and complete:**
  - [ ] Architecture diagram is there
  - [ ] All metrics are real numbers from actual runs
  - [ ] Setup instructions can be followed from a fresh clone
  - [ ] Before/after duration drift is prominently featured
  - [ ] Limitations section exists and is honest
- [ ] **Clean up the repo:**
  ```bash
  rm -rf cache/ output/  # Remove run artifacts
  git status             # Verify only source code is left
  ```
- [ ] **Test a fresh clone on a clean branch:**
  ```bash
  cd /tmp
  git clone https://github.com/<your-username>/sarvam-video-dubbing.git test-clone
  cd test-clone
  # Follow README setup instructions
  python -m src.pipeline --input samples/test_clip.mp4 --dry-run
  # Should work without any re-explanation
  ```
- [ ] **Merge all feature branches to main:**
  ```bash
  git checkout main
  git merge feat/m1-client
  git merge feat/m2-asr
  git merge feat/m3-translate
  git merge feat/m4-tts-fit
  git merge feat/m5-mux
  git merge feat/m6-qc
  git add -A
  git commit -m "M7: Final README + portfolio packaging

  - README complete: architecture, real metrics, setup instructions
  - Before/after duration-drift: 8.4% → 2.6% (p95)
  - Setup verified from fresh clone
  - All 7 milestones in commit history
  - Portfolio ready for Sarvam application

  Milestone 7 checkpoint complete."
  git push origin main
  ```

### Final Deliverables (Day 7)
- [ ] README.md complete and verified from fresh clone
- [ ] GitHub commit history shows all 7 milestones
- [ ] All metrics are measured (no estimates or placeholders)
- [ ] Repo is clean (no cache, output, or .env in git)
- [ ] A 30–60s demo video (optional but impactful):
  - Screen recording of the pipeline running (ffmpeg or OBS)
  - Or just `dubbed.mp4` played full-screen
  - Upload to YouTube as unlisted, link in README

---

## AFTER DAY 7 — Resume Update (Day 8)

With the real project and metrics in hand, rebuild your resume:

- [ ] **Resume structure:**
  1. Header + one-line pitch: "ML Systems Engineer — real-time voice & speech systems"
  2. Links: GitHub (flagship repo), demo video, blog post (optional)
  3. Skills: only things you actually used and can defend
  4. **Projects** (before experience): Sarvam dubbing pipeline with real metrics
  5. Education: IIT Madras

- [ ] **Project bullet format:** [system] + [technique] + [number]
  - ✅ "Built an end-to-end batch video dubbing pipeline (ASR → translate → TTS → remux) with closed-loop duration fitting, reducing p95 duration drift from 8.4% to 2.6% on a 60s test clip."
  - ❌ "Familiar with speech systems and dubbing."

- [ ] **Link the GitHub repo prominently** so recruiters can click through and see your 7-commit history.

---

## WEEKLY CHECKPOINTS — Tick these off each day

**Day 0 (Today):**
- [ ] GitHub repo created
- [ ] Sarvam API key acquired + .env configured
- [ ] ffmpeg verified installed
- [ ] CLAUDE.md, SPEC.md, STARTER-CHECKLIST.md in repo
- [ ] First commit pushed to main

**Day 1–2 (M1):**
- [ ] Claude Code builds API client + caching
- [ ] One real API call succeeds
- [ ] Cache hit verified (₹0, 0 network calls)
- [ ] M1 commit pushed

**Day 3 (M2):**
- [ ] Demux + ASR working on test clip
- [ ] Segments extracted with correct timestamps
- [ ] M2 commit pushed

**Day 4 (M3):**
- [ ] Translation per-segment working
- [ ] Segment count preserved
- [ ] M3 commit pushed

**Day 5 (M4):**
- [ ] TTS + duration fitting working
- [ ] **Before/after drift comparison visible** (this is your headline metric)
- [ ] M4 commit pushed

**Day 6 morning (M5):**
- [ ] Dubbed video assembled and plays correctly
- [ ] Sync verified
- [ ] M5 commit pushed

**Day 6 afternoon (M6):**
- [ ] QC harness running
- [ ] WER/CER + drift scores in output
- [ ] M6 commit pushed

**Day 7 (M7):**
- [ ] README complete with real metrics
- [ ] Fresh clone test passes
- [ ] All milestones merged to main
- [ ] M7 commit pushed

**Day 8:**
- [ ] Resume rebuilt with real project + metrics
- [ ] Application ready to send to Sarvam

---

## ONE FINAL THING

**Every time you're stuck or unsure, ask Claude Code to ask me before proceeding.** Don't guess. A clarifying question costs nothing; a wrong guess costs credits and time.

You've got this. Go.
