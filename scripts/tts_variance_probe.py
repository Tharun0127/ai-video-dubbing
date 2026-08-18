"""
Probe: is Bulbul v3 duration-deterministic? Same text, same pace, N real calls.

Why this exists: the M7 cold-cache run and the warm-cache run disagreed about per-segment
drift, on identical text with identical parameters. Either the synthesiser is not
deterministic or something upstream changed. The translations were byte-identical, so this
script isolates the remaining variable by calling /text-to-speech repeatedly with one fixed
input and measuring the duration of every response.

The result matters to the M4 fit loop directly: if run-to-run duration variance is larger
than the +/-5% convergence band, then a pace computed from one measurement cannot be
trusted to hold on the next call, and closing the loop per run (rather than caching a pace
per phrase) is the only correct design.

Each repeat uses its own cache directory, so every call really hits the network -- a shared
cache would return the first response N times and measure nothing.

Usage:
    python -m scripts.tts_variance_probe --repeats 3
    python -m scripts.tts_variance_probe --repeats 3 --text "..." --pace 1.0

Output: output/tts_variance_probe.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

from src.audio import trim_wav_silence, wav_bytes_duration_s
from src.cache import DiskCache
from src.config import ConfigError, load_config, setup_logging
from src.metrics import MetricsCollector
from src.sarvam_client import SarvamClient

logger = logging.getLogger("tts_variance_probe")

#: Segment 2 of samples/test_clip.mp4 -- the segment whose two runs disagreed the most.
DEFAULT_TEXT = "Jay, आपका card सबसे ऊपर वाला card है। अरे नहीं। वाह, ये तो कमाल है। सच में? हाँ जी।"


def main(argv: list[str] | None = None) -> int:
    """Call /text-to-speech N times with one fixed input and report the duration spread."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesise repeatedly.")
    parser.add_argument("--pace", type=float, default=1.0, help="Fixed pace for every call.")
    parser.add_argument("--repeats", type=int, default=3, help="How many real calls to make.")
    parser.add_argument("--target-lang", default="hi-IN")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    if args.repeats < 2:
        logger.error("--repeats must be >= 2; one call cannot show variance")
        return 1

    try:
        config = load_config(target_lang=args.target_lang)
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 1

    metrics = MetricsCollector()
    observations: list[dict[str, object]] = []

    logger.info("synthesising the same %d-char text %d time(s) at pace=%.2f",
                len(args.text), args.repeats, args.pace)

    for repeat in range(args.repeats):
        # A fresh cache per repeat is the whole point: a shared one would serve call 1's
        # audio to every later call and report perfect determinism by construction.
        scratch = Path(tempfile.mkdtemp(prefix=f"tts_variance_{repeat}_"))
        try:
            with SarvamClient(config, cache=DiskCache(scratch), metrics=metrics) as client:
                payload = client.text_to_speech(
                    args.text,
                    target_language_code=config.target_lang,
                    model=config.tts_model,
                    speaker=config.tts_speaker,
                    pace=args.pace,
                    speech_sample_rate=config.tts_sample_rate,
                )
                audio = client.audio_bytes(payload)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        raw_s = wav_bytes_duration_s(audio)
        trimmed = trim_wav_silence(audio, threshold_db=config.trim_threshold_db)
        observations.append({
            "repeat": repeat + 1,
            "request_id": payload.get("request_id"),
            "raw_duration_s": round(raw_s, 6),
            "trimmed_duration_s": round(trimmed.duration_s, 6),
            "lead_trim_s": round(trimmed.lead_trim_s, 6),
            "trail_trim_s": round(trimmed.trail_trim_s, 6),
            "audio_bytes": len(audio),
        })
        logger.info("  call %d: %.3fs raw, %.3fs trimmed (%d bytes)",
                    repeat + 1, raw_s, trimmed.duration_s, len(audio))

    trimmed_durations = [float(o["trimmed_duration_s"]) for o in observations]
    mean = statistics.fmean(trimmed_durations)
    spread = max(trimmed_durations) - min(trimmed_durations)

    summary = {
        "text": args.text,
        "text_chars": len(args.text),
        "pace": args.pace,
        "model": config.tts_model,
        "speaker": config.tts_speaker,
        "target_language_code": config.target_lang,
        "speech_sample_rate": config.tts_sample_rate,
        "repeats": args.repeats,
        "observations": observations,
        "trimmed_duration_s": {
            "min": round(min(trimmed_durations), 6),
            "max": round(max(trimmed_durations), 6),
            "mean": round(mean, 6),
            "stdev": round(statistics.stdev(trimmed_durations), 6) if len(trimmed_durations) > 1 else 0.0,
            "spread_s": round(spread, 6),
            "spread_pct_of_mean": round(spread / mean * 100.0, 4) if mean > 0 else None,
        },
        "identical_bytes": len({o["audio_bytes"] for o in observations}) == 1,
        "cost_inr": round(metrics.total_cost_inr, 6),
    }

    out_path = Path("output/tts_variance_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = summary["trimmed_duration_s"]
    logger.info("=" * 68)
    logger.info("trimmed duration over %d identical call(s): min %.3fs, max %.3fs, "
                "mean %.3fs, spread %.3fs (%.2f%% of mean)",
                args.repeats, stats["min"], stats["max"], stats["mean"],
                stats["spread_s"], stats["spread_pct_of_mean"])
    logger.info("byte-identical responses: %s", summary["identical_bytes"])
    logger.info("cost: Rs %.4f | wrote %s", metrics.total_cost_inr, out_path)
    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
