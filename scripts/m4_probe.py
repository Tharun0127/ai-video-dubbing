"""
M4 probe: confirm the /text-to-speech wire shape and the SIGN of the `pace` parameter.

the brief is explicit that the duration-fit loop must not rely on an assumed pace direction.
The whole loop is `pace *= actual/target`, which only converges if a HIGHER pace produces
SHORTER audio. If the sense were reversed, that update would drive every segment away
from its target instead of towards it, and the failure would look like "the model is bad
at timing" rather than "the sign is wrong".

So this script measures it rather than trusting the docs:

  1. One raw call, whose response keys and audio header are printed verbatim.
  2. The same Hindi sentence synthesised across the full documented pace range, with each
     duration measured from the returned WAV's own frame count. Monotonically decreasing
     duration confirms higher = faster.
  3. Bulbul's leading/trailing padding measured at every pace, to test the assumption
     that the padding is a fixed offset rather than something that scales with pace --
     which is what decides whether the fit loop must trim before measuring.

Everything goes through SarvamClient, so a second run is served from cache and costs Rs 0.

Usage:
    python -m scripts.m4_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.audio import read_wav_bytes, trim_wav_silence
from src.cache import DiskCache
from src.config import load_config, setup_logging
from src.metrics import MetricsCollector
from src.sarvam_client import SarvamClient

logger = logging.getLogger("m4_probe")

#: Full documented range plus the perceptual clamp bounds the pipeline actually uses.
PACES: tuple[float, ...] = (0.5, 0.85, 1.0, 1.25, 1.5, 2.0)

#: A real translated segment from M3, so the measurement is on the material we dub.
PROBE_TEXT = "आपने ये कैसे किया? मैं हैरान हूँ।"


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write the observations to disk."""
    parser = argparse.ArgumentParser(prog="python -m scripts.m4_probe")
    parser.add_argument("--out", type=Path, default=Path("output/m4_pace_probe.json"))
    parser.add_argument("--text", default=PROBE_TEXT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    config = load_config(input_path=None, source_lang="en-IN", target_lang="hi-IN")
    metrics = MetricsCollector()

    observations: dict[str, Any] = {
        "endpoint": "POST /text-to-speech",
        "model": config.tts_model,
        "speaker": config.tts_speaker,
        "text": args.text,
        "text_chars": len(args.text),
    }

    with SarvamClient(config, cache=DiskCache(config.cache_dir), metrics=metrics) as client:
        rows: list[dict[str, Any]] = []
        for index, pace in enumerate(PACES):
            payload = client.text_to_speech(
                args.text, target_language_code="hi-IN", pace=pace, stage="tts_probe",
            )
            audio = client.audio_bytes(payload)
            info, _frames = read_wav_bytes(audio)
            trimmed = trim_wav_silence(audio, threshold_db=config.trim_threshold_db)

            if index == 0:
                # Print the raw response shape exactly once, with the audio elided so the
                # console stays readable.
                redacted = {
                    k: (f"<base64, {len(v[0])} chars>" if k == "audios" else v)
                    for k, v in payload.items()
                }
                print("\n--- raw /text-to-speech response (audio elided) ---")
                print(json.dumps(redacted, indent=2, ensure_ascii=False))
                print("--- response keys:", sorted(payload.keys()), "---")
                print(f"--- audio header: {audio[:4]!r} ... {audio[8:12]!r}, "
                      f"{info.frame_rate} Hz, {info.channels} ch, "
                      f"{info.sample_width * 8}-bit ---\n")
                observations["response_keys"] = sorted(payload.keys())
                observations["wav"] = {
                    "frame_rate": info.frame_rate,
                    "channels": info.channels,
                    "sample_width_bytes": info.sample_width,
                    "riff_header": audio[:4].decode("ascii", "replace"),
                    "format": audio[8:12].decode("ascii", "replace"),
                }

            rows.append({
                "pace": pace,
                "untrimmed_duration_s": round(info.duration_s, 6),
                "trimmed_duration_s": round(trimmed.duration_s, 6),
                "lead_trim_s": round(trimmed.lead_trim_s, 6),
                "trail_trim_s": round(trimmed.trail_trim_s, 6),
                "padding_total_s": round(trimmed.lead_trim_s + trimmed.trail_trim_s, 6),
                "peak_dbfs": round(trimmed.peak_dbfs, 2),
                "request_id": payload.get("request_id"),
            })

    observations["pace_sweep"] = rows
    observations["conclusion"] = _conclude(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(observations, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics.log_summary()
    logger.info("wrote %s", args.out)

    _print_table(rows, observations["conclusion"])
    return 0


def _conclude(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide the pace direction and whether padding scales, from the measured sweep."""
    trimmed = [r["trimmed_duration_s"] for r in rows]
    padding = [r["padding_total_s"] for r in rows]

    decreasing = all(b < a for a, b in zip(trimmed, trimmed[1:]))
    increasing = all(b > a for a, b in zip(trimmed, trimmed[1:]))

    if decreasing:
        direction = "higher pace = FASTER (shorter audio) -- `pace *= ratio` is correct"
    elif increasing:
        direction = "higher pace = SLOWER (longer audio) -- the update rule must be INVERTED"
    else:
        direction = "NOT MONOTONIC -- do not build the fit loop on this parameter"

    baseline = next((r for r in rows if r["pace"] == 1.0), None)
    span = max(padding) - min(padding) if padding else 0.0

    # Padding matters to the loop only in proportion to the convergence band: the band is
    # +/-5%, so a pad worth more than 5% of the clip can flip a segment's verdict on its
    # own. Reporting it that way is more useful than "does it scale with pace", which a
    # six-point sweep cannot answer.
    padding_pct = [
        100.0 * r["padding_total_s"] / r["trimmed_duration_s"]
        for r in rows if r["trimmed_duration_s"] > 0
    ]

    # If duration were exactly inversely proportional to pace, pace x duration would be
    # constant. It is not, so measure the local elasticity instead: d(log duration) /
    # d(log pace). Exactly -1 would make `pace *= ratio` land on target in one step;
    # steeper than -1 means that update systematically overshoots.
    elasticities: list[dict[str, float]] = []
    for left, right in zip(rows, rows[1:]):
        import math

        if left["trimmed_duration_s"] > 0 and right["trimmed_duration_s"] > 0:
            elasticities.append({
                "from_pace": left["pace"],
                "to_pace": right["pace"],
                "elasticity": round(
                    math.log(right["trimmed_duration_s"] / left["trimmed_duration_s"])
                    / math.log(right["pace"] / left["pace"]), 4,
                ),
            })

    return {
        "pace_direction": direction,
        "monotonic_decreasing": decreasing,
        "padding_min_s": round(min(padding), 6) if padding else None,
        "padding_max_s": round(max(padding), 6) if padding else None,
        "padding_span_s": round(span, 6),
        "padding_max_pct_of_clip": round(max(padding_pct), 3) if padding_pct else None,
        "padding_can_flip_the_5pct_band": bool(padding_pct and max(padding_pct) > 5.0),
        "baseline_trimmed_duration_s": baseline["trimmed_duration_s"] if baseline else None,
        "pace_times_duration": [
            round(r["pace"] * r["trimmed_duration_s"], 4) for r in rows
        ],
        "duration_vs_pace_elasticity": elasticities,
        "note": (
            "Measured on one Hindi sentence. The sign is a property of the API and "
            "generalises; the elasticity magnitudes are one sample."
        ),
    }


def _print_table(rows: list[dict[str, Any]], conclusion: dict[str, Any]) -> None:
    """Print the pace sweep and what it implies for the fit loop."""
    print("=== PACE SWEEP (durations measured from the returned WAV frames) ===")
    print(f"  {'pace':>6} {'untrimmed':>11} {'trimmed':>9} {'lead':>8} {'trail':>8} "
          f"{'padding':>9} {'pace*dur':>9}")
    for row in rows:
        print(f"  {row['pace']:>6.2f} {row['untrimmed_duration_s']:>11.3f} "
              f"{row['trimmed_duration_s']:>9.3f} {row['lead_trim_s']:>8.3f} "
              f"{row['trail_trim_s']:>8.3f} {row['padding_total_s']:>9.3f} "
              f"{row['pace'] * row['trimmed_duration_s']:>9.3f}")
    print(f"\n  CONCLUSION: {conclusion['pace_direction']}")
    print(f"  padding: {conclusion['padding_min_s']:.3f}s - {conclusion['padding_max_s']:.3f}s, "
          f"worst case {conclusion['padding_max_pct_of_clip']:.1f}% of the clip; "
          f"can flip the +/-5% band on its own: "
          f"{conclusion['padding_can_flip_the_5pct_band']}")
    print("\n  duration-vs-pace elasticity (-1.0 = perfectly proportional, so one "
          "`pace *= ratio` step would land exactly):")
    for row in conclusion["duration_vs_pace_elasticity"]:
        print(f"    {row['from_pace']:.2f} -> {row['to_pace']:.2f}: {row['elasticity']:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
