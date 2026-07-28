"""
Milestone 2 acceptance harness: prove the chunking, stitching, and caching actually work.

Every check below reads measured artefacts from the last real run (output/asr_report.json,
output/segments.json, output/chunks/*.wav) -- nothing here is asserted from expectation.

  CHECK 1  chunk plan      -- every chunk under the 30 s REST cap, chunks tile [0, duration]
  CHECK 2  audio integrity -- the chunk WAVs concatenate to a byte-identical copy of the
                             demuxed source, which is the strongest possible statement that
                             the seam dropped and duplicated exactly zero samples
  CHECK 3  seams           -- each boundary is contiguous and sits inside a detected pause
  CHECK 4  timeline        -- segments tile [0, duration]: no gap, no overlap, ordered
  CHECK 5  post-processing -- no segment left under the minimum or over the maximum
  CHECK 6  cache           -- a repeat run under --dry-run serves every call from disk with
                             zero network traffic, enforced by a socket guard that raises on
                             any outbound connection attempt

Usage:
    python -m scripts.m2_verify --input samples/test_clip.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, setup_logging  # noqa: E402
from src.metrics import MetricsCollector  # noqa: E402
from src.pipeline import run  # noqa: E402
from src.stages.asr import TIME_EPSILON_S  # noqa: E402
from src.test_m1_api import NetworkAccessError, no_network_allowed  # noqa: E402


def read_pcm(path: Path) -> tuple[bytes, int]:
    """Return the raw PCM frames and frame rate of a WAV file."""
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes()), handle.getframerate()


def check_chunk_plan(report: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Assert every chunk fits the REST cap and the chunks tile the audio exactly."""
    chunks = report["plan"]["chunks"]
    cap = report["asr"]["max_chunk_s"]
    duration = report["timeline_covered_s"]
    results = []

    over = [c for c in chunks if c["duration_s"] > cap]
    results.append((
        f"all {len(chunks)} chunk(s) within the {cap:.0f}s REST cap", not over,
        f"longest is {max(c['duration_s'] for c in chunks):.3f}s",
    ))
    results.append((
        "more than one chunk (multi-chunk stitching exercised)", len(chunks) >= 2,
        f"{len(chunks)} chunk(s)",
    ))
    results.append((
        "chunk 0 starts at 0.0s", abs(chunks[0]["start_s"]) <= TIME_EPSILON_S,
        f"{chunks[0]['start_s']:.6f}s",
    ))
    results.append((
        "last chunk reaches the end of the audio",
        abs(chunks[-1]["end_s"] - duration) <= TIME_EPSILON_S,
        f"{chunks[-1]['end_s']:.6f}s vs {duration:.6f}s",
    ))
    gaps = [
        (a["index"], b["index"], b["start_s"] - a["end_s"])
        for a, b in zip(chunks, chunks[1:]) if abs(b["start_s"] - a["end_s"]) > TIME_EPSILON_S
    ]
    results.append(("chunks tile with no gap or overlap", not gaps, f"{len(gaps)} bad joint(s)"))
    return results


def check_audio_integrity(report: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Concatenate the chunk WAVs and compare them byte-for-byte with the demuxed source."""
    source_path = Path(report["audio_path"])
    chunk_paths = [Path(t["chunk_path"]) for t in report["raw_transcripts"]]

    if not source_path.exists() or not all(p.exists() for p in chunk_paths):
        return [("chunk audio reconstructs the source", False, "chunk or source WAV missing")]

    source, rate = read_pcm(source_path)
    joined = b"".join(read_pcm(p)[0] for p in chunk_paths)

    lost_frames = (len(source) - len(joined)) // 2
    return [
        (
            "chunk WAVs concatenate to the exact source audio", joined == source,
            f"sha256 {hashlib.sha256(joined).hexdigest()[:16]} vs "
            f"{hashlib.sha256(source).hexdigest()[:16]}",
        ),
        (
            "zero frames dropped or duplicated at the seam(s)", lost_frames == 0,
            f"{lost_frames} frame(s) ({lost_frames / rate:.6f}s)",
        ),
    ]


def check_seams(report: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Assert every chunk seam is contiguous and, where possible, sits inside a real pause."""
    seams = report["stitching"]["seams"]
    if not seams:
        return [("at least one seam to verify", False, "single-chunk run")]

    results = [(
        "every seam is contiguous (no gap, no overlap)",
        all(s["contiguous"] for s in seams),
        f"max |delta| = {max(abs(s['gap_or_overlap_s']) for s in seams):.9f}s",
    )]
    in_silence = [s for s in seams if s["inside_detected_silence"]]
    results.append((
        "every seam falls inside a detected silence (never mid-word)",
        len(in_silence) == len(seams),
        f"{len(in_silence)}/{len(seams)}; min margin to speech "
        f"{min((s['margin_to_speech_s'] for s in in_silence), default=0):.3f}s",
    ))
    results.append((
        "no boundary was forced at the 30s cap",
        all(s["cut_source"] != "hard" for s in seams),
        f"{sum(1 for s in seams if s['cut_source'] == 'hard')} hard cut(s)",
    ))
    return results


def check_timeline(report: dict[str, Any], segments: list[dict[str, Any]]) -> list[tuple[str, bool, str]]:
    """Assert the final segments tile [0, duration] in order, with no gap and no overlap."""
    duration = report["timeline_covered_s"]
    joints = [
        (a["id"], b["id"], b["start"] - a["end"])
        for a, b in zip(segments, segments[1:])
    ]
    bad = [j for j in joints if abs(j[2]) > TIME_EPSILON_S]
    non_positive = [s for s in segments if s["end"] <= s["start"]]

    return [
        ("segments start at 0.0s", abs(segments[0]["start"]) <= TIME_EPSILON_S,
         f"{segments[0]['start']:.3f}s"),
        ("segments end at the audio duration",
         abs(segments[-1]["end"] - duration) <= 0.01,
         f"{segments[-1]['end']:.3f}s vs {duration:.3f}s"),
        ("every segment has positive duration", not non_positive,
         f"{len(non_positive)} bad segment(s)"),
        ("segments tile with no gap or overlap", not bad, f"{len(bad)} bad joint(s)"),
    ]


def check_post_processing(report: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Assert the merge/split pass left nothing outside the configured segment length bounds."""
    pp = report["post_processing"]
    return [
        (f"no segment under the {pp['min_segment_s']:.1f}s minimum",
         pp["still_under_min"] == 0, f"{pp['still_under_min']} remaining"),
        (f"no segment over the {pp['max_segment_s']:.1f}s maximum",
         pp["still_over_max"] == 0, f"{pp['still_over_max']} remaining"),
        ("post-processing changed the segment count",
         pp["count_before"] != pp["count_after_split"],
         f"{pp['count_before']} -> {pp['count_after_merge']} (merge) -> "
         f"{pp['count_after_split']} (split)"),
    ]


def check_cache(input_path: Path) -> list[tuple[str, bool, str]]:
    """Re-run the ASR stage under --dry-run with sockets blocked; any network call is a failure."""
    config = load_config(input_path=input_path, stage="asr", dry_run=True)
    metrics = MetricsCollector()

    try:
        with no_network_allowed():
            exit_code = run(config, metrics)
    except NetworkAccessError as exc:
        return [("repeat run issues zero network calls", False, str(exc))]

    return [
        ("repeat run under --dry-run succeeds", exit_code == 0, f"exit code {exit_code}"),
        ("repeat run issues zero network calls", metrics.cache_misses == 0,
         f"{metrics.cache_misses} network call(s), {metrics.cache_hits} cache hit(s)"),
        ("repeat run costs Rs 0", metrics.total_cost_inr == 0.0,
         f"Rs {metrics.total_cost_inr:.4f} spent, Rs {metrics.cost_avoided_inr:.4f} avoided"),
    ]


def main(argv: list[str] | None = None) -> int:
    """Execute the six M2 checks against the last run's artefacts and print the results."""
    parser = argparse.ArgumentParser(description="Milestone 2 acceptance harness.")
    parser.add_argument("--input", type=Path, default=Path("samples/test_clip.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    setup_logging(verbose=False)

    report_path = args.output_dir / "asr_report.json"
    segments_path = args.output_dir / "segments.json"
    if not report_path.exists() or not segments_path.exists():
        print(f"ERROR: {report_path} or {segments_path} is missing. Run the pipeline first:\n"
              f"  python -m src.pipeline --input {args.input} --stage demux\n"
              f"  python -m src.pipeline --input {args.input} --stage asr")
        return 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    groups = [
        ("1. CHUNK PLAN", check_chunk_plan(report)),
        ("2. AUDIO INTEGRITY", check_audio_integrity(report)),
        ("3. CHUNK SEAMS", check_seams(report)),
        ("4. SEGMENT TIMELINE", check_timeline(report, segments)),
        ("5. POST-PROCESSING", check_post_processing(report)),
        ("6. CACHE", check_cache(args.input)),
    ]

    failures = 0
    print()
    print("=" * 78)
    print(f"MILESTONE 2 ACCEPTANCE -- {report['audio_path']} "
          f"({report['audio_duration_s']:.3f}s)")
    print("=" * 78)
    for title, checks in groups:
        print(f"\n{title}")
        for label, passed, detail in checks:
            failures += 0 if passed else 1
            print(f"  [{'PASS' if passed else 'FAIL'}]  {label:<58} {detail}")

    print()
    print("=" * 78)
    print(f"RESULT: {'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    print("=" * 78)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
