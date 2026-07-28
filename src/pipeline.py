"""
Pipeline orchestrator and CLI entrypoint.

Parses the command line, builds the Config / DiskCache / MetricsCollector /
SarvamClient stack, and runs the requested stages in order, writing
output/metrics.json at the end of every run -- including failed ones, so a crash
still leaves the measurements and the last successful checkpoint on disk.

Stages are registered here and implemented in src/stages/. A stage that has not
been built yet reports the milestone that will deliver it rather than pretending
to succeed.

Input:  CLI arguments.
Output: output/dubbed.mp4, output/metrics.json, output/qc_report.md (as stages land).

Usage:
    python -m src.pipeline --input samples/clip.mp4 --source-lang en-IN --target-lang hi-IN
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from .cache import CacheMissError, DiskCache
from .config import STAGE_ORDER, Config, ConfigError, load_config, setup_logging
from .metrics import MetricsCollector
from .sarvam_client import SarvamAPIError, SarvamClient

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_IMPLEMENTED = 2

#: Milestone that delivers each stage; used for honest "not built yet" messages.
STAGE_MILESTONE: dict[str, str] = {
    "demux": "M2", "asr": "M2", "translate": "M3", "tts": "M4",
    "assemble": "M5", "mux": "M5", "qc": "M6",
}


class StageNotImplementedError(RuntimeError):
    """Raised when a requested stage has not been built yet."""


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser with every flag defined in SPEC.md."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Batch video dubbing pipeline built on Sarvam AI (English -> Hindi by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to the input video or audio file.")
    parser.add_argument("--output", default=Path("output/dubbed.mp4"), type=Path,
                        help="Path to write the dubbed video to.")
    parser.add_argument("--source-lang", default="en-IN",
                        help="BCP-47 source language code (e.g. en-IN).")
    parser.add_argument("--target-lang", default="hi-IN",
                        help="BCP-47 target language code (e.g. hi-IN).")
    parser.add_argument("--no-fit", action="store_true",
                        help="Disable closed-loop TTS duration fitting (for before/after comparison).")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Process at most N segments, to save credits during development.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Serve every API result from cache; fail loudly on any cache miss.")
    parser.add_argument("--stage", choices=STAGE_ORDER, default=None,
                        help="Run a single stage and stop.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory holding the on-disk API cache.")
    parser.add_argument("--metrics-path", type=Path, default=None,
                        help="Where to write metrics.json (defaults to <output dir>/metrics.json).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging, including API request/response bodies.")
    return parser


def stages_to_run(config: Config) -> tuple[str, ...]:
    """Return the ordered stage list for this run: one stage if --stage was given, else all."""
    return (config.stage,) if config.stage else STAGE_ORDER


def run_stage(name: str, config: Config, client: SarvamClient, metrics: MetricsCollector) -> None:
    """Dispatch one named stage; unimplemented stages name the milestone that delivers them."""
    milestone = STAGE_MILESTONE.get(name, "a later milestone")
    raise StageNotImplementedError(
        f"stage {name!r} is not implemented yet -- it arrives in {milestone}. "
        f"Milestone 1 delivers config, cache, metrics, the Sarvam client, and this CLI."
    )


def run(config: Config, metrics: MetricsCollector | None = None) -> int:
    """Execute the configured stages, always writing metrics.json; returns a process exit code."""
    metrics = metrics or MetricsCollector()
    cache = DiskCache(config.cache_dir)
    metrics.duration_fit_enabled = config.enable_duration_fit

    input_duration: float | None = None
    if config.input_path is not None and config.input_path.exists():
        try:
            from .audio import probe_duration_s

            input_duration = probe_duration_s(config.input_path)
        except Exception as exc:  # probing is best-effort; never block the run on it
            logger.warning("could not probe input duration for %s: %s", config.input_path, exc)

    metrics.set_input_info(
        path=str(config.input_path) if config.input_path else "",
        duration_s=input_duration,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )

    metrics_path = config.output_dir / "metrics.json"
    exit_code = EXIT_OK
    completed: list[str] = []

    with SarvamClient(config, cache=cache, metrics=metrics) as client:
        try:
            for stage in stages_to_run(config):
                with metrics.time_stage(stage):
                    run_stage(stage, config, client, metrics)
                completed.append(stage)
        except StageNotImplementedError as exc:
            logger.error("%s", exc)
            exit_code = EXIT_NOT_IMPLEMENTED
        except CacheMissError as exc:
            logger.error("dry-run aborted: %s", exc)
            exit_code = EXIT_ERROR
        except SarvamAPIError as exc:
            logger.exception("Sarvam API call failed during stage dispatch: %s", exc)
            exit_code = EXIT_ERROR
        except Exception:
            logger.exception("unhandled failure; last completed stage: %s",
                             completed[-1] if completed else "none")
            exit_code = EXIT_ERROR

    metrics.log_summary()
    metrics.write(metrics_path)
    if exit_code != EXIT_OK and completed:
        logger.error("Resume from the last successful checkpoint: --stage %s onwards",
                     completed[-1])
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint: parse arguments, build config, run the pipeline, return an exit code."""
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)

    try:
        config = load_config(
            input_path=args.input,
            output_path=args.output,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            dry_run=args.dry_run,
            max_segments=args.max_segments,
            enable_duration_fit=not args.no_fit,
            stage=args.stage,
            verbose=args.verbose,
            cache_dir=args.cache_dir,
        )
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return EXIT_ERROR

    if not config.input_path or not config.input_path.exists():
        logger.error("input file does not exist: %s", config.input_path)
        return EXIT_ERROR

    logger.info("Starting run: %r", config)
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
