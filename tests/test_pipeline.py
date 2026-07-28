"""Tests for the CLI surface and the run loop.

The CLI contract matters as much as the stages: --dry-run and --max-segments exist
so later milestones can be developed without re-spending credits on earlier ones,
so they are tested from milestone 1 rather than bolted on later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import STAGE_ORDER, Config
from src.metrics import MetricsCollector
from src.pipeline import (
    EXIT_ERROR,
    EXIT_NOT_IMPLEMENTED,
    StageNotImplementedError,
    build_arg_parser,
    main,
    run,
    run_stage,
    stages_to_run,
)


# --- argument parsing -------------------------------------------------------------------

def test_parser_accepts_every_spec_flag() -> None:
    """Every flag SPEC.md defines is wired, including the credit-saving ones."""
    args = build_arg_parser().parse_args([
        "--input", "clip.mp4", "--source-lang", "en-IN", "--target-lang", "hi-IN",
        "--output", "out/dubbed.mp4", "--no-fit", "--max-segments", "3",
        "--dry-run", "--stage", "asr", "--verbose",
    ])
    assert args.input == Path("clip.mp4")
    assert args.source_lang == "en-IN"
    assert args.target_lang == "hi-IN"
    assert args.no_fit is True
    assert args.max_segments == 3
    assert args.dry_run is True
    assert args.stage == "asr"
    assert args.verbose is True


def test_parser_defaults_match_the_project_direction() -> None:
    """Defaults are en-IN -> hi-IN with duration fitting on."""
    args = build_arg_parser().parse_args(["--input", "clip.mp4"])
    assert (args.source_lang, args.target_lang) == ("en-IN", "hi-IN")
    assert args.no_fit is False
    assert args.dry_run is False
    assert args.max_segments is None


def test_input_is_required() -> None:
    """Running with no input is a usage error, not a confusing crash later."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args([])


def test_stage_flag_rejects_unknown_stages() -> None:
    """argparse rejects a stage that is not in the pipeline."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--input", "c.mp4", "--stage", "denoise"])


# --- stage selection ---------------------------------------------------------------------

def test_stages_to_run_is_the_full_pipeline_by_default(config: Config) -> None:
    """With no --stage, every stage runs in SPEC order."""
    assert stages_to_run(config) == STAGE_ORDER


def test_stages_to_run_honours_single_stage(config: Config) -> None:
    """--stage asr runs exactly one stage."""
    single = Config(**{**config.__dict__, "stage": "asr"})
    assert stages_to_run(single) == ("asr",)


def test_unimplemented_stage_names_its_milestone(config: Config) -> None:
    """A stage that has not been built says which milestone delivers it (demux/asr landed in M2)."""
    with pytest.raises(StageNotImplementedError, match="M3"):
        run_stage("translate", config, client=None, metrics=MetricsCollector())  # type: ignore[arg-type]


# --- run loop ----------------------------------------------------------------------------

def test_run_writes_metrics_even_when_a_stage_fails(config: Config, tiny_wav: Path) -> None:
    """A crash still leaves measurements on disk; that is how a run gets debugged."""
    cfg = Config(**{**config.__dict__, "input_path": tiny_wav, "stage": "translate"})
    metrics = MetricsCollector()

    assert run(cfg, metrics) == EXIT_NOT_IMPLEMENTED
    metrics_path = cfg.output_dir / "metrics.json"
    assert metrics_path.exists()


def test_run_records_the_measured_input_duration(config: Config, tiny_wav: Path) -> None:
    """The input duration in metrics.json is probed, never assumed."""
    cfg = Config(**{**config.__dict__, "input_path": tiny_wav, "stage": "translate"})
    metrics = MetricsCollector()
    run(cfg, metrics)
    assert metrics.input_info["duration_s"] == pytest.approx(0.5, abs=0.01)


def test_main_reports_a_missing_input_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonexistent input exits with an error rather than a traceback."""
    monkeypatch.setenv("SARVAM_API_KEY", "sk_test_key")
    monkeypatch.setenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
    assert main(["--input", str(tmp_path / "missing.mp4")]) == EXIT_ERROR


def test_main_reports_a_configuration_error(tiny_wav: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad language code exits cleanly with EXIT_ERROR."""
    monkeypatch.setenv("SARVAM_API_KEY", "sk_test_key")
    monkeypatch.setenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
    assert main(["--input", str(tiny_wav), "--source-lang", "en-US"]) == EXIT_ERROR
