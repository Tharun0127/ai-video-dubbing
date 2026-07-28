"""Shared pytest fixtures: an isolated cache, a valid Config, and captured API fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.cache import DiskCache
from src.config import Config
from src.metrics import MetricsCollector

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """A throwaway cache directory, isolated per test."""
    return tmp_path / "cache"


@pytest.fixture
def config(cache_dir: Path, tmp_path: Path) -> Config:
    """A valid Config with a fake key; no test is allowed to touch the real API."""
    return Config(
        api_key="sk_test_not_a_real_key",
        base_url="https://api.sarvam.ai",
        input_path=tmp_path / "input.wav",
        output_path=tmp_path / "out" / "dubbed.mp4",
        cache_dir=cache_dir,
        output_dir=tmp_path / "out",
    )


@pytest.fixture
def cache(cache_dir: Path) -> DiskCache:
    """A DiskCache rooted in the per-test temp directory."""
    return DiskCache(cache_dir)


@pytest.fixture
def metrics() -> MetricsCollector:
    """A fresh metrics collector."""
    return MetricsCollector()


@pytest.fixture
def asr_response() -> dict[str, Any]:
    """The real Saaras v3 response captured on 2026-07-28 (also serves as API documentation)."""
    return json.loads((FIXTURE_DIR / "asr_saaras_v3_transcribe.json").read_text(encoding="utf-8"))


@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    """A 0.5-second 16 kHz mono silent WAV, written with the stdlib (no ffmpeg needed)."""
    import wave

    path = tmp_path / "tiny.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 8000)
    return path
