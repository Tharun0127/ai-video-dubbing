"""Tests for configuration loading, validation, and secret hygiene.

Two of these encode failures actually hit while building M1: a key pasted with
its placeholder angle brackets, and a base URL carrying a /v1 prefix. Both used
to surface as opaque HTTP errors mid-run; both now fail at startup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    ASR_LANGUAGES,
    DEFAULT_BASE_URL,
    Config,
    ConfigError,
    load_config,
)


def make_config(**overrides: object) -> Config:
    """Build a valid Config with overridable fields."""
    defaults: dict[str, object] = {"api_key": "sk_test_key"}
    defaults.update(overrides)
    return Config(**defaults)  # type: ignore[arg-type]


def test_missing_api_key_is_rejected() -> None:
    """An empty key fails at construction with a message naming .env."""
    with pytest.raises(ConfigError, match="SARVAM_API_KEY"):
        make_config(api_key="")


def test_placeholder_wrapped_key_is_rejected() -> None:
    """A key pasted as <sk_...> would produce an opaque 403; catch it at startup instead."""
    with pytest.raises(ConfigError, match="placeholder"):
        make_config(api_key="<sk_realkeyhere>")


def test_quoted_key_is_rejected() -> None:
    """Shell-style quotes around the key are the same failure mode."""
    with pytest.raises(ConfigError, match="placeholder"):
        make_config(api_key='"sk_realkeyhere"')


def test_v1_base_url_is_rejected() -> None:
    """/v1 is the chat-completions prefix only; speech endpoints 404 under it."""
    with pytest.raises(ConfigError, match="/v1"):
        make_config(base_url="https://api.sarvam.ai/v1")


def test_default_base_url_is_accepted() -> None:
    """The documented root base URL passes validation."""
    assert make_config(base_url=DEFAULT_BASE_URL).base_url == DEFAULT_BASE_URL


@pytest.mark.parametrize("lang", ["en-US", "hindi", "hi_IN", ""])
def test_invalid_language_codes_are_rejected(lang: str) -> None:
    """Language codes are validated against the set Sarvam documents."""
    with pytest.raises(ConfigError):
        make_config(source_lang=lang)


@pytest.mark.parametrize("lang", ["en-IN", "hi-IN", "ta-IN", "unknown"])
def test_documented_language_codes_are_accepted(lang: str) -> None:
    """Every code in the confirmed set is usable."""
    assert make_config(source_lang=lang).source_lang == lang
    assert lang in ASR_LANGUAGES


def test_unknown_stage_is_rejected() -> None:
    """--stage only accepts stages that exist in the pipeline."""
    with pytest.raises(ConfigError, match="stage"):
        make_config(stage="denoise")


def test_max_segments_must_be_positive() -> None:
    """--max-segments 0 would silently process nothing."""
    with pytest.raises(ConfigError, match="max-segments"):
        make_config(max_segments=0)


def test_repr_never_leaks_the_api_key() -> None:
    """Configs get logged; the key must never appear in any representation."""
    secret = "sk_super_secret_value_1234"
    cfg = make_config(api_key=secret)
    assert secret not in repr(cfg)
    assert "redacted" in repr(cfg)


def test_key_fingerprint_shows_only_the_last_four_characters() -> None:
    """Enough to identify which key ran, not enough to reuse it."""
    cfg = make_config(api_key="sk_abcdefgh1234")
    assert cfg.key_fingerprint == "...1234"


def test_config_is_immutable() -> None:
    """A frozen config cannot drift mid-run."""
    cfg = make_config()
    with pytest.raises(Exception):
        cfg.source_lang = "hi-IN"  # type: ignore[misc]


def test_load_config_reads_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config picks the key up from a .env file rather than from code."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("SARVAM_BASE_URL", raising=False)
    env = tmp_path / ".env"
    env.write_text("SARVAM_API_KEY=sk_from_dotenv\n", encoding="utf-8")

    cfg = load_config(env_file=env, input_path=tmp_path / "in.wav")
    assert cfg.api_key == "sk_from_dotenv"
    assert cfg.base_url == DEFAULT_BASE_URL


def test_load_config_defaults_to_english_to_hindi(monkeypatch: pytest.MonkeyPatch) -> None:
    """The project's default direction is en-IN -> hi-IN."""
    monkeypatch.setenv("SARVAM_API_KEY", "sk_env_key")
    cfg = load_config()
    assert (cfg.source_lang, cfg.target_lang) == ("en-IN", "hi-IN")
    assert cfg.enable_duration_fit is True
    assert cfg.dry_run is False
