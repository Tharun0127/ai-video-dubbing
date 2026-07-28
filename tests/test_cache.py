"""Tests for cache key correctness and the disk cache round-trip.

Cache key correctness is safety-critical: a key that collides across different
parameters would silently serve the wrong audio, and a key that changes when it
shouldn't would silently burn credits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cache import (
    DiskCache,
    cache_key,
    canonical_params,
    hash_bytes,
    hash_file,
    hash_text,
)

BASE_PARAMS = {"model": "saaras:v3", "mode": "transcribe", "language_code": "en-IN"}


def test_cache_key_is_deterministic() -> None:
    """The same inputs always produce the same key."""
    a = cache_key("asr", "saaras:v3", BASE_PARAMS, "deadbeef")
    b = cache_key("asr", "saaras:v3", BASE_PARAMS, "deadbeef")
    assert a == b
    assert len(a) == 64


def test_cache_key_ignores_param_ordering() -> None:
    """Dict insertion order must not change the key."""
    reordered = {"language_code": "en-IN", "mode": "transcribe", "model": "saaras:v3"}
    assert cache_key("asr", "saaras:v3", BASE_PARAMS, "x") == cache_key(
        "asr", "saaras:v3", reordered, "x"
    )


def test_cache_key_treats_none_as_absent() -> None:
    """Explicitly passing None must hash the same as omitting the parameter."""
    with_none = {**BASE_PARAMS, "input_audio_codec": None}
    assert cache_key("asr", "saaras:v3", BASE_PARAMS, "x") == cache_key(
        "asr", "saaras:v3", with_none, "x"
    )


@pytest.mark.parametrize(
    "stage,model,params,input_hash",
    [
        ("translate", "saaras:v3", BASE_PARAMS, "x"),                    # different stage
        ("asr", "saarika:v2.5", BASE_PARAMS, "x"),                       # different model
        ("asr", "saaras:v3", {**BASE_PARAMS, "mode": "translate"}, "x"), # different param
        ("asr", "saaras:v3", {**BASE_PARAMS, "language_code": "hi-IN"}, "x"),
        ("asr", "saaras:v3", BASE_PARAMS, "y"),                          # different input
    ],
)
def test_cache_key_changes_when_any_component_changes(
    stage: str, model: str, params: dict[str, str], input_hash: str
) -> None:
    """Changing stage, model, any parameter, or the input must force a fresh key."""
    baseline = cache_key("asr", "saaras:v3", BASE_PARAMS, "x")
    assert cache_key(stage, model, params, input_hash) != baseline


def test_canonical_params_is_stable_and_sorted() -> None:
    """Canonical serialisation sorts keys and drops Nones."""
    assert canonical_params({"b": 2, "a": 1, "c": None}) == '{"a":1,"b":2}'


def test_hash_helpers_agree(tmp_path: Path) -> None:
    """hash_file, hash_bytes, and hash_text agree on the same content."""
    content = "नमस्ते world"
    path = tmp_path / "sample.txt"
    path.write_bytes(content.encode("utf-8"))
    assert hash_file(path) == hash_text(content) == hash_bytes(content.encode("utf-8"))


def test_put_then_get_round_trips(cache: DiskCache) -> None:
    """A written entry comes back with its payload intact and counts as a hit."""
    key = cache_key("asr", "saaras:v3", BASE_PARAMS, "abc")
    payload = {"transcript": "hello", "request_id": "r1"}
    cache.put("asr", key, model="saaras:v3", params=BASE_PARAMS, input_hash="abc", payload=payload)

    envelope = cache.get("asr", key)
    assert envelope is not None
    assert envelope["payload"] == payload
    assert envelope["input_hash"] == "abc"
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0
    assert cache.stats.writes == 1


def test_get_on_unknown_key_is_a_miss(cache: DiskCache) -> None:
    """An absent key returns None and increments the miss counter."""
    assert cache.get("asr", "0" * 64) is None
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0


def test_binary_payload_round_trips(cache: DiskCache) -> None:
    """TTS audio bytes survive the round-trip in a .bin sidecar."""
    key = cache_key("tts", "bulbul:v3", {"pace": 1.0}, "text-hash")
    audio = b"RIFF____WAVEfmt "
    cache.put("tts", key, model="bulbul:v3", params={"pace": 1.0},
              input_hash="text-hash", payload={"ok": True}, binary=audio)

    envelope = cache.get("tts", key)
    assert envelope is not None
    assert envelope["binary"] == audio
    assert envelope["has_binary"] is True


def test_corrupt_entry_is_a_miss_not_a_crash(cache: DiskCache, cache_dir: Path) -> None:
    """A truncated JSON envelope degrades to a cache miss with a warning."""
    key = "f" * 64
    entry = cache_dir / "asr" / f"{key}.json"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("{not json", encoding="utf-8")

    assert cache.get("asr", key) is None
    assert cache.stats.misses == 1


def test_disabled_cache_always_misses(cache_dir: Path) -> None:
    """A disabled cache never serves or writes anything."""
    disabled = DiskCache(cache_dir, enabled=False)
    key = cache_key("asr", "saaras:v3", BASE_PARAMS, "abc")
    disabled.put("asr", key, model="saaras:v3", params=BASE_PARAMS,
                 input_hash="abc", payload={"transcript": "x"})
    assert disabled.get("asr", key) is None
    assert disabled.stats.writes == 0


def test_clear_removes_every_entry(cache: DiskCache) -> None:
    """clear() empties the cache and reports how many files went."""
    for i in range(3):
        cache.put("asr", f"{i:064d}", model="m", params={}, input_hash=str(i), payload={"i": i})
    assert cache.clear() == 3
    assert cache.get("asr", f"{0:064d}") is None


def test_stats_to_dict_reports_hit_rate(cache: DiskCache) -> None:
    """Stats serialise with a hit rate for the end-of-run summary."""
    key = cache_key("asr", "m", {}, "h")
    cache.put("asr", key, model="m", params={}, input_hash="h", payload={})
    cache.get("asr", key)
    cache.get("asr", "1" * 64)

    stats = cache.stats.to_dict()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["cache_hit_rate"] == 0.5
