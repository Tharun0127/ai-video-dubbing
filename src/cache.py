"""
Cache stage: content-addressed disk cache for every Sarvam API response.

Caching is a hard requirement, not an optimisation -- development runs the same
clip dozens of times on limited free credits. A second run with identical inputs
and parameters must issue zero network calls and cost zero rupees.

Cache key = sha256(stage + model + canonical_params + input_hash), so any change
to the audio, the text, the model, or a single parameter produces a new key and
correctly forces a fresh call.

Layout:
    cache/<stage>/<key>.json   envelope: key, stage, model, params, input_hash, payload
    cache/<stage>/<key>.bin    raw bytes (TTS audio) when the response is binary

Input:  stage name, model id, params dict, input hash.
Output: cached payload dict / bytes, plus hit/miss counters for metrics.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 1024 * 1024


class CacheMissError(RuntimeError):
    """Raised when --dry-run hits a cache miss; the run must fail loudly, not call the API."""


def hash_bytes(data: bytes) -> str:
    """Return the sha256 hex digest of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    """Return the sha256 hex digest of a string, encoded as UTF-8."""
    return hash_bytes(text.encode("utf-8"))


def hash_file(path: str | os.PathLike[str]) -> str:
    """Return the sha256 hex digest of a file's contents, streamed so large media is safe."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_params(params: dict[str, Any]) -> str:
    """Serialise params to a stable string so equivalent dicts always yield the same key."""
    # sort_keys makes ordering irrelevant; default=str keeps Paths/enums from breaking
    # the digest; keys with a None value are dropped so an explicitly-passed default
    # hashes the same as an omitted one.
    cleaned = {k: v for k, v in params.items() if v is not None}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(stage: str, model: str, params: dict[str, Any], input_hash: str) -> str:
    """Compute the deterministic cache key for one API call."""
    material = f"{stage}\x00{model}\x00{canonical_params(params)}\x00{input_hash}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    """Running hit/miss tally, reported in the end-of-run summary and metrics.json."""

    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def total(self) -> int:
        """Number of cache lookups performed this run."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        """Fraction of lookups served from disk, or None if no lookups happened."""
        return self.hits / self.total if self.total else None

    def to_dict(self) -> dict[str, Any]:
        """Serialise counters for metrics.json."""
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_writes": self.writes,
            "cache_hit_rate": self.hit_rate,
        }


class DiskCache:
    """Disk-backed cache mapping a content-addressed key to a JSON payload and optional bytes."""

    def __init__(self, root: str | os.PathLike[str] = "cache", enabled: bool = True) -> None:
        """Create (or attach to) a cache rooted at `root`; `enabled=False` makes every get a miss."""
        self.root = Path(root)
        self.enabled = enabled
        self.stats = CacheStats()

    def _paths(self, stage: str, key: str) -> tuple[Path, Path]:
        """Return the (envelope json, binary sidecar) paths for a key."""
        stage_dir = self.root / stage
        return stage_dir / f"{key}.json", stage_dir / f"{key}.bin"

    def get(self, stage: str, key: str) -> dict[str, Any] | None:
        """Look up a key; returns the envelope dict on hit, None on miss. Updates stats."""
        if not self.enabled:
            self.stats.misses += 1
            return None

        json_path, bin_path = self._paths(stage, key)
        if not json_path.exists():
            self.stats.misses += 1
            logger.debug("cache MISS stage=%s key=%s", stage, key[:12])
            return None

        try:
            envelope = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt entry is a miss, not a crash -- but it must be visible.
            logger.warning(
                "cache entry unreadable, treating as MISS: stage=%s key=%s path=%s error=%s",
                stage, key[:12], json_path, exc,
            )
            self.stats.misses += 1
            return None

        if bin_path.exists():
            envelope["binary"] = bin_path.read_bytes()

        self.stats.hits += 1
        logger.debug("cache HIT  stage=%s key=%s", stage, key[:12])
        return envelope

    def put(
        self,
        stage: str,
        key: str,
        *,
        model: str,
        params: dict[str, Any],
        input_hash: str,
        payload: dict[str, Any],
        binary: bytes | None = None,
    ) -> None:
        """Write an envelope (and optional binary sidecar) to disk atomically."""
        if not self.enabled:
            return

        json_path, bin_path = self._paths(stage, key)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        envelope = {
            "key": key,
            "stage": stage,
            "model": model,
            "params": {k: v for k, v in params.items() if v is not None},
            "input_hash": input_hash,
            "payload": payload,
            "has_binary": binary is not None,
        }

        if binary is not None:
            _atomic_write(bin_path, binary)
        _atomic_write(json_path, json.dumps(envelope, ensure_ascii=False, indent=2).encode("utf-8"))

        self.stats.writes += 1
        logger.debug("cache WRITE stage=%s key=%s bytes=%s", stage, key[:12], len(binary or b""))

    def clear(self) -> int:
        """Delete every cached entry; returns the number of files removed."""
        if not self.root.exists():
            return 0
        removed = 0
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                path.unlink()
                removed += 1
        logger.info("Cleared cache at %s (%d files)", self.root, removed)
        return removed


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes via a temp file + rename, so an interrupted run never leaves a half entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
