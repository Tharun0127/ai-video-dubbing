"""
Demux stage: extract the input's audio track as 16 kHz mono 16-bit PCM WAV.

Saaras wants 16 kHz mono WAV; the source here is 44.1 kHz stereo AAC inside an MP4,
so this is a genuine resample + downmix + re-encode, not a container passthrough.
The video stream is deliberately left alone -- M5 remuxes the *original* stream with
`-c:v copy`, so the picture is never re-encoded and never loses quality.

Both the source and the extracted file are probed and logged, so the conversion that
actually happened is on the record rather than assumed.

The extracted WAV is reused across runs to keep `--stage asr` instant, but only after
`demux_provenance.json` confirms it came from *this* input. That check is not paranoia:
every downstream stage keys off the WAV rather than the video, so reusing another input's
audio makes the whole pipeline dub the wrong source entirely from cache and report success.

Input:  config.input_path (any container ffmpeg can read).
Output: output/audio_16k_mono.wav, output/demux_provenance.json, and a DemuxResult
        carrying both probes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..audio import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    AudioError,
    extract_audio_16k_mono,
    probe_stream_info,
)
from ..config import Config

logger = logging.getLogger(__name__)

#: Filename the demuxed audio always gets, so a later --stage asr run can find it.
DEMUXED_WAV_NAME = "audio_16k_mono.wav"

#: Sidecar recording which input produced the WAV above. Without it, a second run on a
#: DIFFERENT input would silently reuse the previous input's audio -- see `_is_stale`.
PROVENANCE_NAME = "demux_provenance.json"


@dataclass(frozen=True)
class StreamInfo:
    """Measured properties of one audio stream, from ffprobe."""

    duration_s: float
    sample_rate: int | None
    channels: int | None
    codec_name: str | None

    @classmethod
    def from_probe(cls, probe: dict[str, Any]) -> StreamInfo:
        """Build from the dict `probe_stream_info` returns."""
        return cls(
            duration_s=float(probe.get("duration_s") or 0.0),
            sample_rate=probe.get("sample_rate"),  # type: ignore[arg-type]
            channels=probe.get("channels"),  # type: ignore[arg-type]
            codec_name=probe.get("codec_name"),  # type: ignore[arg-type]
        )

    def describe(self) -> str:
        """One-line human summary used in logs and the milestone report."""
        return (
            f"{self.codec_name or '?'} {self.sample_rate or '?'} Hz "
            f"{self.channels or '?'} ch, {self.duration_s:.3f}s"
        )


@dataclass(frozen=True)
class DemuxResult:
    """What the demux stage produced, with the before/after conversion on the record."""

    source_path: Path
    wav_path: Path
    source: StreamInfo
    extracted: StreamInfo

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the stage report."""
        return {
            "source_path": str(self.source_path),
            "wav_path": str(self.wav_path),
            "source": asdict(self.source),
            "extracted": asdict(self.extracted),
        }


def _provenance(source_path: Path, source: StreamInfo) -> dict[str, Any]:
    """Describe the input that a demuxed WAV was extracted from.

    Path, byte size, and measured duration together: the path alone would not notice a
    file edited in place, and the size alone would not notice two different clips that
    happen to be the same length.
    """
    return {
        "source_path": str(Path(source_path).resolve()),
        "source_size_bytes": Path(source_path).stat().st_size,
        "source_duration_s": round(source.duration_s, 6),
    }


def _stale_reason(
    wav_path: Path,
    provenance_path: Path,
    expected: dict[str, Any],
) -> str | None:
    """Return why an existing demuxed WAV cannot be reused, or None if it can be."""
    if not wav_path.exists():
        return None  # nothing to reuse; the caller extracts anyway
    if not provenance_path.exists():
        return f"{provenance_path.name} is missing, so its origin is unknown"

    try:
        recorded = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"{provenance_path.name} is unreadable ({exc})"

    for field in ("source_path", "source_size_bytes"):
        if recorded.get(field) != expected[field]:
            return f"{field} was {recorded.get(field)!r}, now {expected[field]!r}"

    recorded_duration = recorded.get("source_duration_s")
    if (
        not isinstance(recorded_duration, (int, float))
        or abs(float(recorded_duration) - expected["source_duration_s"]) > 0.01
    ):
        return (
            f"source_duration_s was {recorded_duration}, now {expected['source_duration_s']}"
        )
    return None


def run_demux(config: Config, *, force: bool = False) -> DemuxResult:
    """Extract config.input_path's audio to 16 kHz mono PCM WAV and report both formats."""
    if config.input_path is None:
        raise AudioError("demux requires --input")

    source_probe = probe_stream_info(config.input_path)
    source = StreamInfo.from_probe(source_probe)
    if source.codec_name is None:
        raise AudioError(f"{config.input_path} has no decodable audio stream to dub")

    wav_path = config.output_dir / DEMUXED_WAV_NAME
    provenance_path = config.output_dir / PROVENANCE_NAME
    provenance = _provenance(config.input_path, source)

    # Re-extracting is cheap and deterministic, but skipping it makes `--stage asr`
    # runs instant. The output is byte-identical either way, so the cache is unaffected.
    #
    # Reuse is only safe when the existing WAV came from THIS input. Skipping that check
    # let a run on a new input silently inherit the previous input's audio, and because
    # every downstream stage keys off the WAV rather than the video, the whole pipeline
    # then dubbed the wrong source from cache and reported success. Only the M6 post-mux
    # duration check caught it.
    stale_reason = _stale_reason(wav_path, provenance_path, provenance)
    if wav_path.exists() and not force and stale_reason is None:
        logger.info("demux: reusing existing %s (delete it to force re-extraction)", wav_path)
        extracted = StreamInfo.from_probe(probe_stream_info(wav_path))
    else:
        if wav_path.exists() and stale_reason is not None:
            logger.warning(
                "demux: re-extracting because %s does not belong to this input (%s)",
                wav_path.name, stale_reason,
            )
        extracted = StreamInfo.from_probe(
            extract_audio_16k_mono(config.input_path, wav_path, sample_rate=TARGET_SAMPLE_RATE)
        )
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    logger.info("demux: source    %s", source.describe())
    logger.info("demux: extracted %s", extracted.describe())

    if extracted.sample_rate != TARGET_SAMPLE_RATE or extracted.channels != TARGET_CHANNELS:
        raise AudioError(
            f"demux produced {extracted.sample_rate} Hz / {extracted.channels} ch, expected "
            f"{TARGET_SAMPLE_RATE} Hz / {TARGET_CHANNELS} ch -- Saaras needs 16 kHz mono"
        )

    # ffmpeg trims/pads container padding, so a small delta is normal; a large one means
    # a stream was dropped and every downstream timestamp would be wrong.
    drift = abs(extracted.duration_s - source.duration_s)
    if drift > 0.10:
        logger.warning(
            "demux: extracted audio is %.3fs but the source is %.3fs (%.3fs drift); "
            "downstream timestamps are anchored to the extracted audio",
            extracted.duration_s, source.duration_s, drift,
        )

    return DemuxResult(
        source_path=Path(config.input_path),
        wav_path=wav_path,
        source=source,
        extracted=extracted,
    )
