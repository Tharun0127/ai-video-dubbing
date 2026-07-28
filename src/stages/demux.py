"""
Demux stage: extract the input's audio track as 16 kHz mono 16-bit PCM WAV.

Saaras wants 16 kHz mono WAV; the source here is 44.1 kHz stereo AAC inside an MP4,
so this is a genuine resample + downmix + re-encode, not a container passthrough.
The video stream is deliberately left alone -- M5 remuxes the *original* stream with
`-c:v copy`, so the picture is never re-encoded and never loses quality.

Both the source and the extracted file are probed and logged, so the conversion that
actually happened is on the record rather than assumed.

Input:  config.input_path (any container ffmpeg can read).
Output: output/audio_16k_mono.wav + a DemuxResult carrying both probes.
"""

from __future__ import annotations

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


def run_demux(config: Config, *, force: bool = False) -> DemuxResult:
    """Extract config.input_path's audio to 16 kHz mono PCM WAV and report both formats."""
    if config.input_path is None:
        raise AudioError("demux requires --input")

    source_probe = probe_stream_info(config.input_path)
    source = StreamInfo.from_probe(source_probe)
    if source.codec_name is None:
        raise AudioError(f"{config.input_path} has no decodable audio stream to dub")

    wav_path = config.output_dir / DEMUXED_WAV_NAME

    # Re-extracting is cheap and deterministic, but skipping it makes `--stage asr`
    # runs instant. The output is byte-identical either way, so the cache is unaffected.
    if wav_path.exists() and not force:
        logger.info("demux: reusing existing %s (delete it to force re-extraction)", wav_path)
        extracted = StreamInfo.from_probe(probe_stream_info(wav_path))
    else:
        extracted = StreamInfo.from_probe(
            extract_audio_16k_mono(config.input_path, wav_path, sample_rate=TARGET_SAMPLE_RATE)
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
