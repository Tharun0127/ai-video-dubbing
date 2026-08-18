"""
Configuration stage: typed, validated runtime configuration for the dubbing pipeline.

Loads secrets from `.env` (never from code), validates language codes against the
sets Sarvam actually documents, and holds every tunable constant the pipeline uses
in one place -- model identifiers, API limits, and published prices.

Input:  environment variables (.env) + parsed CLI arguments.
Output: a frozen `Config` dataclass consumed by every stage and by SarvamClient.

All API facts encoded here were confirmed against the live docs on 2026-07-28 and
are recorded with their source URL in docs/api-notes.md.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Confirmed Sarvam API facts (docs.sarvam.ai, verified 2026-07-28 -- see docs/api-notes.md)
# --------------------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.sarvam.ai"

#: Auth header name. Sarvam returns HTTP 403 (not 401) on auth failure.
AUTH_HEADER = "api-subscription-key"

#: Model identifiers, pinned so cache keys stay stable across runs.
#:
#: The translation default is mayura:v1 rather than the sarvam-translate:v1 named in
#: SPEC.md. That is a deliberate, measured deviation: sarvam-translate:v1 is formal-only,
#: and on this conversational source it mistranslated "top card" as "सबसे अच्छा कार्ड"
#: (best card) and "No way" as "कोई बात नहीं" (never mind). mayura:v1 in
#: classic-colloquial got both right. Full side-by-side in docs/m3-results.md.
ASR_MODEL = "saaras:v3"
TRANSLATE_MODEL = "mayura:v1"
TTS_MODEL = "bulbul:v3"

#: The synchronous /speech-to-text endpoint rejects audio longer than this (HTTP 422).
ASR_MAX_AUDIO_S = 30.0

#: Per-request input caps for the text endpoints. The translate cap is per *model*:
#: mayura:v1 rejects above 1000 chars, sarvam-translate:v1 above 2000 (HTTP 422).
TRANSLATE_MAX_CHARS = 2000
MAYURA_MAX_CHARS = 1000
TTS_MAX_CHARS = 2500

#: Translation models confirmed on docs.sarvam.ai (2026-07-29, see docs/api-notes.md).
TRANSLATE_MODELS: frozenset[str] = frozenset({"sarvam-translate:v1", "mayura:v1"})

#: Style modes accepted by POST /translate. Only mayura:v1 honours anything but `formal`.
TRANSLATE_MODES: frozenset[str] = frozenset(
    {"formal", "modern-colloquial", "classic-colloquial", "code-mixed"}
)

#: Per-model capability matrix. Encoded here so an unsupported combination fails locally
#: instead of costing a round-trip and returning a silently formal translation.
TRANSLATE_MODEL_MAX_CHARS: dict[str, int] = {
    "sarvam-translate:v1": TRANSLATE_MAX_CHARS,
    "mayura:v1": MAYURA_MAX_CHARS,
}
TRANSLATE_MODEL_MODES: dict[str, frozenset[str]] = {
    # Documented: "sarvam-translate:v1 = all 22 languages, formal only".
    "sarvam-translate:v1": frozenset({"formal"}),
    "mayura:v1": TRANSLATE_MODES,
}

#: Optional /translate knobs, validated locally before a request is built.
TRANSLATE_OUTPUT_SCRIPTS: frozenset[str] = frozenset(
    {"roman", "fully-native", "spoken-form-in-native"}
)
TRANSLATE_NUMERALS_FORMATS: frozenset[str] = frozenset({"international", "native"})
TRANSLATE_SPEAKER_GENDERS: frozenset[str] = frozenset({"Male", "Female"})

#: Bulbul's documented pace range. The pipeline deliberately clamps tighter than this
#: (see SPEC.md) because speech stops sounding human beyond roughly +/-25%.
#:
#: Direction confirmed by real calls, not by reading the docs alone (scripts/m4_probe.py,
#: recorded in docs/api-notes.md): a HIGHER pace produces SHORTER audio, i.e. faster
#: speech. The fit loop's `pace *= ratio` update depends on that sign and would drive the
#: wrong way if it were reversed.
TTS_API_PACE_MIN = 0.5
TTS_API_PACE_MAX = 2.0
TTS_PERCEPTUAL_PACE_MIN = 0.85
TTS_PERCEPTUAL_PACE_MAX = 1.25

#: Models and voices confirmed on docs.sarvam.ai (2026-07-29).
TTS_MODELS: frozenset[str] = frozenset({"bulbul:v3", "bulbul:v2"})

#: bulbul:v3's documented speaker list, lowercase as the API requires.
TTS_SPEAKERS_V3: frozenset[str] = frozenset(
    {
        "shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran",
        "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan", "sumit",
        "roopa", "kabir", "aayan", "ashutosh", "advait", "anand", "tanya", "tarun",
        "sunny", "mani", "gokul", "vijay", "shruti", "suhani", "mohit", "kavitha",
        "rehan", "soham", "rupali",
    }
)

#: Fixed speaker for v1 (SPEC.md defers voice selection and cloning to the stretch list).
TTS_SPEAKER = "shubh"

#: Output format. 24 kHz is the documented v3 default; WAV keeps decoding in the stdlib.
TTS_SAMPLE_RATE = 24000
TTS_SAMPLE_RATES: frozenset[int] = frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000})
TTS_OUTPUT_CODEC = "wav"

#: Published prices in INR (docs.sarvam.ai/api/getting-started/pricing).
PRICE_STT_INR_PER_HOUR = 30.0
PRICE_STT_DIARIZED_INR_PER_HOUR = 45.0
PRICE_TRANSLATE_INR_PER_10K_CHARS = 20.0
PRICE_TTS_BULBUL_V3_INR_PER_10K_CHARS = 30.0

#: Languages accepted by /speech-to-text when model=saaras:v3.
ASR_LANGUAGES: frozenset[str] = frozenset(
    {
        "unknown", "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN",
        "ta-IN", "te-IN", "en-IN", "gu-IN", "as-IN", "ur-IN", "ne-IN", "kok-IN",
        "ks-IN", "sd-IN", "sa-IN", "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN",
    }
)

#: Output modes for saaras:v3.
ASR_MODES: frozenset[str] = frozenset(
    {"transcribe", "translate", "verbatim", "translit", "codemix"}
)

#: Languages accepted by POST /translate. sarvam-translate:v1 covers all 23; mayura:v1
#: covers 10 Indian languages plus English (docs.sarvam.ai/api/getting-started/models).
TRANSLATE_LANGUAGES: frozenset[str] = ASR_LANGUAGES - {"unknown"}
MAYURA_LANGUAGES: frozenset[str] = frozenset(
    {
        "en-IN", "hi-IN", "bn-IN", "ta-IN", "te-IN", "gu-IN",
        "kn-IN", "ml-IN", "mr-IN", "pa-IN", "od-IN",
    }
)
TRANSLATE_MODEL_LANGUAGES: dict[str, frozenset[str]] = {
    "sarvam-translate:v1": TRANSLATE_LANGUAGES,
    "mayura:v1": MAYURA_LANGUAGES,
}

#: Languages bulbul:v3 can speak -- documented as the same 11 mayura:v1 translates into,
#: which is convenient: any target the translator accepts, the synthesiser can voice.
TTS_LANGUAGES: frozenset[str] = MAYURA_LANGUAGES

#: Pipeline stage names, in execution order. Used by --stage and by metrics.
STAGE_ORDER: tuple[str, ...] = (
    "demux", "asr", "translate", "tts", "assemble", "mux", "qc",
)

# --------------------------------------------------------------------------------------
# M2 chunking and segmentation defaults (see src/stages/asr.py for the rationale)
# --------------------------------------------------------------------------------------

#: Silence threshold in dBFS. Quieter than this for MIN_SILENCE_S counts as a pause worth
#: cutting on. Relaxed automatically (with a WARNING) if a clip's noise floor sits above it.
SILENCE_THRESHOLD_DB = -40.0

#: Shortest pause treated as a legal cut point; below this it is usually a stop consonant.
MIN_SILENCE_S = 0.30

#: Longest chunk sent to /speech-to-text. Kept at the API cap; chunks are usually smaller
#: because the planner balances them across the clip.
MAX_CHUNK_S = ASR_MAX_AUDIO_S

#: Segments shorter than this make M4 duration fitting unstable, so they are merged.
MIN_SEGMENT_S = 1.0

#: Segments longer than this make timing drift unrecoverable, so they are split.
MAX_SEGMENT_S = 15.0

# --------------------------------------------------------------------------------------
# M3 translation defaults (see src/stages/translate.py for the rationale)
# --------------------------------------------------------------------------------------

#: Default style mode, chosen with TRANSLATE_MODEL above because the style modes only
#: exist on mayura:v1. `classic-colloquial` was picked from the measured register
#: comparison, not from the API default (`formal`): it is the only variant that keeps one
#: consistent level of address across all four segments, and it translates the emotional
#: content ("वाह, ये तो कमाल है") instead of leaving it in English the way
#: modern-colloquial does ("वाह, amazing है"). It also expands the most -- mean estimated
#: syllable ratio 1.53 against 1.30 for modern-colloquial -- which is the price M4 pays.
TRANSLATE_MODE = "classic-colloquial"

#: Most segments packed into one /translate request. The character cap usually binds first;
#: this is a second ceiling so one batch failing to parse never costs the whole clip.
TRANSLATE_MAX_BATCH_SEGMENTS = 12

#: Neighbouring segments included in a batch purely as discourse context. Their
#: translations are discarded. /translate has no context parameter (confirmed against the
#: live schema), so context can only be carried by putting the neighbours in the request.
TRANSLATE_CONTEXT_SEGMENTS = 2

#: Fraction of a model's character cap a batch may fill before it is closed. The reply is
#: longer than the request for en->hi, and the cap applies to the input, so this is purely
#: a safety margin against an off-by-a-few packing bug, not an API requirement.
TRANSLATE_BATCH_FILL = 0.8

# --------------------------------------------------------------------------------------
# M4 TTS + duration-fit defaults (see src/stages/tts.py for the loop itself)
# --------------------------------------------------------------------------------------

#: A segment is "fitted" once achieved/target lands inside this band. Widening it saves
#: credits; narrowing it burns attempts chasing drift no listener can hear.
FIT_RATIO_MIN = 0.95
FIT_RATIO_MAX = 1.05

#: Hard ceiling on synthesis calls per segment, including the pace=1.0 baseline.
FIT_MAX_ATTEMPTS = 3

#: Once a proposed pace moves less than this, the loop has either converged or is pinned
#: against a clamp; continuing just spends credits for an inaudible change.
FIT_MIN_PACE_STEP = 0.02

#: Silence-trim settings applied to every TTS response before it is measured.
TTS_TRIM_THRESHOLD_DB = -40.0
TTS_TRIM_WINDOW_MS = 10.0
TTS_TRIM_KEEP_MARGIN_MS = 20.0

# --------------------------------------------------------------------------------------
# M5 assemble + mux defaults (see src/stages/assemble.py and src/stages/mux.py)
# --------------------------------------------------------------------------------------

#: Linear fade applied to each end of every placed clip. SPEC.md asks for 10-20 ms; 15 ms
#: is the midpoint. Its only job is to remove the step discontinuity -- and therefore the
#: click -- where a clip meets the silent track.
ASSEMBLE_FADE_MS = 15.0

#: Filename the assembled continuous dub always gets, so `--stage mux` and `--stage qc`
#: can find it without re-running the earlier stages.
DUBBED_AUDIO_NAME = "dubbed_audio.wav"

#: Audio encoder for the remux. The video stream is copied (`-c:v copy`) and never
#: re-encoded, but PCM cannot go into an MP4, so the new audio track is encoded to AAC.
MUX_AUDIO_CODEC = "aac"
MUX_AUDIO_BITRATE = "192k"

#: Tolerance between the source's duration and the muxed output's, in seconds. Container
#: rounding and AAC's encoder priming account for a few tens of milliseconds; more than
#: this means a stream was truncated.
MUX_DURATION_TOLERANCE_S = 0.25

# --------------------------------------------------------------------------------------
# M6 QC defaults (see src/qc.py)
# --------------------------------------------------------------------------------------

#: SPEC.md's timing threshold: segments drifting more than this are counted and flagged.
QC_DRIFT_THRESHOLD_PCT = 5.0

#: How many worst segments the QC report ranks by default.
QC_FLAG_TOP_N = 5

#: Weights combining the two independent failure modes into one ranking score. Semantic
#: error (WER) and timing error (absolute drift, expressed as a fraction) are not on the
#: same scale, so the weights are stated here rather than buried in the ranking code.
QC_WER_WEIGHT = 1.0
QC_DRIFT_WEIGHT = 1.0


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid; never swallowed silently."""


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration for a single pipeline invocation."""

    # --- secrets & endpoint -------------------------------------------------------
    api_key: str
    base_url: str = DEFAULT_BASE_URL

    # --- job inputs ---------------------------------------------------------------
    input_path: Path | None = None
    output_path: Path = Path("output/dubbed.mp4")
    source_lang: str = "en-IN"
    target_lang: str = "hi-IN"

    # --- run control --------------------------------------------------------------
    dry_run: bool = False
    max_segments: int | None = None
    enable_duration_fit: bool = True
    stage: str | None = None
    verbose: bool = False

    # --- directories --------------------------------------------------------------
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("output")

    # --- HTTP behaviour -----------------------------------------------------------
    request_timeout_s: float = 120.0
    max_retries: int = 3
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0

    # --- models -------------------------------------------------------------------
    asr_model: str = ASR_MODEL
    translate_model: str = TRANSLATE_MODEL
    tts_model: str = TTS_MODEL
    asr_mode: str = "transcribe"

    # --- chunking & segmentation (M2) ----------------------------------------------
    silence_threshold_db: float = SILENCE_THRESHOLD_DB
    min_silence_s: float = MIN_SILENCE_S
    max_chunk_s: float = MAX_CHUNK_S
    min_segment_s: float = MIN_SEGMENT_S
    max_segment_s: float = MAX_SEGMENT_S

    # --- translation (M3) -----------------------------------------------------------
    translate_mode: str = TRANSLATE_MODE
    translate_batch_segments: int = TRANSLATE_MAX_BATCH_SEGMENTS
    translate_context_segments: int = TRANSLATE_CONTEXT_SEGMENTS
    #: Disable batching entirely and send one request per segment (for A/B measurement).
    translate_no_batch: bool = False

    # --- TTS + duration fit (M4) ----------------------------------------------------
    tts_speaker: str = TTS_SPEAKER
    tts_sample_rate: int = TTS_SAMPLE_RATE
    fit_ratio_min: float = FIT_RATIO_MIN
    fit_ratio_max: float = FIT_RATIO_MAX
    fit_max_attempts: int = FIT_MAX_ATTEMPTS
    pace_min: float = TTS_PERCEPTUAL_PACE_MIN
    pace_max: float = TTS_PERCEPTUAL_PACE_MAX
    trim_threshold_db: float = TTS_TRIM_THRESHOLD_DB

    # --- assemble + mux (M5) ---------------------------------------------------------
    fade_ms: float = ASSEMBLE_FADE_MS
    audio_bitrate: str = MUX_AUDIO_BITRATE

    # --- QC (M6) ----------------------------------------------------------------------
    qc_drift_threshold_pct: float = QC_DRIFT_THRESHOLD_PCT
    qc_flag_top_n: int = QC_FLAG_TOP_N

    #: Populated in __post_init__; kept out of __repr__ so the key is never printed.
    _redacted: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ConfigError(
                "SARVAM_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://dashboard.sarvam.ai"
            )
        # A key pasted with its placeholder wrapper still "looks set" but produces an
        # opaque HTTP 403 at call time. Catch it here with a message that names the fix.
        if self.api_key[0] in "<\"'" or self.api_key[-1] in ">\"'":
            raise ConfigError(
                "SARVAM_API_KEY appears to be wrapped in placeholder brackets or quotes. "
                "Store the bare key in .env, e.g. SARVAM_API_KEY=sk_xxx (no <>, no quotes)."
            )
        if self.base_url.rstrip("/").endswith("/v1"):
            # /v1 is only the prefix for Sarvam's OpenAI-compatible chat endpoint; the
            # speech and translate endpoints live at the root and return 404 under /v1.
            raise ConfigError(
                f"SARVAM_BASE_URL is {self.base_url!r}. The speech and translation endpoints "
                f"are served from the root ({DEFAULT_BASE_URL}); only chat completions use /v1. "
                f"Set SARVAM_BASE_URL={DEFAULT_BASE_URL}"
            )
        if self.source_lang not in ASR_LANGUAGES:
            raise ConfigError(
                f"source-lang {self.source_lang!r} is not a Sarvam language code. "
                f"Valid: {sorted(ASR_LANGUAGES)}"
            )
        if self.target_lang not in ASR_LANGUAGES:
            raise ConfigError(
                f"target-lang {self.target_lang!r} is not a Sarvam language code. "
                f"Valid: {sorted(ASR_LANGUAGES)}"
            )
        if self.stage is not None and self.stage not in STAGE_ORDER:
            raise ConfigError(
                f"stage {self.stage!r} is unknown. Valid stages: {list(STAGE_ORDER)}"
            )
        if self.max_segments is not None and self.max_segments < 1:
            raise ConfigError("--max-segments must be >= 1")
        if self.asr_mode not in ASR_MODES:
            raise ConfigError(
                f"asr_mode {self.asr_mode!r} is not one of {sorted(ASR_MODES)}"
            )
        if self.silence_threshold_db >= 0:
            raise ConfigError(
                f"--silence-threshold-db is dBFS and must be negative, got "
                f"{self.silence_threshold_db}"
            )
        if self.min_silence_s <= 0:
            raise ConfigError(f"--min-silence-s must be > 0, got {self.min_silence_s}")
        if not 0 < self.max_chunk_s <= ASR_MAX_AUDIO_S:
            raise ConfigError(
                f"--max-chunk-s must be in (0, {ASR_MAX_AUDIO_S:.0f}]; the sync "
                f"/speech-to-text endpoint returns HTTP 422 beyond that. Got {self.max_chunk_s}"
            )
        if self.min_segment_s <= 0:
            raise ConfigError(f"--min-segment-s must be > 0, got {self.min_segment_s}")
        if self.max_segment_s <= self.min_segment_s:
            raise ConfigError(
                f"--max-segment-s ({self.max_segment_s}) must exceed --min-segment-s "
                f"({self.min_segment_s})"
            )
        if self.translate_model not in TRANSLATE_MODELS:
            raise ConfigError(
                f"translate_model {self.translate_model!r} is not one of "
                f"{sorted(TRANSLATE_MODELS)}"
            )
        allowed_modes = TRANSLATE_MODEL_MODES[self.translate_model]
        if self.translate_mode not in TRANSLATE_MODES:
            raise ConfigError(
                f"--translate-mode {self.translate_mode!r} is not one of {sorted(TRANSLATE_MODES)}"
            )
        if self.translate_mode not in allowed_modes:
            # Sending an unsupported mode does not error -- it silently returns formal
            # output, which would make the register comparison meaningless.
            raise ConfigError(
                f"model {self.translate_model} supports mode(s) {sorted(allowed_modes)}, "
                f"not {self.translate_mode!r}. mayura:v1 is the model with style modes."
            )
        for code, label in ((self.source_lang, "source"), (self.target_lang, "target")):
            # "unknown" is the ASR auto-detect sentinel and is legal for --source-lang.
            # /translate has no equivalent (mayura:v1 spells it "auto", sarvam-translate:v1
            # has none at all), so the translate stage rejects it there instead of guessing
            # a mapping between two different vendors' sentinels.
            if code == "unknown":
                continue
            if code not in TRANSLATE_MODEL_LANGUAGES[self.translate_model]:
                raise ConfigError(
                    f"{label}-lang {code!r} is not supported by {self.translate_model}. "
                    f"Supported: {sorted(TRANSLATE_MODEL_LANGUAGES[self.translate_model])}"
                )
        if self.translate_batch_segments < 1:
            raise ConfigError("--translate-batch-segments must be >= 1")
        if self.translate_context_segments < 0:
            raise ConfigError("--translate-context-segments must be >= 0")
        if self.tts_model not in TTS_MODELS:
            raise ConfigError(f"tts_model {self.tts_model!r} is not one of {sorted(TTS_MODELS)}")
        if self.tts_model == "bulbul:v3" and self.tts_speaker not in TTS_SPEAKERS_V3:
            raise ConfigError(
                f"--tts-speaker {self.tts_speaker!r} is not a bulbul:v3 voice. "
                f"Speakers are lowercase and model-specific; valid: {sorted(TTS_SPEAKERS_V3)}"
            )
        if self.target_lang not in TTS_LANGUAGES:
            raise ConfigError(
                f"target-lang {self.target_lang!r} cannot be synthesised by {self.tts_model}. "
                f"Supported: {sorted(TTS_LANGUAGES)}"
            )
        if self.tts_sample_rate not in TTS_SAMPLE_RATES:
            raise ConfigError(
                f"--tts-sample-rate {self.tts_sample_rate} is not one of "
                f"{sorted(TTS_SAMPLE_RATES)}"
            )
        if not 0 < self.fit_ratio_min < 1 < self.fit_ratio_max:
            raise ConfigError(
                f"the fit band must straddle 1.0, got "
                f"[{self.fit_ratio_min}, {self.fit_ratio_max}]"
            )
        if self.fit_max_attempts < 1:
            raise ConfigError("--fit-max-attempts must be >= 1")
        if not TTS_API_PACE_MIN <= self.pace_min < self.pace_max <= TTS_API_PACE_MAX:
            raise ConfigError(
                f"the pace clamp [{self.pace_min}, {self.pace_max}] must sit inside the "
                f"API's documented range [{TTS_API_PACE_MIN}, {TTS_API_PACE_MAX}] and be "
                f"ordered. The tighter default is deliberate: see SPEC.md."
            )
        if self.trim_threshold_db >= 0:
            raise ConfigError(
                f"--trim-threshold-db is dBFS and must be negative, got {self.trim_threshold_db}"
            )
        if self.fade_ms < 0:
            raise ConfigError(f"--fade-ms must be >= 0, got {self.fade_ms}")
        if self.qc_drift_threshold_pct <= 0:
            raise ConfigError(
                f"--qc-drift-threshold-pct must be > 0, got {self.qc_drift_threshold_pct}"
            )
        if self.qc_flag_top_n < 1:
            raise ConfigError(f"--qc-flag-top-n must be >= 1, got {self.qc_flag_top_n}")

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Repr with the API key redacted, so configs are safe to log."""
        return (
            f"Config(base_url={self.base_url!r}, input_path={self.input_path!r}, "
            f"source_lang={self.source_lang!r}, target_lang={self.target_lang!r}, "
            f"dry_run={self.dry_run}, max_segments={self.max_segments}, "
            f"enable_duration_fit={self.enable_duration_fit}, stage={self.stage!r}, "
            f"translate_model={self.translate_model!r}, translate_mode={self.translate_mode!r}, "
            f"cache_dir={str(self.cache_dir)!r}, api_key='***redacted***')"
        )

    @property
    def key_fingerprint(self) -> str:
        """Last 4 characters of the API key, for logs that need to identify which key ran."""
        return f"...{self.api_key[-4:]}" if len(self.api_key) >= 4 else "***"


def load_config(
    *,
    input_path: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] = "output/dubbed.mp4",
    source_lang: str = "en-IN",
    target_lang: str = "hi-IN",
    dry_run: bool = False,
    max_segments: int | None = None,
    enable_duration_fit: bool = True,
    stage: str | None = None,
    verbose: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
    env_file: str | os.PathLike[str] | None = None,
    asr_mode: str = "transcribe",
    silence_threshold_db: float = SILENCE_THRESHOLD_DB,
    min_silence_s: float = MIN_SILENCE_S,
    max_chunk_s: float = MAX_CHUNK_S,
    min_segment_s: float = MIN_SEGMENT_S,
    max_segment_s: float = MAX_SEGMENT_S,
    translate_model: str = TRANSLATE_MODEL,
    translate_mode: str = TRANSLATE_MODE,
    translate_batch_segments: int = TRANSLATE_MAX_BATCH_SEGMENTS,
    translate_context_segments: int = TRANSLATE_CONTEXT_SEGMENTS,
    translate_no_batch: bool = False,
    tts_speaker: str = TTS_SPEAKER,
    tts_sample_rate: int = TTS_SAMPLE_RATE,
    fit_ratio_min: float = FIT_RATIO_MIN,
    fit_ratio_max: float = FIT_RATIO_MAX,
    fit_max_attempts: int = FIT_MAX_ATTEMPTS,
    pace_min: float = TTS_PERCEPTUAL_PACE_MIN,
    pace_max: float = TTS_PERCEPTUAL_PACE_MAX,
    trim_threshold_db: float = TTS_TRIM_THRESHOLD_DB,
    fade_ms: float = ASSEMBLE_FADE_MS,
    audio_bitrate: str = MUX_AUDIO_BITRATE,
    qc_drift_threshold_pct: float = QC_DRIFT_THRESHOLD_PCT,
    qc_flag_top_n: int = QC_FLAG_TOP_N,
) -> Config:
    """Load .env, merge it with CLI arguments, and return a validated Config."""
    load_dotenv(dotenv_path=env_file, override=False)

    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    base_url = os.getenv("SARVAM_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    resolved_cache = Path(cache_dir or os.getenv("SARVAM_CACHE_DIR", "cache"))

    cfg = Config(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        input_path=Path(input_path) if input_path is not None else None,
        output_path=Path(output_path),
        source_lang=source_lang,
        target_lang=target_lang,
        dry_run=dry_run,
        max_segments=max_segments,
        enable_duration_fit=enable_duration_fit,
        stage=stage,
        verbose=verbose,
        cache_dir=resolved_cache,
        output_dir=Path(output_path).parent or Path("output"),
        asr_mode=asr_mode,
        silence_threshold_db=silence_threshold_db,
        min_silence_s=min_silence_s,
        max_chunk_s=max_chunk_s,
        min_segment_s=min_segment_s,
        max_segment_s=max_segment_s,
        translate_model=translate_model,
        translate_mode=translate_mode,
        translate_batch_segments=translate_batch_segments,
        translate_context_segments=translate_context_segments,
        translate_no_batch=translate_no_batch,
        tts_speaker=tts_speaker,
        tts_sample_rate=tts_sample_rate,
        fit_ratio_min=fit_ratio_min,
        fit_ratio_max=fit_ratio_max,
        fit_max_attempts=fit_max_attempts,
        pace_min=pace_min,
        pace_max=pace_max,
        trim_threshold_db=trim_threshold_db,
        fade_ms=fade_ms,
        audio_bitrate=audio_bitrate,
        qc_drift_threshold_pct=qc_drift_threshold_pct,
        qc_flag_top_n=qc_flag_top_n,
    )
    logger.debug("Loaded %r (key %s)", cfg, cfg.key_fingerprint)
    return cfg


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once; DEBUG shows API request/response bodies."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-20s %(message)s",
        datefmt="%H:%M:%S",
    )
    # requests/urllib3 DEBUG leaks header names into logs and is far too noisy.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
