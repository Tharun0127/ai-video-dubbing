"""
M3 probe: confirm the /translate wire shape, then measure register and batching for real.

The project standards forbid implementing against a guessed API, so this script does three things
against the live endpoint before the stage is trusted, and writes everything it observes
to output/m3_register_probe.json:

  1. One raw call, whose unmodified JSON response is printed -- the confirmed shape.
  2. The register comparison: the same four real segments translated by
     sarvam-translate:v1 and by mayura:v1 in each of its four style modes, side by side.
     This doubles as the batching test, since every variant is sent as one numbered-line
     request and the reply is parsed with the pipeline's own strict parser.
  3. A context ablation: the same segments translated one per request with no
     neighbours, so the effect of the context window can be seen rather than assumed.

Every call goes through SarvamClient, so a second run is served from cache and costs Rs 0.

Usage:
    python -m scripts.m3_probe [--segments output/segments.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.cache import DiskCache
from src.config import load_config, setup_logging
from src.metrics import MetricsCollector
from src.sarvam_client import SarvamClient
from src.stages.translate import (
    count_syllables,
    measure_expansion,
    parse_batch,
    render_batch,
)

logger = logging.getLogger("m3_probe")

#: The variants compared. mayura:v1 is the only model with style modes; sarvam-translate:v1
#: is documented as formal-only, and is included as the baseline the SPEC names.
VARIANTS: list[tuple[str, str, str]] = [
    ("sarvam-translate:v1", "formal", "SPEC default; the only mode this model supports"),
    ("mayura:v1", "formal", "same register, different model -- isolates model from mode"),
    ("mayura:v1", "modern-colloquial", "contemporary conversational"),
    ("mayura:v1", "classic-colloquial", "traditional conversational"),
    ("mayura:v1", "code-mixed", "Hindi-English mixing, as urban speakers actually talk"),
]


def main(argv: list[str] | None = None) -> int:
    """Run the probe end to end and write the observations to disk."""
    parser = argparse.ArgumentParser(prog="python -m scripts.m3_probe")
    parser.add_argument("--segments", type=Path, default=Path("output/segments.json"))
    parser.add_argument("--out", type=Path, default=Path("output/m3_register_probe.json"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    segments = json.loads(args.segments.read_text(encoding="utf-8"))
    texts = [str(s["text"]) for s in segments]
    logger.info("probing with %d real segment(s) from %s", len(texts), args.segments)

    config = load_config(input_path=None, source_lang="en-IN", target_lang="hi-IN")
    metrics = MetricsCollector()
    observations: dict[str, Any] = {
        "endpoint": "POST /translate",
        "segments_source": str(args.segments),
        "source_texts": texts,
    }

    with SarvamClient(config, cache=DiskCache(config.cache_dir), metrics=metrics) as client:
        # --- 1. raw shape -------------------------------------------------------------
        raw = client.translate(
            texts[0],
            source_language_code="en-IN",
            target_language_code="hi-IN",
            model="sarvam-translate:v1",
            mode="formal",
            stage="translate_probe",
        )
        print("\n--- raw /translate response (unmodified) ---")
        print(json.dumps(raw, indent=2, ensure_ascii=False))
        print("--- response keys:", sorted(raw.keys()), "---\n")
        observations["raw_response"] = raw
        observations["response_keys"] = sorted(raw.keys())

        # --- 2. register comparison, one request per segment --------------------------
        # Isolated calls, so the comparison measures register alone and is not confounded
        # by whether a given model happens to survive the batching protocol.
        variants: list[dict[str, Any]] = []
        for model, mode, why in VARIANTS:
            isolated: list[str] = []
            for text in texts:
                reply = client.translate(
                    text,
                    source_language_code="en-IN",
                    target_language_code="hi-IN",
                    model=model, mode=mode, numerals_format="native",
                    stage="translate_probe",
                )
                isolated.append((reply.get("translated_text") or "").strip())
            variants.append({
                "model": model,
                "mode": mode,
                "rationale": why,
                "lines": isolated,
                "per_segment": [
                    _measure(texts[i], isolated[i], segments[i]) for i in range(len(texts))
                ],
            })
            logger.info("variant %-20s %-20s translated %d segment(s)",
                        model, mode, len(isolated))
        observations["register_variants"] = variants

        # --- 3. batching + context: the same segments as one numbered-line request ----
        body = render_batch(texts)
        observations["batch_request_body"] = body
        observations["batch_request_chars"] = len(body)

        batched: list[dict[str, Any]] = []
        for model, mode, _why in VARIANTS:
            reply = client.translate(
                body,
                source_language_code="en-IN",
                target_language_code="hi-IN",
                model=model, mode=mode, numerals_format="native",
                stage="translate_probe",
            )
            translated = reply.get("translated_text", "")
            parsed = parse_batch(translated, len(texts))
            isolated = next(
                v["lines"] for v in variants if v["model"] == model and v["mode"] == mode
            )
            batched.append({
                "model": model,
                "mode": mode,
                "raw_reply": translated,
                "round_tripped": parsed is not None,
                "lines": parsed,
                "reply_lines_in_response": len(translated.splitlines()),
                "differs_from_isolated": (
                    [i for i in range(len(texts)) if parsed[i] != isolated[i]]
                    if parsed else None
                ),
            })
            logger.info("batch  %-20s %-20s round_tripped=%s",
                        model, mode, parsed is not None)
        observations["batch_protocol"] = batched

    observations["cost"] = {
        "network_calls": metrics.cache_misses,
        "cache_hits": metrics.cache_hits,
        "cost_inr": round(metrics.total_cost_inr, 6),
        "cost_avoided_by_cache_inr": round(metrics.cost_avoided_inr, 6),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(observations, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics.log_summary()
    logger.info("wrote %s", args.out)

    _print_side_by_side(texts, variants)
    _print_batch_result(observations["batch_protocol"])
    return 0


def _print_batch_result(batched: list[dict[str, Any]]) -> None:
    """Print whether each model preserved the numbered-line structure."""
    print("\n=== BATCHING PROTOCOL (4 segments as one numbered-line request) ===")
    print(f"  {'model / mode':<42} {'round-trips':<12} {'reply lines':<12} "
          f"{'differs from isolated'}")
    for row in batched:
        label = f"{row['model']} / {row['mode']}"
        differs = row["differs_from_isolated"]
        print(f"  {label:<42} {str(row['round_tripped']):<12} "
              f"{row['reply_lines_in_response']:<12} "
              f"{differs if differs is not None else 'n/a'}")


def _measure(source: str, target: str, segment: dict[str, Any]) -> dict[str, Any]:
    """Measure one source/translation pair, including the syllable estimates."""
    expansion = measure_expansion(
        source, target, source_lang="en-IN", target_lang="hi-IN",
        duration_s=float(segment["end"]) - float(segment["start"]),
    )
    return {"translation": target, **expansion.to_dict()}


def _print_side_by_side(texts: list[str], variants: list[dict[str, Any]]) -> None:
    """Print the register comparison as a per-segment side-by-side block."""
    print("\n=== REGISTER COMPARISON (same four real segments) ===")
    for index, source in enumerate(texts):
        print(f"\n[{index}] EN: {source}")
        for variant in variants:
            lines = variant["lines"]
            rendered = lines[index] if lines else "<did not round-trip>"
            label = f"{variant['model']} / {variant['mode']}"
            ratio = len(rendered) / len(source) if source else 0.0
            syllables = count_syllables(rendered, "hi-IN")
            print(f"    {label:<42} {rendered}")
            print(f"    {'':<42} (chars {len(rendered):>3}, ratio {ratio:.2f}, "
                  f"syllables~{syllables})")


if __name__ == "__main__":
    sys.exit(main())
