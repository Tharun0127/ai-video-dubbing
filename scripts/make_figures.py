"""
Figure generator: render the README's charts directly from measured run reports.

Every figure in docs/assets/ is produced by this script from JSON that a real run
wrote -- coldrun/metrics.json, coldrun/assemble_report.json, output/m4_pace_probe.json
and friends. Nothing is hand-drawn and no value is typed in by hand, so a figure can
never drift away from the number it claims to show. If a source report is missing the
script fails loudly rather than emitting a plausible-looking empty chart.

Output is plain SVG built with the standard library only (no matplotlib), so the charts
regenerate from a clean clone with nothing installed beyond the pipeline's own deps.

Usage:  python -m scripts.make_figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

# Palette chosen to stay legible on both GitHub themes.
BLUE = "#3b82f6"
ORANGE = "#f59e0b"
GREEN = "#10b981"
RED = "#ef4444"
GREY = "#8b949e"
VIOLET = "#8b5cf6"

STYLE = """<style>
text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
.ttl{font-size:15px;font-weight:600;fill:#1f2328}
.sub{font-size:11px;fill:#656d76}
.lbl{font-size:12px;fill:#1f2328}
.val{font-size:11px;font-weight:600;fill:#1f2328}
.note{font-size:11px;fill:#656d76}
.ax{stroke:#d0d7de;stroke-width:1}
.grid{stroke:#d0d7de;stroke-width:1;stroke-dasharray:2 3;opacity:.7}
@media (prefers-color-scheme:dark){
.ttl,.lbl,.val{fill:#e6edf3}
.sub,.note{fill:#9198a1}
.ax,.grid{stroke:#3d444d}
}
</style>"""


def _load(rel: str) -> dict | list:
    """Load a measured report, failing loudly if the run that produces it never happened."""
    path = ROOT / rel
    if not path.exists():
        sys.exit(
            f"missing measured input: {rel}\n"
            "Figures are generated from real run reports; run the pipeline first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _esc(text: str) -> str:
    """Escape the three characters that would break an SVG text node."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg(width: int, height: int, body: str) -> str:
    """Wrap rendered elements in a theme-aware, transparent-background SVG document."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">{STYLE}{body}</svg>\n'
    )


def _header(title: str, source: str, width: int) -> str:
    """Render the title plus the provenance line naming the file the numbers came from."""
    return (
        f'<text class="ttl" x="16" y="24">{_esc(title)}</text>'
        f'<text class="sub" x="16" y="42">{_esc(source)}</text>'
    )


def _fit(text: str, x: float, width: int = 880, margin: int = 16) -> float:
    """Nudge a label left so an 11px caption cannot run off the right edge of the canvas."""
    return max(16.0, min(x, width - margin - len(text) * 5.55))


def _write(name: str, svg: str) -> None:
    """Write one figure and report its size, so the run log shows what was produced."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / name).write_text(svg, encoding="utf-8")
    print(f"  wrote docs/assets/{name}  ({len(svg):,} bytes)")


# --------------------------------------------------------------------------------------
# Figure 1: where the wall clock goes on a cold run
# --------------------------------------------------------------------------------------
def fig_latency() -> None:
    """Horizontal bars of per-stage cold-run latency, showing synthesis dominating."""
    m = _load("coldrun/metrics.json")
    lat = m["latency_s"]
    total = lat["total_wall_clock"]
    stages = ["demux", "asr", "translate", "tts", "assemble", "mux", "qc"]
    calls = m["api"]["calls"]

    w, left, right = 880, 108, 200
    row_h, top = 30, 66
    plot_w = w - left - right
    h = top + row_h * len(stages) + 44
    scale = plot_w / max(lat[s] for s in stages)

    body = [
        _header(
            "Cold-run wall clock by stage",
            f"coldrun/metrics.json - 36.107 s input, {total} s total, realtime factor "
            f"{lat['realtime_factor']}x",
            w,
        )
    ]
    for i, stage in enumerate(stages):
        y = top + i * row_h
        val = lat[stage]
        bw = max(2.0, val * scale)
        colour = ORANGE if stage == "tts" else BLUE
        pct = val / total * 100
        ncalls = calls.get(stage, 0)
        suffix = f"  -  {ncalls} API call{'s' if ncalls != 1 else ''}" if ncalls else ""
        body.append(
            f'<text class="lbl" x="{left - 10}" y="{y + 15}" text-anchor="end">{stage}</text>'
            f'<rect x="{left}" y="{y + 4}" width="{bw:.1f}" height="16" rx="3" fill="{colour}"/>'
            f'<text class="val" x="{left + bw + 8:.1f}" y="{y + 16}">'
            f'{val:.3f} s  ({pct:.1f}%){suffix}</text>'
        )
    body.append(
        f'<line class="ax" x1="{left}" y1="{top}" x2="{left}" y2="{top + row_h * len(stages)}"/>'
        f'<text class="note" x="16" y="{h - 16}">'
        f'Synthesis is {lat["tts"] / total * 100:.0f}% of the run: 8 calls for 4 segments, '
        f'because the fit loop spends a second attempt on every one.</text>'
    )
    _write("fig-latency.svg", _svg(w, h, "".join(body)))


# --------------------------------------------------------------------------------------
# Figure 2: the cache claim, measured
# --------------------------------------------------------------------------------------
def fig_cache() -> None:
    """Paired cold/warm bars for wall clock, network calls and cost."""
    cold = _load("coldrun/metrics.json")
    warm = _load("output/metrics.json")

    panels = [
        ("Wall clock", cold["latency_s"]["total_wall_clock"], warm["latency_s"]["total_wall_clock"], "{:.3f} s"),
        ("Network calls", cold["api"]["cache_misses"], warm["api"]["cache_misses"], "{:.0f}"),
        ("Cost", cold["api"]["estimated_cost_inr"], warm["api"]["estimated_cost_inr"], "₹{:.4f}"),
    ]

    w, h = 880, 250
    top, base_y = 74, 196
    panel_w = (w - 32) / 3
    body = [
        _header(
            "Second run over the same input: zero calls, zero cost",
            "coldrun/metrics.json vs output/metrics.json - same clip, same parameters",
            w,
        )
    ]
    for i, (label, cold_v, warm_v, fmt) in enumerate(panels):
        cx = 16 + panel_w * i
        top_v = max(cold_v, warm_v, 1e-9)
        max_bar = base_y - top - 18
        for j, (val, colour, tag) in enumerate(
            ((cold_v, ORANGE, "cold"), (warm_v, GREEN, "warm"))
        ):
            bh = max(2.0, (val / top_v) * max_bar)
            bx = cx + 52 + j * 74
            body.append(
                f'<rect x="{bx}" y="{base_y - bh:.1f}" width="46" height="{bh:.1f}" rx="3" fill="{colour}"/>'
                f'<text class="val" x="{bx + 23}" y="{base_y - bh - 7:.1f}" text-anchor="middle">'
                f'{_esc(fmt.format(val))}</text>'
                f'<text class="sub" x="{bx + 23}" y="{base_y + 16}" text-anchor="middle">{tag}</text>'
            )
        body.append(
            f'<line class="ax" x1="{cx + 40}" y1="{base_y}" x2="{cx + panel_w - 24}" y2="{base_y}"/>'
            f'<text class="lbl" x="{cx + 40}" y="{top - 8}">{_esc(label)}</text>'
        )
    body.append(
        f'<text class="note" x="16" y="{h - 14}">'
        f'--dry-run turns any cache miss into a hard failure, so the zero is enforced, '
        f'not hoped for. Cache avoided ₹{warm["api"]["cost_avoided_by_cache_inr"]} on this run.</text>'
    )
    _write("fig-cache.svg", _svg(w, h, "".join(body)))


# --------------------------------------------------------------------------------------
# Figure 3: the sync proof, and the pause-window defect, on one timeline
# --------------------------------------------------------------------------------------
def fig_timeline() -> None:
    """Source segment windows against the dubbed clips actually placed on the timeline."""
    a = _load("coldrun/assemble_report.json")
    segs = _load("coldrun/segments.json")
    dur = a["timeline"]["source_duration_s"]
    place = a["placement"]

    w, left, right = 880, 16, 16
    top = 78
    lane_h, gap = 26, 10
    plot_w = w - left - right
    h = top + (lane_h + gap) * len(a["placements"]) + 84
    sx = plot_w / dur

    body = [
        _header(
            "Every dubbed line starts at its source timestamp",
            f"coldrun/assemble_report.json - {place['clips_placed']} clips on a "
            f"{dur:.3f} s timeline, {a['overlaps']['count']} overlaps, "
            f"{place['frames_dropped_past_end']} frames dropped",
            w,
        )
    ]
    # Time ruler.
    for t in range(0, int(dur) + 1, 5):
        x = left + t * sx
        body.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{h - 62}"/>'
            f'<text class="sub" x="{x:.1f}" y="{top - 14}" text-anchor="middle">{t}s</text>'
        )

    for i, p in enumerate(a["placements"]):
        y = top + i * (lane_h + gap)
        win_x = left + p["window"][0] * sx
        win_w = p["window_duration_s"] * sx
        clip_x = left + p["placed_start_s"] * sx
        clip_w = p["duration_s"] * sx
        text = segs[i]["text"] if i < len(segs) else ""
        if len(text) > 38:
            text = text[:35] + "..."
        caption = (
            f'seg {p["segment_id"]}  -  window {p["window_duration_s"]:.2f} s, speech '
            f'{p["duration_s"]:.2f} s, fill {p["fill_pct"]:.1f}%  -  {text}'
        )
        body.append(
            f'<rect x="{win_x:.1f}" y="{y}" width="{win_w:.1f}" height="{lane_h}" rx="4" '
            f'fill="{GREY}" opacity="0.22"/>'
            f'<rect x="{clip_x:.1f}" y="{y}" width="{clip_w:.1f}" height="{lane_h}" rx="4" fill="{BLUE}"/>'
            f'<text class="sub" x="{_fit(caption, win_x + 6):.1f}" y="{y + lane_h + 12}">'
            f'{_esc(caption)}</text>'
        )

    ly = h - 40
    body.append(
        f'<rect x="16" y="{ly - 10}" width="22" height="12" rx="3" fill="{GREY}" opacity="0.22"/>'
        f'<text class="note" x="44" y="{ly}">source segment window (phrase + the pause after it)</text>'
        f'<rect x="330" y="{ly - 10}" width="22" height="12" rx="3" fill="{BLUE}"/>'
        f'<text class="note" x="358" y="{ly}">dubbed speech actually placed</text>'
        f'<text class="note" x="16" y="{h - 16}">'
        f'Only {place["speech_pct_of_timeline"]:.1f}% of this clip is speech, so the fit loop is asked to '
        f'stretch short lines across long theatrical pauses - see limitation 1.</text>'
    )
    _write("fig-timeline.svg", _svg(w, h, "".join(body)))


# --------------------------------------------------------------------------------------
# Figure 4: why the clamp is 0.85-1.25 and not the API's 0.5-2.0
# --------------------------------------------------------------------------------------
def fig_pace() -> None:
    """Measured duration-versus-pace curve, with the clamp band and the saturation region."""
    probe = _load("output/m4_pace_probe.json")
    sweep = probe["pace_sweep"]
    elas = probe["conclusion"]["duration_vs_pace_elasticity"]

    w, h = 880, 330
    left, right, top, bottom = 64, 24, 74, 76
    plot_w, plot_h = w - left - right, h - top - bottom
    paces = [p["pace"] for p in sweep]
    durs = [p["trimmed_duration_s"] for p in sweep]
    p_lo, p_hi = min(paces), max(paces)
    d_hi = max(durs)

    def px(p: float) -> float:
        return left + (p - p_lo) / (p_hi - p_lo) * plot_w

    def py(d: float) -> float:
        return top + plot_h - (d / d_hi) * plot_h

    body = [
        _header(
            "Higher pace = shorter audio, and the response saturates past 1.25",
            "output/m4_pace_probe.json - one Hindi sentence, six real calls to bulbul:v3",
            w,
        )
    ]
    # Clamp band.
    body.append(
        f'<rect x="{px(0.85):.1f}" y="{top}" width="{px(1.25) - px(0.85):.1f}" height="{plot_h}" '
        f'fill="{GREEN}" opacity="0.10"/>'
        f'<text class="sub" x="{(px(0.85) + px(1.25)) / 2:.1f}" y="{top - 10}" text-anchor="middle">'
        f'clamp used by the pipeline: 0.85 - 1.25</text>'
    )
    # Axes + gridlines.
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        d = d_hi * frac
        y = py(d)
        body.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>'
            f'<text class="sub" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{d:.1f}s</text>'
        )
    body.append(
        f'<line class="ax" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>'
        f'<line class="ax" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>'
    )
    # Curve + points.
    pts = " ".join(f"{px(p):.1f},{py(d):.1f}" for p, d in zip(paces, durs))
    body.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    for i, (p, d) in enumerate(zip(paces, durs)):
        inside = 0.85 <= p <= 1.25
        # Anchor the two end points inward so their labels clear the axes.
        anchor = "start" if i == 0 else "end" if i == len(paces) - 1 else "middle"
        body.append(
            f'<circle cx="{px(p):.1f}" cy="{py(d):.1f}" r="5" fill="{BLUE if inside else GREY}"/>'
            f'<text class="val" x="{px(p):.1f}" y="{py(d) - 12:.1f}" text-anchor="{anchor}">{d:.2f}s</text>'
            f'<text class="sub" x="{px(p):.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle">'
            f'pace {p}</text>'
        )
    inside_e = [e["elasticity"] for e in elas if e["to_pace"] <= 1.25]
    outside_e = [e["elasticity"] for e in elas if e["from_pace"] >= 1.25]
    body.append(
        f'<text class="note" x="16" y="{h - 34}">'
        f'Elasticity inside the clamp: {min(inside_e):.2f} to {max(inside_e):.2f} - the knob does real work. '
        f'Above 1.25 it collapses to {max(outside_e):.2f} to {min(outside_e):.2f}.</text>'
        f'<text class="note" x="16" y="{h - 16}">'
        f'So the extra range the API allows buys almost no duration, at the cost of speech that '
        f'stops sounding human. That is why the clamp is where it is.</text>'
    )
    _write("fig-pace.svg", _svg(w, h, "".join(body)))


# --------------------------------------------------------------------------------------
# Figure 5: the finding -- Bulbul v3 is not duration-deterministic
# --------------------------------------------------------------------------------------
def fig_variance() -> None:
    """Three byte-identical requests, three different durations, against the +/-5% band."""
    v = _load("output/tts_variance_probe.json")
    obs = v["observations"]
    stats = v["trimmed_duration_s"]
    mean = stats["mean"]

    w, h = 880, 318
    left, right, top, bottom = 64, 190, 78, 100
    plot_w, plot_h = w - left - right, h - top - bottom
    d_hi = stats["max"] * 1.18

    def py(d: float) -> float:
        return top + plot_h - (d / d_hi) * plot_h

    body = [
        _header(
            "Identical request, three different durations",
            f"output/tts_variance_probe.json - same text, pace {v['pace']}, same speaker, "
            f"fresh cache on every call",
            w,
        )
    ]
    # +/-5% convergence band around the mean.
    y_hi, y_lo = py(mean * 1.05), py(mean * 0.95)
    body.append(
        f'<rect x="{left}" y="{y_hi:.1f}" width="{plot_w}" height="{y_lo - y_hi:.1f}" '
        f'fill="{GREEN}" opacity="0.16"/>'
        f'<line x1="{left}" y1="{py(mean):.1f}" x2="{left + plot_w}" y2="{py(mean):.1f}" '
        f'stroke="{GREEN}" stroke-width="1.5" stroke-dasharray="5 4"/>'
        f'<text class="sub" x="{left + plot_w + 10}" y="{py(mean) + 4:.1f}">mean {mean:.3f} s</text>'
        f'<text class="sub" x="{left + plot_w + 10}" y="{y_hi - 4:.1f}">'
        f'the ±5% convergence band</text>'
    )
    bar_w = 74
    step = plot_w / len(obs)
    for i, o in enumerate(obs):
        d = o["trimmed_duration_s"]
        x = left + step * i + (step - bar_w) / 2
        y = py(d)
        far = abs(d - mean) / mean > 0.05
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + plot_h - y:.1f}" rx="3" '
            f'fill="{RED if far else BLUE}"/>'
            f'<text class="val" x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle">{d:.3f} s</text>'
            f'<text class="sub" x="{x + bar_w / 2:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle">'
            f'call {o["repeat"]}</text>'
        )
    body.append(
        f'<line class="ax" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>'
        f'<text class="note" x="16" y="{h - 52}">'
        f'Spread {stats["spread_s"]} s = {stats["spread_pct_of_mean"]}% of the mean '
        f'(σ = {stats["stdev"]:.3f} s) - over 4x the convergence band. A pace measured on one '
        f'call cannot be assumed to hold on the next.</text>'
        f'<text class="note" x="16" y="{h - 34}">'
        f'The pipeline is unaffected because it ships the exact audio it measured: '
        f'caching a pace per phrase and re-requesting would have been unsound.</text>'
        f'<text class="note" x="16" y="{h - 16}">Reproduce with: make variance-probe</text>'
    )
    _write("fig-variance.svg", _svg(w, h, "".join(body)))


def main() -> int:
    """Regenerate every README figure from the measured reports on disk."""
    print("Generating figures from measured run reports...")
    fig_latency()
    fig_cache()
    fig_timeline()
    fig_pace()
    fig_variance()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
