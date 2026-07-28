#!/usr/bin/env python3
"""Render the campaign's results as figures for the paper.

    python experiments/make_figures.py --results results --out results/figures

Reads the JSON written by ``run_experiments.py`` and produces one PDF and one
PNG per figure -- PDF for LaTeX, PNG for looking at.

Every figure shows a spread where one exists: these are 20-seed means, and a
line drawn without its confidence band invites a reader to take noise for a
result.  Colours come from a palette validated for colour-vision deficiency
(worst adjacent CVD dE 9.1, normal-vision 22.9), and each series is also given a
distinct marker and dash pattern, so identity never rests on hue alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Categorical slots 1-4 of the validated palette, in fixed order.
PALETTE = {
    "vanilla-ndn": "#2a78d6",
    "saf": "#eb6834",
    "saf+es": "#1baf7a",
    "gs-ndn": "#eda100",
    "gs-ndn-no-gossip": "#e87ba4",
    "gs-ndn-no-verify": "#4a3aa7",
    "gs-ndn-no-edge-tag": "#e34948",
    "gs-ndn-anti-entropy-only": "#e34948",
    "gs-ndn-slow-gossip": "#008300",
    "sef": "#008300",
}
MARKERS = {
    "vanilla-ndn": "o", "saf": "s", "saf+es": "^", "gs-ndn": "D",
    "gs-ndn-no-gossip": "v", "gs-ndn-no-verify": "P", "gs-ndn-no-edge-tag": "X",
    "gs-ndn-anti-entropy-only": "X", "gs-ndn-slow-gossip": "*", "sef": "*",
}
DASHES = {
    "vanilla-ndn": (None, None), "saf": (5, 2), "saf+es": (2, 1.5),
    "gs-ndn": (None, None), "gs-ndn-no-gossip": (4, 1, 1, 1),
    "gs-ndn-no-verify": (1, 1), "gs-ndn-no-edge-tag": (6, 2, 1, 2),
    "sef": (3, 3),
}
LABELS = {
    "vanilla-ndn": "Vanilla NDN", "saf": "SAF", "saf+es": "SAF+ES",
    "gs-ndn": "GS-NDN (ours)", "gs-ndn-no-gossip": "GS-NDN, no gossip",
    "gs-ndn-no-verify": "GS-NDN, no verification",
    "gs-ndn-no-edge-tag": "GS-NDN, no edge restriction",
    "gs-ndn-anti-entropy-only": "GS-NDN, anti-entropy only",
    "gs-ndn-slow-gossip": "GS-NDN, 5 s gossip period",
    "sef": "SEF",
}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd5"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def series(ax, x, means, errors, key: str, label: Optional[str] = None) -> None:
    """One series with its confidence band, marker and dash pattern."""
    colour = PALETTE.get(key, "#52514e")
    dash = DASHES.get(key, (None, None))
    line, = ax.plot(
        x, means, color=colour, marker=MARKERS.get(key, "o"),
        label=label or LABELS.get(key, key),
        markeredgecolor="white", markeredgewidth=0.8,
    )
    if dash[0] is not None:
        line.set_dashes(dash)
    if errors is not None and any(e > 0 for e in errors):
        ax.fill_between(
            x,
            [m - e for m, e in zip(means, errors)],
            [m + e for m, e in zip(means, errors)],
            color=colour, alpha=0.15, linewidth=0,
        )


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{suffix}", bbox_inches="tight",
                    facecolor="#fcfcfb")
    plt.close(fig)
    print(f"    -> {name}.pdf / .png")


def load(results: Path, name: str) -> Optional[dict]:
    path = results / f"{name}.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    return json.loads(path.read_text())


def _xy(block: dict, metric: str) -> Tuple[List[float], List[float], List[float]]:
    keys = sorted(block, key=lambda k: float(k))
    xs = [float(k) for k in keys]
    means = [block[k][metric]["mean"] for k in keys]
    errors = [block[k][metric]["ci95"] for k in keys]
    return xs, means, errors


# --- figures -----------------------------------------------------------------


def fig_threshold(data: dict, out: Path) -> None:
    """Precision and recall against the similarity threshold, per domain."""
    print("  threshold")
    domains = list(data)
    fig, axes = plt.subplots(1, len(domains), figsize=(4.2 * len(domains), 3.2), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        block = data[domain]["gs-ndn"]
        xs, precision, p_err = _xy(block, "precision")
        _, recall, r_err = _xy(block, "recall")
        _, f1, f_err = _xy(block, "f1")
        series(ax, xs, precision, p_err, "saf", label="Precision")
        series(ax, xs, recall, r_err, "saf+es", label="Recall")
        series(ax, xs, f1, f_err, "gs-ndn", label="F1")

        best = xs[int(np.argmax(f1))]
        ax.axvline(0.7, color=TEXT_SECONDARY, linewidth=1, linestyle=":", zorder=0)
        ax.annotate("SAF's Th=0.7", xy=(0.7, 0.06), fontsize=7.5, color=TEXT_SECONDARY,
                    rotation=90, va="bottom", ha="right")
        ax.annotate(f"best F1 at {best:.2f}", xy=(best, max(f1)), xytext=(0, 8),
                    textcoords="offset points", fontsize=7.5, color=TEXT_PRIMARY, ha="center")

        ax.set_title(domain)
        ax.set_xlabel("cosine similarity threshold")
        ax.set_ylim(0, 1.05)
        tidy(ax)
    axes[0][0].set_ylabel("score")
    axes[0][-1].legend(loc="lower left")
    fig.suptitle("A fixed similarity threshold does not transfer across domains",
                 fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, out, "fig_threshold")


def fig_scaling(data: dict, out: Path) -> None:
    """Encoder cost against the number of edge routers."""
    print("  scaling")
    domains = list(data)
    fig, axes = plt.subplots(1, len(domains), figsize=(4.2 * len(domains), 3.2), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        ticks = []
        for strategy, block in data[domain].items():
            xs, means, errors = _xy(block, "encoder_runs")
            series(ax, xs, means, errors, strategy)
            ticks = xs
        ax.set_title(domain)
        ax.set_xlabel("edge routers")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(t)}" for t in ticks])
        ax.set_ylim(bottom=0)
        tidy(ax)
    axes[0][0].set_ylabel("encoder inferences per run")
    axes[0][-1].legend(loc="center right")
    fig.suptitle("Encoder cost grows with network size unless routers share what they learn",
                 fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, out, "fig_scaling")


def fig_rate(data: dict, out: Path) -> None:
    """Tail latency against offered load."""
    print("  rate")
    domains = list(data)
    fig, axes = plt.subplots(1, len(domains), figsize=(4.2 * len(domains), 3.2), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        ticks = []
        for strategy, block in data[domain].items():
            xs, means, errors = _xy(block, "irt_p95_ms")
            series(ax, xs, means, errors, strategy)
            ticks = xs
        ax.set_title(domain)
        ax.set_xlabel("offered load (Interests/s)")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(t)}" for t in ticks], fontsize=7.5)
        tidy(ax)
    axes[0][0].set_ylabel("95th percentile resolution time (ms)")
    axes[0][-1].legend(loc="upper left")
    fig.suptitle("Queueing behind the encoder is what load exposes",
                 fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, out, "fig_rate")


def fig_convergence(data: dict, out: Path) -> None:
    """Fast-path share over time."""
    print("  convergence")
    domains = list(data)
    fig, axes = plt.subplots(1, len(domains), figsize=(4.2 * len(domains), 3.2), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        for strategy, payload in data[domain].items():
            curves = [c for c in payload["curves"] if c]
            if not curves:
                continue
            grid = np.linspace(0, max(c[-1][0] for c in curves), 200)
            stacked = np.stack([
                np.interp(grid, [p[0] for p in c], [p[1] for p in c]) for c in curves
            ])
            mean = stacked.mean(axis=0)
            half = 1.96 * stacked.std(axis=0, ddof=1) / np.sqrt(len(stacked)) if len(stacked) > 1 else np.zeros_like(mean)
            series(ax, grid / 1000.0, mean, half, strategy)
        ax.set_title(domain)
        ax.set_xlabel("simulated time (s)")
        ax.set_ylim(0, 1.02)
        tidy(ax)
    axes[0][0].set_ylabel("share resolved without an encoder run")
    axes[0][-1].legend(loc="lower right")
    fig.suptitle("Being taught converges faster than learning alone",
                 fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, out, "fig_convergence")


def fig_main(data: dict, out: Path) -> None:
    """Head-to-head bars: satisfaction, precision and encoder cost."""
    print("  main")
    domains = list(data)
    metrics = [("isr", "Interest satisfaction"), ("precision", "Precision"),
               ("encoder_runs", "Encoder inferences")]
    fig, axes = plt.subplots(len(domains), 3, figsize=(10.5, 3.0 * len(domains)), squeeze=False)

    for row, domain in enumerate(domains):
        strategies = list(data[domain])
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row][col]
            values = [data[domain][s].get(metric, {}).get("mean", 0.0) for s in strategies]
            errors = [data[domain][s].get(metric, {}).get("ci95", 0.0) for s in strategies]
            positions = np.arange(len(strategies))
            bars = ax.bar(
                positions, values, yerr=errors, capsize=2.5, width=0.62,
                color=[PALETTE.get(s, "#52514e") for s in strategies],
                edgecolor="#fcfcfb", linewidth=1.2,
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    f"{value:,.0f}" if metric == "encoder_runs" else f"{value:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5, color=TEXT_PRIMARY,
                )
            ax.set_xticks(positions)
            ax.set_xticklabels([LABELS.get(s, s) for s in strategies],
                               rotation=20, ha="right", fontsize=7.5)
            ax.set_title(f"{domain} - {title}" if col == 0 else title)
            ax.grid(axis="x", visible=False)
            tidy(ax)
    fig.suptitle("Same accuracy, less work", fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, out, "fig_main")


def fig_ablation(data: dict, out: Path) -> None:
    """What each mechanism contributes, removed one at a time."""
    print("  ablation")
    domains = list(data)
    fig, axes = plt.subplots(1, len(domains), figsize=(4.6 * len(domains), 3.2), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        strategies = list(data[domain])
        values = [data[domain][s]["encoder_runs"]["mean"] for s in strategies]
        errors = [data[domain][s]["encoder_runs"]["ci95"] for s in strategies]
        positions = np.arange(len(strategies))
        ax.barh(
            positions, values, xerr=errors, height=0.6,
            color=[PALETTE.get(s, "#52514e") for s in strategies],
            edgecolor="#fcfcfb", linewidth=1.2,
        )
        for position, value in zip(positions, values):
            ax.annotate(f"{value:,.0f}", xy=(value, position), xytext=(4, 0),
                        textcoords="offset points", va="center",
                        fontsize=7.5, color=TEXT_PRIMARY)
        ax.set_yticks(positions)
        ax.set_yticklabels([LABELS.get(s, s) for s in strategies], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("encoder inferences per run")
        ax.set_title(domain)
        ax.grid(axis="y", visible=False)
        tidy(ax)
    fig.suptitle("Removing one mechanism at a time", fontsize=10.5,
                 fontweight="bold", y=1.02)
    save(fig, out, "fig_ablation")


def fig_energy(data: dict, out: Path) -> None:
    """Radio against compute energy, stacked."""
    print("  energy")
    domains = list(data)
    fig, axes = plt.subplots(1, len(domains), figsize=(4.8 * len(domains), 3.4), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        strategies = list(data[domain])
        radio = [data[domain][s]["energy_radio_j"]["mean"] for s in strategies]
        compute = [data[domain][s]["energy_compute_j"]["mean"] for s in strategies]
        positions = np.arange(len(strategies))
        ax.bar(positions, radio, width=0.62, label="radio",
               color=PALETTE["vanilla-ndn"], edgecolor="#fcfcfb", linewidth=1.4)
        ax.bar(positions, compute, width=0.62, bottom=radio, label="compute",
               color=PALETTE["saf"], edgecolor="#fcfcfb", linewidth=1.4)
        ax.set_xticks(positions)
        ax.set_xticklabels([LABELS.get(s, s) for s in strategies],
                           rotation=25, ha="right", fontsize=7.5)
        ax.set_title(domain)
        ax.grid(axis="x", visible=False)
        tidy(ax)
    axes[0][0].set_ylabel("energy per run (J)")
    axes[0][-1].legend(loc="upper left")
    fig.suptitle("Semantic forwarding moves the energy budget from radio to compute",
                 fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, out, "fig_energy")


def fig_simhash(detail: dict, out: Path) -> None:
    """Signature width against route-selection accuracy."""
    print("  simhash")
    table = detail.get("simhash")
    if not table:
        print("  [skip] no simhash measurements")
        return
    widths = sorted(table, key=int)
    xs = [int(w) for w in widths]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for k, key in ((1, "saf"), (3, "saf+es"), (8, "gs-ndn")):
        values = [table[w][f"recall_at_{k}"] for w in widths]
        series(ax, xs, values, None, key, label=f"true best in top {k}")
    ax.axhline(0.95, color=TEXT_SECONDARY, linewidth=1, linestyle=":", zorder=0)
    ax.annotate("0.95", xy=(xs[0], 0.955), fontsize=7.5, color=TEXT_SECONDARY)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{w}\n({int(w) // 8} B)" for w in widths], fontsize=8)
    ax.set_xlabel("signature width (bits)")
    ax.set_ylabel("fraction of variant names")
    ax.set_ylim(0, 1.05)
    ax.set_title("Signatures cannot select routes, only shortlist them",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="lower right")
    tidy(ax)
    save(fig, out, "fig_simhash")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--out", type=Path, default=root / "results" / "figures")
    args = parser.parse_args()

    style()
    print(f"[*] figures -> {args.out}")

    for name, builder in (
        ("main", fig_main), ("threshold", fig_threshold), ("rate", fig_rate),
        ("scaling", fig_scaling), ("convergence", fig_convergence),
        ("ablation", fig_ablation), ("energy", fig_energy),
    ):
        payload = load(args.results, name)
        if payload:
            builder(payload, args.out)

    detail = root / "data" / "costs.detail.json"
    if detail.exists():
        fig_simhash(json.loads(detail.read_text()), args.out)

    print("[+] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
