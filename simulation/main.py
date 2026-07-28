"""
═══════════════════════════════════════════════════════════════════════
  Semantic NDN Routing Simulation — Main Entry Point
  ───────────────────────────────────────────────────
  Runs all three strategies over the same workload and generates
  publication-quality comparison plots.

  Usage:
      python main.py
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Dict

import config
from simulation import (
    EmbeddingEngine,
    EmbeddingStore,
    NDNSimulator,
    RequestResult,
    generate_workload,
)


# ═══════════════════════════════════════════════════════════════════
#  METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(results: List[RequestResult]) -> Dict:
    """Compute aggregate performance metrics from simulation results."""
    total = len(results)
    resolved = [r for r in results if r.resolved]

    pdr = len(resolved) / total if total > 0 else 0.0
    avg_irt = float(np.mean([r.latency_ms for r in resolved])) if resolved else 0.0

    exact_count = sum(1 for r in results if r.match_type == "exact")
    semantic_count = sum(1 for r in results if r.match_type == "semantic")
    es_count = sum(1 for r in results if r.match_type == "es_cache")
    nack_count = sum(1 for r in results if r.match_type == "nack")

    return {
        "pdr": pdr,
        "avg_irt_ms": avg_irt,
        "total": total,
        "resolved": len(resolved),
        "exact": exact_count,
        "semantic": semantic_count,
        "es_cache": es_count,
        "nack": nack_count,
    }


def compute_rolling_latency(
    results: List[RequestResult], window: int = 50
) -> List[float]:
    """
    Compute a rolling average of per-request latency.
    NACKed requests are treated as NaN (excluded from the window).
    """
    latencies = np.array([
        r.latency_ms if r.resolved else np.nan for r in results
    ])
    rolling = []
    for i in range(len(latencies)):
        start = max(0, i - window + 1)
        window_vals = latencies[start:i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        rolling.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
    return rolling


# ═══════════════════════════════════════════════════════════════════
#  CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════

def print_report(m_vanilla: Dict, m_semantic: Dict, m_ours: Dict,
                 es: EmbeddingStore) -> None:
    """Print a formatted results summary to the console."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  SEMANTIC NDN ROUTING — SIMULATION RESULTS".center(68) + "║")
    print("╠" + "═" * 68 + "╣")

    header = f"{'Metric':<30} {'Vanilla':>10} {'Pure Sem.':>10} {'Ours (ES)':>10}"
    print("║ " + header + " ║")
    print("╟" + "─" * 68 + "╢")

    rows = [
        ("PDR",
         f"{m_vanilla['pdr']:.1%}",
         f"{m_semantic['pdr']:.1%}",
         f"{m_ours['pdr']:.1%}"),
        ("Avg IRT (ms)",
         f"{m_vanilla['avg_irt_ms']:.2f}",
         f"{m_semantic['avg_irt_ms']:.2f}",
         f"{m_ours['avg_irt_ms']:.2f}"),
        ("Exact Matches",
         str(m_vanilla['exact']),
         str(m_semantic['exact']),
         str(m_ours['exact'])),
        ("Semantic Lookups",
         str(m_vanilla['semantic']),
         str(m_semantic['semantic']),
         str(m_ours['semantic'])),
        ("ES Cache Hits",
         str(m_vanilla['es_cache']),
         str(m_semantic['es_cache']),
         str(m_ours['es_cache'])),
        ("NACKs (Unresolved)",
         str(m_vanilla['nack']),
         str(m_semantic['nack']),
         str(m_ours['nack'])),
    ]

    for label, v, s, o in rows:
        line = f"{label:<30} {v:>10} {s:>10} {o:>10}"
        print("║ " + line + " ║")

    print("╟" + "─" * 68 + "╢")
    print("║ " + f"{'ES Cache Entries Learned:':<30} {len(es.cache):>10}"
          + " " * 21 + " ║")
    print("║ " + f"{'ES Final Hit Ratio:':<30} {es.hit_ratio:>10.1%}"
          + " " * 21 + " ║")
    print("╚" + "═" * 68 + "╝")


# ═══════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════

def generate_plots(
    results_vanilla: List[RequestResult],
    results_semantic: List[RequestResult],
    results_ours: List[RequestResult],
    es: EmbeddingStore,
    m_vanilla: Dict,
    m_semantic: Dict,
    m_ours: Dict,
) -> str:
    """
    Generate a 2×2 publication-quality comparison figure.

    Plots:
      [0,0] PDR Comparison (bar)
      [0,1] Average IRT Comparison (bar)
      [1,0] Latency Convergence Over Time (line, rolling avg)
      [1,1] Embedding Store Cache Hit Ratio Evolution (line)
    """

    # ── Dark theme styling ──────────────────────────────────────
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlepad": 12,
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.alpha": 0.8,
        "font.family": "sans-serif",
        "font.size": 11,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
    })

    colors = {
        "vanilla": "#f85149",    # Red
        "semantic": "#f0883e",   # Orange
        "ours": "#3fb950",       # Green
        "accent": "#58a6ff",     # Blue accent
    }
    bar_colors = [colors["vanilla"], colors["semantic"], colors["ours"]]
    labels = ["Vanilla NDN", "Pure Semantic", "Semantic NDN\n+ ES (Ours)"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Semantic Routing in NDN Networks\n"
        "Smart Hospital IoT — Performance Comparison",
        fontsize=16, fontweight="bold", color="#f0f6fc", y=0.98,
    )

    # ── [0,0] PDR Comparison ────────────────────────────────────
    ax = axes[0, 0]
    pdrs = [m_vanilla["pdr"] * 100,
            m_semantic["pdr"] * 100,
            m_ours["pdr"] * 100]
    bars = ax.bar(labels, pdrs, color=bar_colors,
                  edgecolor="white", linewidth=0.5, width=0.55)
    ax.set_ylabel("Packet Delivery Ratio (%)")
    ax.set_title("① Packet Delivery Ratio", fontweight="bold",
                 color="#f0f6fc")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, pdrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=12, color="#f0f6fc")

    # ── [0,1] Average IRT Comparison ───────────────────────────
    ax = axes[0, 1]
    irts = [m_vanilla["avg_irt_ms"],
            m_semantic["avg_irt_ms"],
            m_ours["avg_irt_ms"]]
    bars = ax.bar(labels, irts, color=bar_colors,
                  edgecolor="white", linewidth=0.5, width=0.55)
    ax.set_ylabel("Average Interest Resolution Time (ms)")
    ax.set_title("② Average Latency (IRT)", fontweight="bold",
                 color="#f0f6fc")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, irts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.2f} ms", ha="center", va="bottom",
                fontweight="bold", fontsize=11, color="#f0f6fc")

    # ── [1,0] Latency Convergence Over Time ────────────────────
    ax = axes[1, 0]
    x = np.arange(len(results_vanilla))

    roll_vanilla = compute_rolling_latency(results_vanilla, window=50)
    roll_semantic = compute_rolling_latency(results_semantic, window=50)
    roll_ours = compute_rolling_latency(results_ours, window=50)

    ax.plot(x, roll_semantic, color=colors["semantic"],
            alpha=0.9, linewidth=1.8, label="Pure Semantic")
    ax.plot(x, roll_ours, color=colors["ours"],
            alpha=0.9, linewidth=1.8, label="Semantic + ES (Ours)")
    ax.plot(x, roll_vanilla, color=colors["vanilla"],
            alpha=0.9, linewidth=1.8, label="Vanilla NDN")

    # Add annotation for convergence point
    ax.axhline(y=config.EXACT_MATCH_DELAY, color=colors["accent"],
               linestyle=":", alpha=0.4,
               label=f"Exact-Match Floor ({config.EXACT_MATCH_DELAY:.1f} ms)")

    ax.set_xlabel("Request Number")
    ax.set_ylabel("Rolling Avg Latency (ms, window=50)")
    ax.set_title("③ Latency Convergence Over Time", fontweight="bold",
                 color="#f0f6fc")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # ── [1,1] ES Cache Hit Ratio Evolution ─────────────────────
    ax = axes[1, 1]
    hit_history = es.hit_ratio_history

    if hit_history:
        hit_pct = [h * 100 for h in hit_history]
        es_x = np.arange(len(hit_pct))

        ax.plot(es_x, hit_pct, color=colors["ours"], linewidth=2.0)
        ax.fill_between(es_x, hit_pct, alpha=0.12, color=colors["ours"])

        final = hit_pct[-1]
        ax.axhline(y=final, color=colors["accent"], linestyle="--",
                   alpha=0.5, label=f"Final: {final:.1f}%")

        # Mark the point where cache is fully warmed
        if len(hit_pct) > 10:
            # Find the first point where hit ratio exceeds 90%
            for i, h in enumerate(hit_pct):
                if h >= 90.0:
                    ax.axvline(x=i, color="#f0883e", linestyle=":",
                               alpha=0.5, label=f"90% reached at query #{i}")
                    break

    ax.set_xlabel("Semantic Query Number (ES lookups only)")
    ax.set_ylabel("Cumulative Cache Hit Ratio (%)")
    ax.set_title("④ Embedding Store Cache Convergence",
                 fontweight="bold", color="#f0f6fc")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # ── Save ────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ndn_semantic_routing_results.png",
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"\n[✓] Plot saved → {output_path}")
    plt.show()
    return output_path


# ═══════════════════════════════════════════════════════════════════
#  MATCH-TYPE BREAKDOWN PLOT (BONUS)
# ═══════════════════════════════════════════════════════════════════

def generate_breakdown_plot(
    m_vanilla: Dict, m_semantic: Dict, m_ours: Dict,
) -> str:
    """Generate a stacked bar chart showing match-type breakdown."""

    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "font.family": "sans-serif",
        "font.size": 12,
    })

    strategies = ["Vanilla NDN", "Pure Semantic", "Semantic NDN\n+ ES (Ours)"]
    metrics = [m_vanilla, m_semantic, m_ours]

    exact_vals = [m["exact"] for m in metrics]
    semantic_vals = [m["semantic"] for m in metrics]
    es_vals = [m["es_cache"] for m in metrics]
    nack_vals = [m["nack"] for m in metrics]

    x = np.arange(len(strategies))
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Request Resolution Breakdown by Match Type",
                 fontsize=14, fontweight="bold", color="#f0f6fc")

    b1 = ax.bar(x, exact_vals, width, label="Exact Match",
                color="#3fb950", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x, es_vals, width, bottom=exact_vals,
                label="ES Cache Hit", color="#58a6ff",
                edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x, semantic_vals, width,
                bottom=[e + c for e, c in zip(exact_vals, es_vals)],
                label="Full Semantic", color="#f0883e",
                edgecolor="white", linewidth=0.5)
    b4 = ax.bar(x, nack_vals, width,
                bottom=[e + c + s for e, c, s in
                        zip(exact_vals, es_vals, semantic_vals)],
                label="NACK (Drop)", color="#f85149",
                edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Number of Requests")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ndn_match_breakdown.png",
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"[✓] Breakdown plot saved → {output_path}")
    plt.show()
    return output_path


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    print("═" * 60)
    print("  Semantic Routing in NDN Networks — Simulation")
    print("  Smart Hospital IoT Scenario")
    print("═" * 60)
    print()

    # ── 1. Initialize Embedding Engine ─────────────────────────
    engine = EmbeddingEngine()

    # Pre-compute embeddings for ALL names (canonical + variants)
    all_names = list(config.PRODUCER_NAMES) + list(config.SEMANTIC_VARIANTS.keys())
    engine.precompute_embeddings(all_names)

    # Verify that all semantic pairs meet the threshold
    engine.verify_semantic_pairs()

    # ── 2. Generate Workload ───────────────────────────────────
    rng = np.random.default_rng(config.RANDOM_SEED)
    workload = generate_workload(rng)

    n_exact = sum(1 for r in workload if r in set(config.PRODUCER_NAMES))
    n_variant = sum(1 for r in workload if r in config.SEMANTIC_VARIANTS)
    n_unique_variants = len(set(r for r in workload if r in config.SEMANTIC_VARIANTS))

    print(f"[*] Workload generated: {len(workload)} Interest packets")
    print(f"    ├─ Exact canonical names:    {n_exact} "
          f"({n_exact/len(workload):.1%})")
    print(f"    ├─ Semantic variants:        {n_variant} "
          f"({n_variant/len(workload):.1%})")
    print(f"    └─ Unique variant names:     {n_unique_variants}")
    print()

    # ── 3. Run Simulations ─────────────────────────────────────
    sim = NDNSimulator(engine)

    results_vanilla = sim.run_vanilla_ndn(workload)
    results_semantic = sim.run_pure_semantic(workload)
    results_ours, es = sim.run_semantic_with_es(workload)

    # ── 4. Compute Metrics ─────────────────────────────────────
    m_vanilla = compute_metrics(results_vanilla)
    m_semantic = compute_metrics(results_semantic)
    m_ours = compute_metrics(results_ours)

    # ── 5. Print Report ────────────────────────────────────────
    print_report(m_vanilla, m_semantic, m_ours, es)

    elapsed = time.time() - start_time
    print(f"\n[✓] Simulation completed in {elapsed:.1f}s")

    # ── 6. Generate Plots ──────────────────────────────────────
    print("\n[*] Generating comparison plots...\n")
    generate_plots(
        results_vanilla, results_semantic, results_ours,
        es, m_vanilla, m_semantic, m_ours,
    )
    generate_breakdown_plot(m_vanilla, m_semantic, m_ours)

    print("\n[✓] All done! Open the PNG files to view the results.")


if __name__ == "__main__":
    main()
