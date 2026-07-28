#!/usr/bin/env python3
"""Run the evaluation campaign and write results as JSON.

    python experiments/run_experiments.py --all
    python experiments/run_experiments.py --experiment scaling --seeds 20

Each experiment is repeated across seeds and reported with a 95% confidence
interval, because a single run of a stochastic simulation is an anecdote; SEF
averages 20 seeded runs and the same standard is applied here.

Experiments:

``main``        every strategy on both domains, one operating point
``threshold``   precision and recall against the similarity threshold
``threshold_transfer``
                a fixed threshold tuned on one domain, evaluated on the other,
                against a risk controller calibrating live on that other domain
``rate``        latency against offered load, where queueing appears
``scaling``     encoder cost against the number of edge routers
``convergence`` fast-path share over time, learning alone against being taught
``ablation``    GS-NDN with each of its mechanisms removed in turn
``energy``      radio and compute energy, including the SEF baseline
``encoder``     MiniLM against the lexical control
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsndn import datasets  # noqa: E402
from gsndn.embeddings import PrecomputedBackend  # noqa: E402
from gsndn.metrics import aggregate, format_table  # noqa: E402
from gsndn.runner import RunResult, ScenarioConfig, run_once  # noqa: E402
from gsndn.adversary import AdversaryConfig  # noqa: E402
from gsndn.churn import ChurnConfig  # noqa: E402
from gsndn.workload import WorkloadConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
COSTS_PATH = ROOT / "data" / "costs.json"

SEMANTIC_STRATEGIES = ("saf", "saf+es", "gs-ndn")
RISK_STRATEGIES = ("saf+es", "gs-ndn", "rc-ndn")
ALL_STRATEGIES = ("vanilla-ndn",) + SEMANTIC_STRATEGIES

REPORT_COLUMNS = (
    "isr", "precision", "recall", "f1", "false_positive_rate",
    "irt_mean_ms", "irt_p95_ms", "encoder_runs",
)


class Bench:
    """Holds the loaded catalogs and embedding bundles across many runs."""

    def __init__(self, domains: Sequence[str], model: str) -> None:
        self.model = model
        self.catalogs = {d: datasets.load(d) for d in domains}
        self.backends = {
            d: PrecomputedBackend(ROOT / "data" / "embeddings" / f"{d}.{model}.npz")
            for d in domains
        }

    def run(self, config: ScenarioConfig, seeds: Sequence[int]) -> List[RunResult]:
        return [
            run_once(
                config.with_seed(seed),
                backend=self.backends[config.domain],
                catalog=self.catalogs[config.domain],
            )
            for seed in seeds
        ]

    def metrics(self, config: ScenarioConfig, seeds: Sequence[int]) -> List[Dict[str, float]]:
        return [r.metrics for r in self.run(config, seeds)]


def base_config(domain: str, **overrides) -> ScenarioConfig:
    workload = WorkloadConfig(
        rate_per_s=overrides.pop("rate_per_s", 150.0),
        duration_ms=overrides.pop("duration_ms", 60_000.0),
        variation_rate=overrides.pop("variation_rate", 0.4),
        distractor_rate=overrides.pop("distractor_rate", 0.1),
        overlap=overrides.pop("overlap", 0.5),
    )
    churn = overrides.pop("churn", None)
    adversary = overrides.pop("adversary", None)
    extra = {}
    if churn is not None:
        extra["churn"] = churn
    if adversary is not None:
        extra["adversary"] = adversary
    return ScenarioConfig(
        domain=domain, workload=workload, costs_path=COSTS_PATH,
        threshold=overrides.pop("threshold", 0.6), **extra, **overrides,
    )


# --- experiments -------------------------------------------------------------


def exp_main(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Head-to-head at one operating point, on both domains."""
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows = {}
        for strategy in ALL_STRATEGIES + ("rc-ndn",):
            config = base_config(domain, strategy=strategy, epsilon=0.2, n_edges=8)
            rows[strategy] = aggregate(bench.metrics(config, seeds))
        out[domain] = rows
        print(f"\n--- {domain} ---")
        print(format_table(rows, REPORT_COLUMNS + ("risk_realised_error",)))
    return out


def exp_threshold(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Where the similarity threshold should sit, and whether it transfers."""
    thresholds = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for strategy in ("saf+es", "gs-ndn"):
            per_threshold = {}
            for threshold in thresholds:
                config = base_config(domain, strategy=strategy, threshold=threshold,
                                     duration_ms=30_000.0)
                per_threshold[str(threshold)] = aggregate(bench.metrics(config, seeds))
            rows[strategy] = per_threshold
        out[domain] = rows
        print(f"\n--- {domain}: threshold sweep ---")
        for strategy, per_threshold in rows.items():
            print(f"  {strategy}")
            for threshold, stats in per_threshold.items():
                print(f"    Th={threshold}: precision {stats['precision']['mean']:.3f} "
                      f"recall {stats['recall']['mean']:.3f} f1 {stats['f1']['mean']:.3f}")
    return out


#: The fixed-threshold sweep points, shared by ``threshold`` and
#: ``threshold_transfer`` so the two are reading the same operating points.
THRESHOLD_POINTS = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)

#: Budgets the risk controller is asked to hold, matched to ``exp_risk`` so the
#: two experiments' rc-ndn arms are directly comparable.
EPSILON_POINTS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def exp_threshold_transfer(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Does a threshold tuned on one domain still hold its budget on another?

    A fixed cosine threshold traces an efficiency frontier: sweep it and every
    point is a (realised error, satisfaction) pair, and any operator who can
    measure realised error on their own catalog can pick the point they want.
    That is a fair opponent, and section 3's flat precision curve is not an
    argument against it. The claim that has to be tested is narrower: the
    threshold is chosen on the catalog available at tuning time, and the network
    it runs on is not that catalog.

    So the comparison here is deliberately transitional. The frontier is built on
    domain A -- for each budget, the most permissive threshold whose realised
    error on A stays inside it -- and then that same threshold is evaluated on
    domain B, where nobody got to tune. Against it stands rc-ndn given the same
    budget and run on B from cold, calibrating on B's own traffic.

    Both arms use exactly the configuration ``exp_risk`` uses, so the rc-ndn
    numbers here and there are the same measurement.
    """
    frontier: Dict[str, object] = {}
    controlled: Dict[str, object] = {}

    for domain in bench.catalogs:
        per_threshold = {}
        for threshold in THRESHOLD_POINTS:
            config = base_config(domain, strategy="gs-ndn", threshold=threshold,
                                 n_edges=8)
            per_threshold[str(threshold)] = aggregate(bench.metrics(config, seeds))
        frontier[domain] = per_threshold

        per_epsilon = {}
        for epsilon in EPSILON_POINTS:
            config = base_config(domain, strategy="rc-ndn", epsilon=epsilon, n_edges=8)
            per_epsilon[str(epsilon)] = aggregate(bench.metrics(config, seeds))
        controlled[domain] = per_epsilon

        print(f"\n--- {domain}: fixed-threshold frontier ---")
        for threshold, stats in per_threshold.items():
            print(f"    Th={threshold:<5} realised {stats['risk_realised_error']['mean']:.4f}"
                  f"  isr {stats['isr']['mean']:.3f}")
        print(f"--- {domain}: rc-ndn, live ---")
        for epsilon, stats in per_epsilon.items():
            print(f"    eps={epsilon:<5} realised {stats['risk_realised_error']['mean']:.4f}"
                  f"  isr {stats['isr']['mean']:.3f}")

    transfer = _transfer_analysis(frontier, controlled)
    for pair, rows in transfer.items():
        print(f"\n--- transfer: {pair} ---")
        for row in rows:
            if row["tuned_threshold"] is None:
                print(f"    eps={row['epsilon']:<5} no threshold meets this budget on the tuning domain")
                continue
            print(
                f"    eps={row['epsilon']:<5} Th*={row['tuned_threshold']:<5} "
                f"transferred err {row['transferred_error']:.4f} isr {row['transferred_isr']:.3f} "
                f"[{'held' if row['transferred_within_budget'] else 'OVER'}]  |  "
                f"rc-ndn err {row['rc_error']:.4f} isr {row['rc_isr']:.3f}  "
                f"-> {row['verdict']}"
            )
    return {"frontier": frontier, "rc_ndn": controlled, "transfer": transfer}


def _tune_threshold(per_threshold: Dict[str, object], epsilon: float):
    """The most permissive threshold whose realised error stays inside budget.

    Most permissive rather than safest, because that is what an operator tuning
    on their own catalog would pick: the budget is a constraint, and satisfaction
    is what they are maximising subject to it. Ties in error are broken by
    the lower threshold for the same reason.
    """
    feasible = [
        (float(t), stats) for t, stats in per_threshold.items()
        if stats["risk_realised_error"]["mean"] <= epsilon
    ]
    if not feasible:
        return None, None
    threshold, stats = min(feasible, key=lambda pair: pair[0])
    return threshold, stats


def _transfer_analysis(
    frontier: Dict[str, object], controlled: Dict[str, object]
) -> Dict[str, List[Dict[str, object]]]:
    """Tune on one domain, evaluate on the other, and score the outcome."""
    out: Dict[str, List[Dict[str, object]]] = {}
    domains = list(frontier)
    for tune in domains:
        for evaluate in domains:
            if tune == evaluate:
                continue
            rows: List[Dict[str, object]] = []
            for epsilon in EPSILON_POINTS:
                threshold, tuned_stats = _tune_threshold(frontier[tune], epsilon)
                rc = controlled[evaluate][str(epsilon)]
                row: Dict[str, object] = {
                    "epsilon": epsilon,
                    "tuned_threshold": threshold,
                    "rc_error": rc["risk_realised_error"]["mean"],
                    "rc_error_ci95": rc["risk_realised_error"]["ci95"],
                    "rc_isr": rc["isr"]["mean"],
                    "rc_isr_ci95": rc["isr"]["ci95"],
                    "rc_within_budget": rc["risk_realised_error"]["mean"] <= epsilon,
                }
                if threshold is None:
                    row.update({
                        "verdict": "no feasible threshold on tuning domain",
                    })
                    rows.append(row)
                    continue

                evaluated = frontier[evaluate][str(threshold)]
                row.update({
                    "tuning_domain_error": tuned_stats["risk_realised_error"]["mean"],
                    "tuning_domain_isr": tuned_stats["isr"]["mean"],
                    "transferred_error": evaluated["risk_realised_error"]["mean"],
                    "transferred_error_ci95": evaluated["risk_realised_error"]["ci95"],
                    "transferred_isr": evaluated["isr"]["mean"],
                    "transferred_isr_ci95": evaluated["isr"]["ci95"],
                    "transferred_within_budget":
                        evaluated["risk_realised_error"]["mean"] <= epsilon,
                    "verdict": _verdict(
                        rc["risk_realised_error"]["mean"], rc["isr"]["mean"],
                        evaluated["risk_realised_error"]["mean"], evaluated["isr"]["mean"],
                    ),
                })
                rows.append(row)
            out[f"{tune}->{evaluate}"] = rows
    return out


def _verdict(rc_error: float, rc_isr: float, other_error: float, other_isr: float) -> str:
    """Where rc-ndn sits relative to the transferred threshold on both axes.

    Stated as dominance rather than as a single score, because the two axes
    trade against each other and collapsing them would hide which one moved.
    """
    better_error = rc_error < other_error
    better_isr = rc_isr > other_isr
    if better_error and better_isr:
        return "rc-ndn dominates"
    if not better_error and not better_isr:
        return "transferred threshold dominates"
    return "neither dominates (trade)"


def exp_rate(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Latency against offered load, on SAF's own single-router topology.

    The whole offered load lands on one access-edge router here, which is the
    setting SAF measures and the one where its bottleneck past roughly 210
    Interests/s appears.  Spread across several edge routers the same total load
    leaves each of them far from saturation, and every strategy looks alike --
    a true result, but not a measurement of the bottleneck.
    """
    rates = (30, 60, 100, 150, 200, 250, 300)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for strategy in SEMANTIC_STRATEGIES:
            per_rate = {}
            for rate in rates:
                config = base_config(
                    domain, strategy=strategy, topology="single_edge",
                    rate_per_s=float(rate), duration_ms=30_000.0,
                )
                per_rate[str(rate)] = aggregate(bench.metrics(config, seeds))
            rows[strategy] = per_rate
        out[domain] = rows
        print(f"\n--- {domain}: offered load, single edge router ---")
        for strategy, per_rate in rows.items():
            cells = "  ".join(
                f"{r}:{s['irt_p95_ms']['mean']:7.1f}" for r, s in per_rate.items()
            )
            print(f"  {strategy:<10} p95 IRT  {cells}")
    return out


def exp_scaling(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Encoder cost against network size: the claim gossip exists to support."""
    edge_counts = (1, 2, 4, 6, 8, 12, 16)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for strategy in SEMANTIC_STRATEGIES:
            per_count = {}
            for n_edges in edge_counts:
                config = base_config(domain, strategy=strategy, n_edges=n_edges,
                                     duration_ms=30_000.0)
                per_count[str(n_edges)] = aggregate(bench.metrics(config, seeds))
            rows[strategy] = per_count
        out[domain] = rows
        print(f"\n--- {domain}: edge routers ---")
        for strategy, per_count in rows.items():
            cells = "  ".join(
                f"{n}:{s['encoder_runs']['mean']:6.0f}" for n, s in per_count.items()
            )
            print(f"  {strategy:<10} encoder runs  {cells}")
    return out


def _thin(curve: List[tuple], points: int = 400) -> List[tuple]:
    """Keep at most ``points`` evenly spaced samples of a convergence curve."""
    if len(curve) <= points:
        return curve
    step = len(curve) / points
    return [curve[int(i * step)] for i in range(points)] + [curve[-1]]


def exp_convergence(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Fast-path share over time: learning alone against being taught."""
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for strategy in ("saf+es", "gs-ndn-no-gossip", "gs-ndn"):
            config = base_config(domain, strategy=strategy, n_edges=8, duration_ms=60_000.0)
            results = bench.run(config, seeds[:3])
            rows[strategy] = {
                # One point per request would be a few hundred thousand numbers
                # per arm, and the figure resamples onto a fixed grid anyway.
                "curves": [_thin(r.curve(window=300)) for r in results],
                "summary": aggregate([r.metrics for r in results]),
            }
        out[domain] = rows
        print(f"\n--- {domain}: convergence ---")
        for strategy, data in rows.items():
            final = [c[-1][1] for c in data["curves"] if c]
            mean_final = sum(final) / len(final) if final else 0.0
            print(f"  {strategy:<20} final fast-path share {mean_final:.3f}")
    return out


def exp_ablation(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Remove one mechanism at a time and see what each was worth.

    Two arms vary the gossip configuration rather than the strategy: disabling
    rumour pushes leaves periodic anti-entropy as the only path, and widening the
    gossip period slows it down. Both isolate *how fast* sharing happens from
    *whether* it happens.
    """
    arms: Dict[str, Dict[str, object]] = {
        "gs-ndn": {"strategy": "gs-ndn"},
        "gs-ndn-no-gossip": {"strategy": "gs-ndn-no-gossip"},
        "gs-ndn-no-verify": {"strategy": "gs-ndn-no-verify"},
        "gs-ndn-anti-entropy-only": {"strategy": "gs-ndn", "gossip_rumour_push": False},
        "gs-ndn-slow-gossip": {"strategy": "gs-ndn", "gossip_interval_ms": 5000.0},
    }
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows = {}
        for label, overrides in arms.items():
            config = base_config(domain, n_edges=8, **overrides)
            rows[label] = aggregate(bench.metrics(config, seeds))
        out[domain] = rows
        print(f"\n--- {domain}: ablation ---")
        print(format_table(
            rows,
            ("isr", "precision", "encoder_runs", "gossip_bytes", "gossip_coverage"),
        ))
    return out


def exp_energy(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Radio against compute energy, with SEF on the mesh it was designed for."""
    out: Dict[str, object] = {}
    columns = (
        "energy_total_j", "energy_radio_j", "energy_compute_j",
        "energy_compute_share", "energy_per_interest_mj", "isr",
    )
    for domain in bench.catalogs:
        rows = {}
        for strategy in ALL_STRATEGIES:
            config = base_config(domain, strategy=strategy, n_edges=8, duration_ms=30_000.0)
            rows[strategy] = aggregate(bench.metrics(config, seeds))
        for strategy in ("sef", "gs-ndn"):
            config = base_config(domain, strategy=strategy, topology="grid",
                                 grid_side=4, duration_ms=30_000.0)
            rows[f"{strategy} (grid)"] = aggregate(bench.metrics(config, seeds))
        out[domain] = rows
        print(f"\n--- {domain}: energy ---")
        print(format_table(rows, columns))
    return out


def exp_encoder(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Does a transformer earn its cost against a character n-gram control?"""
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows = {}
        for model in ("all-MiniLM-L6-v2-onnx", "lexical-char35-512"):
            bundle = ROOT / "data" / "embeddings" / f"{domain}.{model}.npz"
            if not bundle.exists():
                print(f"  [skip] no bundle for {model}; run export_embeddings.py --backend lexical")
                continue
            backend = PrecomputedBackend(bundle)
            for threshold in (0.5, 0.6, 0.7):
                config = base_config(domain, strategy="gs-ndn", threshold=threshold,
                                     duration_ms=30_000.0)
                config = replace(config, embedding_model=model)
                metrics = [
                    run_once(config.with_seed(s), backend=backend,
                             catalog=bench.catalogs[domain]).metrics
                    for s in seeds
                ]
                rows[f"{model} Th={threshold}"] = aggregate(metrics)
        out[domain] = rows
        print(f"\n--- {domain}: encoder comparison ---")
        print(format_table(rows, ("isr", "precision", "recall", "f1")))
    return out


def exp_risk(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Does the error budget hold, and what does tightening it cost?

    The realised error is measured out of sample, over exactly the decisions the
    budget governs -- semantic resolutions, whether fresh, cached or learned from
    a neighbour. Exact FIB matches are excluded: they were never a judgement
    call and including them would dilute any rate towards zero.
    """
    epsilons = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for strategy in ("rc-ndn", "rc-ndn-no-explore", "rc-ndn-no-evidence"):
            per_epsilon = {}
            for epsilon in epsilons:
                config = base_config(
                    domain, strategy=strategy, epsilon=epsilon, n_edges=8,
                )
                per_epsilon[str(epsilon)] = aggregate(bench.metrics(config, seeds))
            rows[strategy] = per_epsilon
        out[domain] = rows
        print(f"\n--- {domain}: error budget ---")
        for strategy, per_epsilon in rows.items():
            print(f"  {strategy}")
            for epsilon, stats in per_epsilon.items():
                held = "ok " if stats["risk_realised_error"]["mean"] <= float(epsilon) else "OVER"
                print(f"    eps={epsilon:<5} realised {stats['risk_realised_error']['mean']:.4f} "
                      f"[{held}]  isr {stats['isr']['mean']:.3f}  "
                      f"coverage {stats['risk_coverage']['mean']:.3f}")
    return out


CHURN_STRATEGIES = ("saf+es", "gs-ndn", "gs-ndn-no-verify", "rc-ndn", "rc-ndn-aci")


def exp_churn(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Producers that move, and producers that quietly narrow what they answer.

    Two arms, because the first one on its own turned out to answer a different
    question than it was asked.

    ``move``   the original: departures and relocations. A relocation changes the
               right answer without changing any similarity score, which is why
               it was expected to be where feedback earns its cost. It is not,
               and :mod:`gsndn.churn` now says why -- withdrawing the route also
               invalidates every Embedding Store entry that pointed at it, so
               the stale mapping is destroyed by the event rather than caught by
               anybody, uniformly for every strategy.

    ``drift``  schema drift: no route event at all, no invalidation, no score
               change. A producer simply stops recognising some of the wordings
               it used to answer, and a producer's live refusal is the only thing
               in the system that can notice. This is the arm the original was
               missing.

    What drift should move is wasted work rather than satisfaction. A wording its
    producer has stopped answering cannot be satisfied by anyone, so no strategy
    recovers it; what separates them is how long they keep spending a round trip
    to be told no. ``producer_refusal_rate`` is that quantity.
    """
    intervals = (0.0, 10.0, 5.0, 2.0, 1.0)
    arms = {
        "move": lambda interval: ChurnConfig(interval_s=interval),
        "drift": lambda interval: ChurnConfig(interval_s=interval, drift_share=1.0),
    }
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        per_arm: Dict[str, object] = {}
        for arm, make_churn in arms.items():
            rows: Dict[str, object] = {}
            for strategy in CHURN_STRATEGIES:
                per_interval = {}
                for interval in intervals:
                    config = base_config(
                        domain, strategy=strategy, epsilon=0.2, n_edges=8,
                        churn=make_churn(interval),
                    )
                    per_interval[str(interval)] = aggregate(bench.metrics(config, seeds))
                rows[strategy] = per_interval
            per_arm[arm] = rows
            print(f"\n--- {domain}: producer churn ({arm}) ---")
            for strategy, per_interval in rows.items():
                isr = "  ".join(
                    f"{i}s:{s['isr']['mean']:.3f}" for i, s in per_interval.items()
                )
                print(f"  {strategy:<18} ISR      {isr}")
            for strategy, per_interval in rows.items():
                refusals = "  ".join(
                    f"{i}s:{s['producer_refusal_rate']['mean']:.3f}"
                    for i, s in per_interval.items()
                )
                print(f"  {strategy:<18} refused  {refusals}")
        out[domain] = per_arm
    return out


def exp_poisoning(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """A share of routers lies, in both currencies gossip carries."""
    shares = (0.0, 0.125, 0.25, 0.5)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for strategy in ("gs-ndn", "gs-ndn-no-verify", "rc-ndn"):
            per_share = {}
            for share in shares:
                config = base_config(
                    domain, strategy=strategy, epsilon=0.2, n_edges=8,
                    adversary=AdversaryConfig(compromised_share=share),
                )
                per_share[str(share)] = aggregate(bench.metrics(config, seeds))
            rows[strategy] = per_share
        out[domain] = rows
        print(f"\n--- {domain}: compromised routers ---")
        for strategy, per_share in rows.items():
            cells = "  ".join(
                f"{k}:{v['isr']['mean']:.3f}/{v['risk_realised_error']['mean']:.3f}"
                for k, v in per_share.items()
            )
            print(f"  {strategy:<18} ISR/err  {cells}")
    return out


def exp_heterogeneity(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Routers running different encoders in one network.

    Mappings pool across the boundary because a route that served a name serves
    it however anyone chose to ask. Scores do not: 0.62 from MiniLM and 0.62
    from a character n-gram model are not the same measurement.

    That argument was previously asserted and not tested: the experiment ran only
    the arm that refuses to pool, so "graceful degradation" had nothing to be
    graceful *against*. ``rc-ndn-naive-mix`` is the missing baseline -- the same
    controller with the guard removed, gossiping scores across the encoder
    boundary as though they were commensurable.

    It has to be a risk-controlled arm. GS-NDN gossips mappings and no scores at
    all, so there is nothing for a naive version of it to mix; the guard exists
    on the evidence path, and only the evidence path can be measured with it off.
    """
    shares = (0.0, 0.25, 0.5)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for strategy in ("gs-ndn", "rc-ndn", "rc-ndn-naive-mix"):
            per_share = {}
            for share in shares:
                config = base_config(
                    domain, strategy=strategy, epsilon=0.2, n_edges=8,
                    alt_embedding_model="lexical-char35-512",
                    heterogeneous_share=share,
                )
                per_share[str(share)] = aggregate(bench.metrics(config, seeds))
            rows[strategy] = per_share
        out[domain] = rows
        print(f"\n--- {domain}: encoder heterogeneity ---")
        for strategy, per_share in rows.items():
            isr = "  ".join(f"{k}:{v['isr']['mean']:.3f}" for k, v in per_share.items())
            print(f"  {strategy:<18} ISR       {isr}")
        for strategy, per_share in rows.items():
            err = "  ".join(
                f"{k}:{v['risk_realised_error']['mean']:.4f}" for k, v in per_share.items()
            )
            print(f"  {strategy:<18} realised  {err}")
        for strategy, per_share in rows.items():
            evidence = "  ".join(
                f"{k}:{v['gossip_evidence_dropped_encoder']['mean']:.0f}"
                f"/{v['gossip_evidence_mixed_encoder']['mean']:.0f}"
                for k, v in per_share.items()
            )
            print(f"  {strategy:<18} dropped/mixed  {evidence}")
    return out


EXPERIMENTS: Dict[str, Callable[[Bench, Sequence[int]], Dict[str, object]]] = {
    "risk": exp_risk,
    "churn": exp_churn,
    "poisoning": exp_poisoning,
    "heterogeneity": exp_heterogeneity,
    "main": exp_main,
    "threshold": exp_threshold,
    "threshold_transfer": exp_threshold_transfer,
    "rate": exp_rate,
    "scaling": exp_scaling,
    "convergence": exp_convergence,
    "ablation": exp_ablation,
    "energy": exp_energy,
    "encoder": exp_encoder,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", choices=sorted(EXPERIMENTS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--domains", nargs="+", default=list(datasets.DOMAINS))
    parser.add_argument("--model", default="all-MiniLM-L6-v2-onnx")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    chosen = sorted(EXPERIMENTS) if args.all else (args.experiment or ["main"])
    seeds = list(range(1, args.seeds + 1))

    if not COSTS_PATH.exists():
        print(f"[!] {COSTS_PATH} missing -- latencies will be assumed, not measured.")
        print("    Run: python experiments/bench_micro.py")

    bench = Bench(args.domains, args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    for name in chosen:
        print(f"\n{'=' * 72}\n{name.upper()}  ({len(seeds)} seeds)\n{'=' * 72}")
        started = time.perf_counter()
        payload = EXPERIMENTS[name](bench, seeds)
        elapsed = time.perf_counter() - started
        destination = args.out / f"{name}.json"
        destination.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\n[+] {name} finished in {elapsed:.1f}s -> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
