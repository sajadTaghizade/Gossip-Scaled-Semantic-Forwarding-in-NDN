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
``ontology``    invented synonyms against ones transcribed from published
                IoT ontologies (Brick, SAREF, Haystack, SSN/SOSA)
``churn``       producers that move, and producers that quietly narrow what
                they answer to
``drift_paired`` the drift arm's separations, paired seed by seed
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
from gsndn.embeddings import NameIndex, PrecomputedBackend  # noqa: E402
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


#: Per-worker cache of the catalogs and embedding bundles, so a pool process
#: loads each ``.npz`` once rather than once per run.
_WORKER_CACHE: Dict[str, object] = {}


def _worker_assets(domain: str, model: str):
    key = f"{domain}.{model}"
    cached = _WORKER_CACHE.get(key)
    if cached is None:
        cached = (
            PrecomputedBackend(ROOT / "data" / "embeddings" / f"{key}.npz"),
            datasets.load(domain),
        )
        _WORKER_CACHE[key] = cached
    return cached


def _run_one(payload) -> Dict[str, float]:
    """One seeded run, in whichever process picks it up.

    A module-level function taking only picklable arguments, because that is
    what a process pool can dispatch. Determinism is unaffected by which worker
    runs what: every run is a pure function of its own ``ScenarioConfig``, and
    results are reassembled in submission order rather than completion order.
    """
    config, model = payload
    backend, catalog = _worker_assets(config.domain, model)
    return run_once(config, backend=backend, catalog=catalog).metrics


class Bench:
    """Holds the loaded catalogs and embedding bundles across many runs."""

    def __init__(self, domains: Sequence[str], model: str, jobs: int = 1) -> None:
        self.model = model
        self.jobs = max(1, int(jobs))
        self.catalogs = {d: datasets.load(d) for d in domains}
        self.backends = {
            d: PrecomputedBackend(ROOT / "data" / "embeddings" / f"{d}.{model}.npz")
            for d in domains
        }
        self._pool = None

    def pool(self):
        """A lazily created process pool, reused across every experiment."""
        if self.jobs == 1:
            return None
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor

            self._pool = ProcessPoolExecutor(max_workers=self.jobs)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

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
        pool = self.pool()
        if pool is None:
            return [r.metrics for r in self.run(config, seeds)]
        # ``map`` preserves input order, so the aggregate is identical to the
        # serial one regardless of how the work was distributed.
        return list(pool.map(
            _run_one, [(config.with_seed(seed), self.model) for seed in seeds]
        ))


def base_config(domain: str, **overrides) -> ScenarioConfig:
    workload = WorkloadConfig(
        rate_per_s=overrides.pop("rate_per_s", 150.0),
        duration_ms=overrides.pop("duration_ms", 60_000.0),
        variation_rate=overrides.pop("variation_rate", 0.4),
        distractor_rate=overrides.pop("distractor_rate", 0.1),
        overlap=overrides.pop("overlap", 0.5),
        vocabulary_arrival_s=overrides.pop("vocabulary_arrival_s", 0.0),
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
        for strategy in ALL_STRATEGIES + (
            "gs-ndn-robust", "rc-ndn", "rc-ndn-robust",
            "gs-ndn-reasons", "rc-ndn-reasons",
        ):
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
    #: The same controller reading the producer's refusal reason. This arm is
    #: the one that says whether the withdrawn efficiency claim was a property
    #: of risk control or an artefact of counting false refusals as errors.
    controlled_reasons: Dict[str, object] = {}

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

        per_epsilon_reasons = {}
        for epsilon in EPSILON_POINTS:
            config = base_config(
                domain, strategy="rc-ndn-reasons", epsilon=epsilon, n_edges=8,
            )
            per_epsilon_reasons[str(epsilon)] = aggregate(bench.metrics(config, seeds))
        controlled_reasons[domain] = per_epsilon_reasons

        print(f"\n--- {domain}: fixed-threshold frontier ---")
        for threshold, stats in per_threshold.items():
            print(f"    Th={threshold:<5} realised {stats['risk_realised_error']['mean']:.4f}"
                  f"  isr {stats['isr']['mean']:.3f}")
        print(f"--- {domain}: rc-ndn, live ---")
        for epsilon, stats in per_epsilon.items():
            print(f"    eps={epsilon:<5} realised {stats['risk_realised_error']['mean']:.4f}"
                  f"  isr {stats['isr']['mean']:.3f}")

    transfer = _transfer_analysis(frontier, controlled)
    transfer_reasons = _transfer_analysis(frontier, controlled_reasons)
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
    for pair, rows in transfer_reasons.items():
        print(f"\n--- transfer (reason-aware): {pair} ---")
        for row in rows:
            if row["tuned_threshold"] is None:
                continue
            print(
                f"    eps={row['epsilon']:<5} Th*={row['tuned_threshold']:<5} "
                f"transferred err {row['transferred_error']:.4f} isr {row['transferred_isr']:.3f} "
                f"[{'held' if row['transferred_within_budget'] else 'OVER'}]  |  "
                f"rc-ndn-reasons err {row['rc_error']:.4f} isr {row['rc_isr']:.3f}  "
                f"-> {row['verdict']}"
            )

    def tally(rows_by_pair):
        counts = {}
        for rows in rows_by_pair.values():
            for row in rows:
                if row["tuned_threshold"] is None:
                    continue
                counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
        return counts

    print("\n--- verdict tally ---")
    print(f"  rc-ndn          {tally(transfer)}")
    print(f"  rc-ndn-reasons  {tally(transfer_reasons)}")

    return {
        "frontier": frontier,
        "rc_ndn": controlled,
        "rc_ndn_reasons": controlled_reasons,
        "transfer": transfer,
        "transfer_reasons": transfer_reasons,
    }


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
        # Verification restricted to locally resolved mappings, which is what
        # the code did until the defect was found. Against "gs-ndn" this is
        # exactly what extending verification to the gossip path is worth.
        "gs-ndn-unverified-import": {"strategy": "gs-ndn-unverified-import"},
        "gs-ndn-robust": {"strategy": "gs-ndn-robust"},
        "gs-ndn-anti-entropy-only": {"strategy": "gs-ndn", "gossip_rumour_push": False},
        # A tenfold longer period. Sends far fewer bytes and encodes more;
        # exp_gossip_period measures the whole frontier, this row keeps it
        # visible in the ablation at the main operating point.
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
        for strategy in ("rc-ndn", "rc-ndn-reasons",
                         "rc-ndn-no-explore", "rc-ndn-no-evidence"):
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


CHURN_STRATEGIES = (
    "saf+es", "gs-ndn", "gs-ndn-no-verify", "rc-ndn", "rc-ndn-aci",
    # Schema drift is the event that *produces* unknown-wording refusals --
    # a producer narrowing what it answers to is exactly a declaration going
    # incomplete -- so this is where reading the refusal reason should matter
    # most, and the arm has to be here for that to be measurable.
    "rc-ndn-reasons",
)


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


def exp_drift_paired(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """The drift arm's separations, tested pairwise on identical seeds.

    ``exp_churn`` reports each strategy's mean with its own confidence interval,
    and the differences under schema drift are small enough that those intervals
    overlap while the underlying difference is real: every arm sees the same
    workload, the same drift events and the same producers, so the seed-to-seed
    variation the intervals are made of is shared and cancels. This runs the
    hardest drift point only, pairs the runs by seed, and reports the
    distribution of the difference rather than the difference of the means.

    Two baselines, because two different questions are being asked.

    Against ``gs-ndn-no-verify``: does verification pay under an event only a
    producer refusal can detect? That is the question the move arm could not
    answer, and the reason this arm exists.

    Against ``rc-ndn``: does an adaptive budget beat a fixed one when the
    distribution moves under it? Adaptive Conformal Inference is motivated
    entirely by exchangeability failing, and schema drift is exchangeability
    failing, so this is the arm where it should show if it shows anywhere.
    """
    baselines = ("gs-ndn-no-verify", "rc-ndn")
    arms = ("saf+es", "gs-ndn", "gs-ndn-no-verify", "rc-ndn", "rc-ndn-aci")
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        per_strategy: Dict[str, List[Dict[str, float]]] = {}
        for strategy in arms:
            config = base_config(
                domain, strategy=strategy, epsilon=0.2, n_edges=8,
                churn=ChurnConfig(interval_s=1.0, drift_share=1.0),
            )
            per_strategy[strategy] = bench.metrics(config, seeds)

        against: Dict[str, object] = {}
        for baseline in baselines:
            rows: Dict[str, object] = {}
            for strategy in arms:
                if strategy == baseline:
                    continue
                entry: Dict[str, object] = {}
                for metric in ("isr", "producer_refusal_rate", "risk_realised_error"):
                    deltas = [
                        a[metric] - b[metric]
                        for a, b in zip(per_strategy[strategy], per_strategy[baseline])
                    ]
                    entry[metric] = _paired(deltas)
                rows[strategy] = entry
            against[baseline] = rows

            print(f"\n--- {domain}: schema drift at 1 s, paired against {baseline} ---")
            for strategy, entry in rows.items():
                isr = entry["isr"]
                refused = entry["producer_refusal_rate"]
                print(
                    f"  {strategy:<18} ISR {isr['mean']:+.4f}±{isr['ci95']:.4f} "
                    f"[{'resolved' if isr['resolved'] else 'not resolved'}]  "
                    f"refusals {refused['mean']:+.4f}±{refused['ci95']:.4f} "
                    f"[{'resolved' if refused['resolved'] else 'not resolved'}]  "
                    f"wins {isr['wins']}/{isr['n']}"
                )
        out[domain] = against
    return out


def _paired(deltas: Sequence[float]) -> Dict[str, float]:
    """Mean paired difference, its interval, and how often it had the sign."""
    import math

    n = len(deltas)
    mean = sum(deltas) / n
    if n > 1:
        variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        ci95 = 1.96 * math.sqrt(variance / n)
    else:
        ci95 = 0.0
    return {
        "mean": mean,
        "ci95": ci95,
        "n": n,
        "wins": sum(1 for d in deltas if d > 0),
        # Does the interval exclude zero -- i.e. is the sign of this difference
        # something twenty seeds actually settle?
        "resolved": abs(mean) > ci95,
    }


def exp_poisoning(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """A share of routers lies, in both currencies gossip carries."""
    shares = (0.0, 0.125, 0.25, 0.5)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for strategy in (
            "gs-ndn", "gs-ndn-unverified-import", "gs-ndn-no-verify",
            "gs-ndn-robust", "rc-ndn", "rc-ndn-robust", "rc-ndn-reasons",
        ):
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


def exp_ontology(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Are the invented synonyms easier than the ones somebody standardised?

    The catalogs are generated from a lexicon we wrote, so the recognition
    results could in principle be measuring our lexicon rather than the encoder.
    The grounded catalogs add a seventh rewording per service taken from Brick
    Schema, SAREF, Project Haystack or W3C SSN/SOSA where those vocabularies name
    the quantity at all -- see :mod:`gsndn.datasets.ontology` for the table and
    for what it does not fix.

    The headline measurement is deterministic and needs no seeds, because it is
    a property of the encoder and the names rather than of the simulation: for
    every reworded name, does the true service come out top of a cosine search
    over the FIB, and at what score. Grouping that by rewrite family puts the
    ontology-sourced wordings directly against the invented ones on the same
    catalog and the same encoder.

    ``ontology`` counts only the services a published vocabulary actually names.
    ``ontology-fallback`` is the rest of that family -- clinical metrics and bus
    timetables, which no building or sensing ontology models -- and is reported
    separately rather than averaged in, because averaging it in would let our own
    lexicon flatter a number labelled as standardised.
    """
    out: Dict[str, object] = {}
    for domain in datasets.GROUNDED_DOMAINS:
        catalog = datasets.load(domain)
        backend = PrecomputedBackend(
            ROOT / "data" / "embeddings" / f"{domain}.{bench.model}.npz"
        )
        index = NameIndex(backend, catalog.canonical_names)
        families = catalog.metadata["rewrite_family"]

        # Only some services have a standardised wording at all, and they are
        # not a random sample -- they are the environmental ones, which are also
        # the ones the invented families handle most easily. Comparing the
        # ontology family against invented families computed over *all* services
        # would therefore compare two different service sets and call the
        # difference an encoder result. Every family is scored twice: over
        # everything, and restricted to the services a published vocabulary
        # actually names. The restricted column is the comparable one.
        category = {s.canonical: s.category for s in catalog.services}
        grounded_service = {
            canonical for canonical, key in category.items()
            if datasets.ontology.terms_for(key)
        }

        def blank() -> Dict[str, float]:
            return {"n": 0, "rank1": 0, "true_score": 0.0, "best_score": 0.0}

        per_family: Dict[str, Dict[str, float]] = {}
        restricted: Dict[str, Dict[str, float]] = {}
        for interest in catalog.by_kind(datasets.VARIANT):
            family = families.get(interest.name, "unknown")
            query = backend.encode([interest.name])[0]
            scores = index.similarities(query)
            best = int(scores.argmax())
            hit = int(index.names[best] == interest.expected)
            true_score = float(scores[index.names.index(interest.expected)])

            targets = [per_family.setdefault(family, blank())]
            if interest.expected in grounded_service:
                targets.append(restricted.setdefault(family, blank()))
            for bucket in targets:
                bucket["n"] += 1
                bucket["rank1"] += hit
                bucket["true_score"] += true_score
                bucket["best_score"] += float(scores[best])

        def summarise_families(source: Dict[str, Dict[str, float]]):
            return {
                family: {
                    "n": data["n"],
                    "rank1": data["rank1"] / data["n"],
                    "mean_true_score": data["true_score"] / data["n"],
                    "mean_best_score": data["best_score"] / data["n"],
                }
                for family, data in sorted(source.items())
            }

        rows = summarise_families(per_family)
        rows_restricted = summarise_families(restricted)
        out[domain] = {
            "recognition": rows,
            "recognition_grounded_services_only": rows_restricted,
            "coverage": catalog.metadata["ontology"],
        }
        print(f"\n--- {domain}: recognition by rewrite family ---")
        print(f"  {'family':<20} {'all services':>22}   {'grounded services only':>24}")
        for family, data in rows.items():
            limited = rows_restricted.get(family)
            cell = (
                f"n={limited['n']:<4} rank-1 {limited['rank1']:.3f}"
                if limited else "--"
            )
            print(f"  {family:<20} n={data['n']:<4} rank-1 {data['rank1']:.3f}"
                  f"  score {data['mean_true_score']:.3f}   {cell}")

    # And end to end, on the grounded catalogs, at the campaign's operating
    # point. Not comparable line-for-line with section 1's table -- a grounded
    # catalog has a seventh wording per service and a larger distractor set, so
    # it is a different dataset -- but it says whether the pipeline behaves at
    # all on names it did not invent.
    grounded_bench = Bench(datasets.GROUNDED_DOMAINS, bench.model)
    end_to_end: Dict[str, object] = {}
    for domain in datasets.GROUNDED_DOMAINS:
        rows = {}
        for strategy in ("saf+es", "gs-ndn", "rc-ndn"):
            config = base_config(domain, strategy=strategy, epsilon=0.2, n_edges=8)
            rows[strategy] = aggregate(grounded_bench.metrics(config, seeds))
        end_to_end[domain] = rows
        print(f"\n--- {domain}: end to end ---")
        print(format_table(rows, ("isr", "precision", "risk_realised_error")))
    out["end_to_end"] = end_to_end
    return out


def exp_coverage(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """What an incomplete producer declaration costs.

    The feedback channel this work calibrates against is a producer deciding
    from its own declared schema, and that declaration is allowed to be
    incomplete: ``alias_coverage`` is the share of a service's known wordings
    its operator bothered to declare. Below 1.0 a producer refuses requests
    that were genuinely meant for it, so the labels the risk controller learns
    from are not merely noisy but biased against the routes that are hardest
    to word.

    Two numbers per setting. ``undeclared_share`` is a property of the
    declarations alone -- what fraction of the wordings a service can be asked
    by it did not declare -- read straight off the schemas without running
    anything. The rest is end to end, including a strategy that never consults
    feedback (``saf+es``), because the cost of a narrow declaration is a
    producer-side effect and should appear whether or not a strategy learns
    from it.
    """
    from gsndn import datasets as _datasets
    from gsndn.admission import build_schemas, _metric_term

    coverages = (1.0, 0.9, 0.7, 0.5)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        catalog = bench.catalogs[domain]
        rows: Dict[str, object] = {}
        for coverage in coverages:
            # What the declarations themselves leave out, before any traffic.
            wordings: Dict[str, set] = {}
            for interest in catalog.by_kind(_datasets.VARIANT):
                if interest.expected:
                    wordings.setdefault(interest.expected, set()).add(interest.name)
            declared_n = askable_n = 0
            for seed in seeds:
                schemas = build_schemas(
                    catalog, sorted(wordings), alias_coverage=coverage, seed=seed,
                )
                for canonical, schema in schemas.items():
                    terms = {
                        _metric_term(w, schema.instance)
                        for w in wordings.get(canonical, ())
                    }
                    terms = {t for t in terms if t is not None}
                    askable_n += len(terms)
                    declared_n += len(terms & set(schema.declared))
            undeclared = 1.0 - (declared_n / askable_n if askable_n else 1.0)

            per_strategy = {}
            for strategy in ("saf+es", "gs-ndn", "gs-ndn-reasons",
                             "rc-ndn", "rc-ndn-reasons"):
                config = base_config(
                    domain, strategy=strategy, epsilon=0.2, n_edges=8,
                    alias_coverage=coverage,
                )
                per_strategy[strategy] = aggregate(bench.metrics(config, seeds))
            rows[str(coverage)] = {
                "undeclared_share": undeclared,
                "strategies": per_strategy,
            }
        out[domain] = rows
        print(f"\n--- {domain}: declared alias coverage ---")
        for coverage, cell in rows.items():
            print(f"  coverage {coverage}  undeclared {cell['undeclared_share']:.3f}")
            for strategy, m in cell["strategies"].items():
                print(f"      {strategy:<10} ISR {m['isr']['mean']:.3f}"
                      f"  refusal {m['producer_refusal_rate']['mean']:.3f}")
    return out


#: An approximation of an unbounded sync layer, for the comparison in
#: exp_horizon. It is GS-NDN with the anti-entropy parameters opened up:
#: every router reconciles with every peer it has each round, and no cap on how
#: many entries one exchange may carry.
#:
#: This is a *bound*, not an implementation of NDN Sync. ChronoSync, PSync and
#: SVS differ from each other and from this in how state is digested, how
#: differences are detected, and how much of the dataset a single exchange
#: names; NLSR floods prefix LSAs over such a layer rather than gossiping
#: resolutions. What this arm answers is narrower and is the only question the
#: horizon result actually raises: if the sync were free of the fanout and
#: delta limits our gossip imposes, how much more of the encoder cost would it
#: remove, and what would it cost in messages to do so.
FULL_SYNC = "gs-ndn-full-sync"
FULL_SYNC_FANOUT = 64
FULL_SYNC_MAX_DELTA = 1_000_000


def horizon_config(domain: str, strategy: str, **kw):
    """Config for one horizon arm, expanding the pseudo-strategy above."""
    if strategy == FULL_SYNC:
        return base_config(
            domain, strategy="gs-ndn",
            gossip_fanout=FULL_SYNC_FANOUT, gossip_max_delta=FULL_SYNC_MAX_DELTA,
            **kw,
        )
    return base_config(domain, strategy=strategy, **kw)


def exp_horizon(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """How much of the sharing win is a warm-up cost that amortises away.

    Section 2 measures encoder inferences over a 60-second run and reports that
    a per-router cache grows 60% from 1 to 16 edge routers while gossip grows
    10%. That comparison is real but it is scoped to a horizon, and the scope
    was not stated. The reason is mechanical: a cold cache costs one inference
    per router per distinct wording, so N routers pay it N times, but they pay
    it *once*. Run for longer and that fixed cost is divided over more traffic,
    the per-router caches warm, and the gap closes.

    So the honest quantity is not one growth figure but how it moves with the
    horizon. Absolute counts are not comparable across run lengths -- a longer
    run simply carries more traffic -- so the rate per thousand requests is
    what is reported alongside them.

    This does not overturn section 2. It bounds it: sharing is worth most to a
    network that is still learning its catalog, and least to one that has been
    up long enough for every router to have seen everything.
    """
    horizons = (60_000.0, 240_000.0, 600_000.0)
    edges = (1, 4, 16)
    strategies = ("saf", "saf+es", "gs-ndn", FULL_SYNC)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for horizon in horizons:
            per_edges = {}
            for n_edges in edges:
                per_strategy = {}
                for strategy in strategies:
                    config = horizon_config(
                        domain, strategy, n_edges=n_edges, duration_ms=horizon,
                    )
                    per_strategy[strategy] = aggregate(bench.metrics(config, seeds))
                per_edges[str(n_edges)] = per_strategy
            rows[str(horizon)] = per_edges
        out[domain] = rows

        print(f"\n--- {domain}: encoder inferences by horizon ---")
        for horizon, per_edges in rows.items():
            secs = float(horizon) / 1000.0
            print(f"  {secs:.0f}s")
            for strategy in strategies:
                cells = []
                for n_edges in edges:
                    m = per_edges[str(n_edges)][strategy]
                    runs = m["encoder_runs"]["mean"]
                    per_k = runs / m["requests"]["mean"] * 1000.0
                    cells.append(f"{n_edges}:{runs:.0f} ({per_k:.1f}/k)")
                first = per_edges[str(edges[0])][strategy]["encoder_runs"]["mean"]
                last = per_edges[str(edges[-1])][strategy]["encoder_runs"]["mean"]
                growth = (last - first) / first * 100.0 if first else float("nan")
                print(f"    {strategy:<8} {'  '.join(cells)}   growth {growth:+.1f}%")
            top = per_edges[str(edges[-1])]
            es = top["saf+es"]["encoder_runs"]["mean"]
            gs = top["gs-ndn"]["encoder_runs"]["mean"]
            fs = top[FULL_SYNC]["encoder_runs"]["mean"]
            print(f"    -> at {edges[-1]} edges gossip saves {(es - gs) / es * 100:.1f}% of inferences, "
                  f"full-sync {(es - fs) / es * 100:.1f}%")
            gs_bytes = top["gs-ndn"]["gossip_bytes"]["mean"]
            fs_bytes = top[FULL_SYNC]["gossip_bytes"]["mean"]
            gs_sent = top["gs-ndn"]["gossip_mappings_sent"]["mean"]
            fs_sent = top[FULL_SYNC]["gossip_mappings_sent"]["mean"]
            ratio = fs_bytes / gs_bytes if gs_bytes else float("nan")
            print(f"       messages: gs-ndn {gs_sent:.0f} mappings / {gs_bytes / 1000:.0f} kB, "
                  f"full-sync {fs_sent:.0f} / {fs_bytes / 1000:.0f} kB  ({ratio:.1f}x bytes)")
    return out


ROBUST_PAIRS = (("gs-ndn", "gs-ndn-robust"), ("rc-ndn", "rc-ndn-robust"))

#: The pre-fix behaviour, carried through the same sweep so the two halves of
#: the recovery -- extending verification to imports, then attributing those
#: verdicts to peers -- are never reported as one number.
ROBUST_REFERENCE = "gs-ndn-unverified-import"


def exp_robust(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """Does attributing a refusal to whoever supplied the claim blunt §9?

    Section 9 measures the attack landing and names the two defences it does
    not implement. This is the second of them -- reputation -- and the arm is a
    single variable against its own baseline: ``gs-ndn-robust`` differs from
    ``gs-ndn`` only in that a retraction is charged to the peer that taught the
    mapping, a pair this router has disproved is not reinstalled from gossip,
    and no single peer may own more than a quarter of a route's calibration
    window.

    Two things are reported that a satisfaction number alone would hide. The
    share at 0.0 is the *cost* of the defence on an honest network, which is
    the number that decides whether it can be left switched on. And
    ``rep_peers_distrusted`` says whether the mechanism actually identified
    anybody, as opposed to helping for some unrelated reason -- a defence that
    improves the metric without ever distrusting a compromised peer has not
    been shown to work.
    """
    shares = (0.0, 0.125, 0.25, 0.5)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        arms = (ROBUST_REFERENCE,) + tuple(s for pair in ROBUST_PAIRS for s in pair)
        for strategy in arms:
            per_share = {}
            for share in shares:
                config = base_config(
                    domain, strategy=strategy, epsilon=0.2, n_edges=8,
                    adversary=AdversaryConfig(compromised_share=share),
                )
                per_share[str(share)] = aggregate(bench.metrics(config, seeds))
            rows[strategy] = per_share
        out[domain] = rows

        print(f"\n--- {domain}: reputation against a poisoning attacker ---")
        reference = rows[ROBUST_REFERENCE]
        print(f"  {ROBUST_REFERENCE} (verification restricted to local mappings)")
        for share in shares:
            m = reference[str(share)]
            print(
                f"    share {share:<5} ISR {m['isr']['mean']:.3f}   "
                f"err {m['risk_realised_error']['mean']:.3f}   "
                f"poison live {m['adv_poison_live']['mean']:.0f}"
            )
        for base, robust in ROBUST_PAIRS:
            print(f"  {base} -> {robust}")
            for share in shares:
                b = rows[base][str(share)]
                r = rows[robust][str(share)]
                d_isr = r["isr"]["mean"] - b["isr"]["mean"]
                d_err = (
                    r["risk_realised_error"]["mean"] - b["risk_realised_error"]["mean"]
                )
                print(
                    f"    share {share:<5} ISR {b['isr']['mean']:.3f} -> "
                    f"{r['isr']['mean']:.3f} ({d_isr:+.3f})   "
                    f"err {b['risk_realised_error']['mean']:.3f} -> "
                    f"{r['risk_realised_error']['mean']:.3f} ({d_err:+.3f})   "
                    f"poison live {b['adv_poison_live']['mean']:.0f} -> "
                    f"{r['adv_poison_live']['mean']:.0f}   "
                    f"distrusted {r['rep_peers_distrusted']['mean']:.1f}"
                )
    return out


def exp_vocabulary(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """What the sharing win is actually proportional to.

    Section 2 reports that gossip's saving falls from 26% to 7.5% as the run
    goes from 60 to 600 seconds, and treats that as a scope condition on the
    result. It is a scope condition on the *experiment*. Every run there uses a
    closed vocabulary: all 300 rewordings are askable from t=0, so once every
    router has met all of them there is nothing left for anyone to learn and
    sharing necessarily stops paying. The decay measures the catalog running
    out, not the protocol wearing off.

    A deployment's vocabulary does not run out -- new client applications,
    vendors and integrations keep introducing phrasings nobody has resolved
    yet. ``vocabulary_arrival_s`` is the rate at which that happens, and this
    experiment sweeps it against the horizon that exposed the decay.

    The prediction under test, stated before the run: the saving decays toward
    zero only when arrivals are off, and settles at a positive floor set by the
    arrival rate when they are on. If it decays to the same place regardless,
    the reframing is wrong and §2's scope condition stands as originally
    written.
    """
    # 0.0 is the closed catalog §2 measured; the rest introduce the catalog's
    # 300 rewordings over roughly 150s, 600s and 1500s respectively.
    arrivals = (0.0, 0.5, 2.0, 5.0)
    horizons = (60_000.0, 240_000.0, 600_000.0)
    n_edges = 16
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for arrival in arrivals:
            per_horizon = {}
            for horizon in horizons:
                per_strategy = {}
                for strategy in ("saf+es", "gs-ndn"):
                    config = base_config(
                        domain, strategy=strategy, n_edges=n_edges,
                        duration_ms=horizon, vocabulary_arrival_s=arrival,
                    )
                    per_strategy[strategy] = aggregate(bench.metrics(config, seeds))
                per_horizon[str(horizon)] = per_strategy
            rows[str(arrival)] = per_horizon
        out[domain] = rows

        print(f"\n--- {domain}: sharing win against vocabulary arrival rate ---")
        print(f"    (encoder inferences saved by gossip, {n_edges} edge routers)")
        header = "  ".join(f"{h/1000:>7.0f}s" for h in horizons)
        print(f"    {'arrival':<10} {header}")
        for arrival in arrivals:
            cells = []
            for horizon in horizons:
                pair = rows[str(arrival)][str(horizon)]
                es = pair["saf+es"]["encoder_runs"]["mean"]
                gs = pair["gs-ndn"]["encoder_runs"]["mean"]
                cells.append(f"{(es - gs) / es * 100:>7.1f}%" if es else "      -")
            label = "closed" if arrival == 0.0 else f"{arrival}s"
            print(f"    {label:<10} {'  '.join(cells)}")
    return out


def exp_gossip_period(bench: Bench, seeds: Sequence[int]) -> Dict[str, object]:
    """The anti-entropy period is a frontier, not a setting with a right answer.

    Lengthening it sends fewer gossip bytes and costs inference savings, and the
    exchange rate between the two is not constant: it depends on how many edge
    routers are waiting to learn from each other. Measured at 8 edges the trade
    looks nearly free -- the ablation's 5 s arm sends 28% fewer bytes for 3%
    more encoder work, which is what first recommended it as a default. At 16
    edges the same change costs about nine points of inference saving, because
    a longer period delays every router's learning and that delay is paid once
    per router.

    So the default sits at 500 ms, where the scaling claim of §2 is strongest,
    and this sweep is what an operator with different priorities reads instead.
    Reporting the curve is also the answer to the reviewer who prices our
    inference saving against the bytes it costs: both axes are here.
    """
    periods = (250.0, 500.0, 1000.0, 2000.0, 5000.0)
    edges = (4, 16)
    out: Dict[str, object] = {}
    for domain in bench.catalogs:
        rows: Dict[str, object] = {}
        for n_edges in edges:
            # The per-router cache is the reference both axes are measured
            # against, and it does not gossip, so it is run once per edge count.
            reference = aggregate(bench.metrics(
                base_config(domain, strategy="saf+es", n_edges=n_edges), seeds
            ))
            per_period = {"saf+es": reference}
            for period in periods:
                config = base_config(
                    domain, strategy="gs-ndn", n_edges=n_edges,
                    gossip_interval_ms=period,
                )
                per_period[str(period)] = aggregate(bench.metrics(config, seeds))
            rows[str(n_edges)] = per_period
        out[domain] = rows

        print(f"\n--- {domain}: gossip period frontier ---")
        for n_edges in edges:
            block = rows[str(n_edges)]
            baseline = block["saf+es"]["encoder_runs"]["mean"]
            print(f"  {n_edges} edge routers (saf+es reference: {baseline:.0f} inferences)")
            for period in periods:
                m = block[str(period)]
                runs = m["encoder_runs"]["mean"]
                saving = (baseline - runs) / baseline * 100 if baseline else float("nan")
                print(
                    f"    {period / 1000:>5.2f}s  inferences {runs:>7.0f} "
                    f"(saving {saving:>5.1f}%)   gossip {m['gossip_bytes']['mean']:>9,.0f} B"
                    f"   ISR {m['isr']['mean']:.4f}"
                )
    return out


EXPERIMENTS: Dict[str, Callable[[Bench, Sequence[int]], Dict[str, object]]] = {
    "gossip_period": exp_gossip_period,
    "robust": exp_robust,
    "vocabulary": exp_vocabulary,
    "horizon": exp_horizon,
    "coverage": exp_coverage,
    "risk": exp_risk,
    "churn": exp_churn,
    "drift_paired": exp_drift_paired,
    "poisoning": exp_poisoning,
    "heterogeneity": exp_heterogeneity,
    "main": exp_main,
    "threshold": exp_threshold,
    "threshold_transfer": exp_threshold_transfer,
    "ontology": exp_ontology,
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
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="seeded runs to execute in parallel (0 = one per CPU core). "
             "Results are identical to --jobs 1: every run is a pure function "
             "of its own seeded config, and output is reassembled in "
             "submission order.",
    )
    args = parser.parse_args()
    if args.jobs == 0:
        import os

        args.jobs = os.cpu_count() or 1

    chosen = sorted(EXPERIMENTS) if args.all else (args.experiment or ["main"])
    seeds = list(range(1, args.seeds + 1))

    if not COSTS_PATH.exists():
        print(f"[!] {COSTS_PATH} missing -- latencies will be assumed, not measured.")
        print("    Run: python experiments/bench_micro.py")

    bench = Bench(args.domains, args.model, jobs=args.jobs)
    if bench.jobs > 1:
        print(f"[+] running {bench.jobs} seeded runs in parallel")
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
