"""Assemble a scenario, run it, and score it.

One entry point, ``run_once``, so that every strategy is exercised through
identical topology, workload, cost model and scoring code.  The only thing that
varies between arms of an experiment is the object plugged into each router's
strategy slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import datasets, topology as topo
from .datasets import VARIANT
from .costs import CostModel, LinkProfile
from .des import Simulator
from .embeddings import EmbeddingBackend, NameIndex, PrecomputedBackend
from .energy import EnergyLedger, EnergyModel, charge_network
from .adversary import Adversary, AdversaryConfig
from .churn import ChurnConfig, ChurnDriver
from .gossip import GossipProtocol
from .metrics import Collector, convergence_curve, summarise
from .network import Router
from .packets import Interest
from .strategies import build_strategy
from .workload import Workload, WorkloadConfig, generate

DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"


@dataclass
class ScenarioConfig:
    """Everything that defines one run."""

    domain: str = "hospital"
    strategy: str = "gs-ndn"
    threshold: float = 0.7

    topology: str = "multi_edge"
    n_edges: int = 6
    n_producers: int = 4
    grid_side: int = 4
    consumers_per_edge: int = 1

    es_capacity: int = 1024
    cs_capacity: int = 0
    queue_capacity: Optional[int] = None

    #: Share of a service's known wordings that its producer actually declares.
    #: Below 1.0, producers refuse requests that were genuinely meant for them,
    #: so the feedback the network learns from is incomplete.
    alias_coverage: float = 1.0
    #: Probability a producer's verdict is simply wrong, either way.
    feedback_noise: float = 0.0

    #: Share of semantic forwarding decisions allowed to be wrong. This is the
    #: risk-controlled strategies' only tuning knob; ``threshold`` survives for
    #: them only as the prior applied to routes with no evidence yet.
    epsilon: float = 0.05
    risk_confidence: float = 0.9
    #: Share of budget-refused decisions forwarded anyway, to keep learning.
    explore_rate: float = 0.05

    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    links: LinkProfile = field(default_factory=LinkProfile)
    churn: ChurnConfig = field(default_factory=ChurnConfig)
    adversary: AdversaryConfig = field(default_factory=AdversaryConfig)

    gossip_interval_ms: float = 500.0
    gossip_fanout: int = 2
    gossip_max_delta: int = 32
    gossip_rumour_push: bool = True

    embedding_model: str = "all-MiniLM-L6-v2-onnx"
    #: A second encoder, given to ``heterogeneous_share`` of the routers. Scores
    #: from two embedding spaces are not comparable, so this is what forces the
    #: distinction between what gossip can pool and what it cannot.
    alt_embedding_model: Optional[str] = None
    heterogeneous_share: float = 0.0
    bundle_dir: Optional[Path] = None
    costs_path: Optional[Path] = None

    seed: int = 0

    def bundle_path(self) -> Path:
        directory = self.bundle_dir or DEFAULT_BUNDLE_DIR
        return Path(directory) / f"{self.domain}.{self.embedding_model}.npz"

    def with_seed(self, seed: int) -> "ScenarioConfig":
        return replace(
            self, seed=seed,
            workload=replace(self.workload, seed=seed),
            churn=replace(self.churn, seed=seed),
            adversary=replace(self.adversary, seed=seed),
        )


@dataclass
class RunResult:
    config: ScenarioConfig
    metrics: Dict[str, float]
    collector: Collector
    gossip: Dict[str, float] = field(default_factory=dict)
    router_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    strategy_stats: Dict[str, float] = field(default_factory=dict)
    cost_model_measured: bool = False

    def curve(self, window: int = 200):
        return convergence_curve(self.collector, window)


def _build_topology(
    config: ScenarioConfig, sim: Simulator, names: Sequence[str], catalog
) -> topo.Topology:
    shared = {
        "profile": config.links,
        "n_producers": config.n_producers,
        "cs_capacity": config.cs_capacity,
        "es_capacity": config.es_capacity,
        "queue_capacity": config.queue_capacity,
        "admission": topo.AdmissionSpec(
            catalog=catalog,
            alias_coverage=config.alias_coverage,
            noise=config.feedback_noise,
            seed=config.seed,
        ),
    }
    if config.topology == "multi_edge":
        return topo.multi_edge(
            sim, names, n_edges=config.n_edges,
            consumers_per_edge=config.consumers_per_edge, **shared,
        )
    if config.topology == "single_edge":
        return topo.single_edge(
            sim, names, consumers_per_edge=config.consumers_per_edge, **shared
        )
    if config.topology == "grid":
        return topo.grid(sim, names, side=config.grid_side, **shared)
    raise ValueError(f"unknown topology {config.topology!r}")


def run_once(
    config: ScenarioConfig,
    *,
    backend: Optional[EmbeddingBackend] = None,
    catalog=None,
) -> RunResult:
    """Build the scenario, drive it to completion and score it.

    ``backend`` and ``catalog`` can be passed in so a sweep of many runs loads
    the embedding bundle once rather than once per run.
    """
    catalog = catalog or datasets.load(config.domain)
    backend = backend or PrecomputedBackend(config.bundle_path())
    costs = CostModel.load_or_default(config.costs_path)
    if backend.cost.measured and backend.cost.encode_ms > 0:
        costs.encode_ms = backend.cost.encode_ms

    sim = Simulator()
    topology = _build_topology(config, sim, catalog.canonical_names, catalog)

    # Every router that may run the encoder needs the FIB embeddings. Building
    # the index once and sharing it matches the deployment, where embeddings are
    # computed when a route is installed rather than per Interest.
    index = NameIndex(backend, catalog.canonical_names)
    alt_index = None
    if config.alt_embedding_model and config.heterogeneous_share > 0.0:
        alt_backend = PrecomputedBackend(
            (config.bundle_dir or DEFAULT_BUNDLE_DIR)
            / f"{config.domain}.{config.alt_embedding_model}.npz"
        )
        alt_index = NameIndex(alt_backend, catalog.canonical_names)

    strategy = build_strategy(
        config.strategy, threshold=config.threshold, costs=costs, seed=config.seed,
        epsilon=config.epsilon, confidence=config.risk_confidence,
        explore_rate=config.explore_rate,
    )
    gossip: Optional[GossipProtocol] = None
    if getattr(strategy, "gossip", False):
        gossip = GossipProtocol(
            topology.network, sim,
            interval_ms=config.gossip_interval_ms,
            fanout=config.gossip_fanout,
            max_delta=config.gossip_max_delta,
            rumour_push=config.gossip_rumour_push,
            apply_cost_ms=costs.gossip_apply_ms_per_entry,
            seed=config.seed,
        )
        gossip.attach()

    routers = topology.routers
    # Deterministic assignment by position, so which routers run which encoder
    # is a property of the scenario rather than of iteration order.
    n_alt = int(round(len(routers) * config.heterogeneous_share)) if alt_index else 0
    for position, router in enumerate(routers):
        router.strategy = strategy
        router.name_index = alt_index if position < n_alt else index
        router.gossip_protocol = gossip
        strategy.attach(router)

    collector = Collector()
    workload = generate(catalog, topology.consumers, config.workload)

    for consumer_id in topology.consumers:
        consumer = topology.network.nodes[consumer_id]
        consumer.on_complete = (  # type: ignore[attr-defined]
            lambda interest, outcome, latency: collector.record(
                interest, outcome, latency, sim.now
            )
        )

    for request in workload.requests:
        consumer = topology.network.nodes[request.consumer]
        sim.schedule(
            request.at_ms,
            _emit,
            consumer,
            request.name,
            request.expected,
            request.kind,
        )

    churn = ChurnDriver(topology, sim, config.churn)
    adversary = Adversary(
        topology, gossip, config.adversary,
        interval_ms=config.gossip_interval_ms,
        targets=[i.name for i in catalog.by_kind(VARIANT)],
    )
    if gossip is not None:
        gossip.start()
    churn.start()
    adversary.start()

    horizon = config.workload.duration_ms + config.links.interest_lifetime_ms * 2
    sim.run(until=horizon)
    adversary.stop()
    churn.stop()
    if gossip is not None:
        gossip.stop()

    # Anything still outstanding never came back; count it rather than dropping
    # it, otherwise a strategy that stalls Interests looks better than one that
    # rejects them.
    for consumer_id in topology.consumers:
        topology.network.nodes[consumer_id].expire(0.0)  # type: ignore[attr-defined]

    metrics = summarise(collector)
    metrics.update(_router_metrics(topology.routers, sim.now))
    ledger = charge_network(
        topology.network, EnergyLedger(EnergyModel()), config.workload.duration_ms
    )
    metrics.update(ledger.report(len(collector)))

    gossip_report = gossip.report() if gossip is not None else {}
    metrics.update(gossip_report)
    # Strategy counters belong in the aggregated metrics too: calibrated-route
    # counts and boundary spread are results, not diagnostics, and leaving them
    # only on the run object meant a sweep reported them as zero.
    metrics.update({
        k: float(v) for k, v in strategy.stats().items()
        if isinstance(v, (int, float))
    })
    metrics.update(churn.report())
    metrics.update(adversary.audit())

    return RunResult(
        config=config,
        metrics=metrics,
        collector=collector,
        gossip=gossip_report,
        router_stats={r.id: _one_router(r, sim.now) for r in topology.routers},
        strategy_stats=strategy.stats(),
        cost_model_measured=costs.fully_measured,
    )


def _emit(consumer, name: str, expected: Optional[str], kind: str) -> None:
    consumer.send(
        Interest(
            name=name, origin=consumer.id, created_at=consumer.sim.now,
            expected=expected, kind=kind,
        )
    )


def _one_router(router: Router, now: float) -> Dict[str, float]:
    return {
        "interests_seen": router.interests_seen,
        "interests_forwarded": router.interests_forwarded,
        "interests_rejected": router.interests_rejected,
        "encoder_runs": router.encoder_runs,
        "cpu_utilization": router.cpu.utilization,
        "cpu_mean_wait_ms": router.cpu.mean_wait,
        "cpu_max_queue": router.cpu.max_queue_len,
        "es_entries": len(router.es),
        "es_hit_ratio": router.es.hit_ratio,
        "es_imported": router.es.imported,
        "es_evictions": router.es.evictions,
        "bytes_tx": router.bytes_tx,
        "bytes_rx": router.bytes_rx,
    }


def _router_metrics(routers: Sequence[Router], now: float) -> Dict[str, float]:
    if not routers:
        return {}
    encoder_runs = sum(r.encoder_runs for r in routers)
    return {
        "encoder_runs": encoder_runs,
        "encoder_runs_per_router": encoder_runs / len(routers),
        "cpu_utilization_max": max(r.cpu.utilization for r in routers),
        "cpu_wait_mean_ms": sum(r.cpu.mean_wait for r in routers) / len(routers),
        "cpu_queue_max": max(r.cpu.max_queue_len for r in routers),
        "es_entries_total": sum(len(r.es) for r in routers),
        "es_imported_total": sum(r.es.imported for r in routers),
        "es_hit_ratio_mean": sum(r.es.hit_ratio for r in routers) / len(routers),
        "pit_aggregated": sum(r.pit.aggregated for r in routers),
    }
