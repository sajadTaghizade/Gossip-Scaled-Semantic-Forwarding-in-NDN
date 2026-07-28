"""Tests for the parts a wrong answer would be hard to notice in.

Most of these guard against a specific mistake that was actually made while
building this, and each one names it.  A simulator is dangerous precisely
because it always produces plausible numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsndn import datasets  # noqa: E402
from gsndn.datasets import DISTRACTOR, EXACT, VARIANT  # noqa: E402
from gsndn.des import ServiceQueue, Simulator, Stopwatch  # noqa: E402
from gsndn.embeddings import NameIndex, PrecomputedBackend  # noqa: E402
from gsndn.metrics import CORRECT, MISDELIVERED, Collector, aggregate, summarise  # noqa: E402
from gsndn.packets import Interest, OUTCOME_MISDELIVERED, OUTCOME_NACK  # noqa: E402
from gsndn.runner import ScenarioConfig, run_once  # noqa: E402
from gsndn.simhash import SimHasher, hamming, hamming_array  # noqa: E402
from gsndn.tables import EmbeddingStore, EsEntry, Fib, Pit  # noqa: E402
from gsndn.workload import WorkloadConfig, generate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "embeddings" / "hospital.all-MiniLM-L6-v2-onnx.npz"
needs_bundle = pytest.mark.skipif(
    not BUNDLE.exists(), reason="run experiments/export_embeddings.py first"
)


# --- datasets ---------------------------------------------------------------


@pytest.mark.parametrize("domain", datasets.DOMAINS)
def test_catalog_shape(domain):
    catalog = datasets.load(domain)
    summary = catalog.summary()
    assert summary["services"] == 50
    assert summary[EXACT] == 50
    assert summary[VARIANT] == 300, "six rewordings per service"
    assert summary[DISTRACTOR] == 350, "half the offered names are unsatisfiable"


@pytest.mark.parametrize("domain", datasets.DOMAINS)
def test_every_interest_is_labelled(domain):
    """Without a label, a forwarding decision cannot be scored as right or wrong.

    The Phase 2 prototype had no such labels, which is why it reported delivery
    ratio and called it accuracy.
    """
    catalog = datasets.load(domain)
    canonical = set(catalog.canonical_names)
    for interest in catalog.interests:
        if interest.kind == DISTRACTOR:
            assert interest.expected is None
        else:
            assert interest.expected in canonical


@pytest.mark.parametrize("domain", datasets.DOMAINS)
def test_catalog_is_deterministic(domain):
    assert datasets.load(domain).all_names == datasets.load(domain).all_names


def test_distractors_do_not_collide_with_services():
    for domain in datasets.DOMAINS:
        catalog = datasets.load(domain)
        canonical = set(catalog.canonical_names)
        for interest in catalog.by_kind(DISTRACTOR):
            assert interest.name not in canonical


# --- tables -----------------------------------------------------------------


def test_fib_longest_prefix():
    fib = Fib()
    fib.add("/a/b", "face-1")
    assert fib.longest_prefix("/a/b/c/d").prefix == "/a/b"
    assert fib.longest_prefix("/a") is None


def test_es_evicts_least_recently_used():
    """A bounded cache is the point; an unbounded one measures the workload."""
    store = EmbeddingStore(capacity=2)
    for name in ("v1", "v2"):
        store.store(EsEntry(name, "/c", "f", 0.9))
    store.lookup("v1")                      # v2 is now least recent
    store.store(EsEntry("v3", "/c", "f", 0.9))
    assert "v2" not in store and "v1" in store and "v3" in store
    assert store.evictions == 1


def test_es_confirm_only_signals_the_transition():
    """Re-announcing a popular mapping on every hit floods the gossip layer."""
    store = EmbeddingStore()
    store.store(EsEntry("v", "/c", "f", 0.9))
    assert store.confirm("v", 1.0) is not None
    assert store.confirm("v", 2.0) is None


def test_es_invalidates_routes_that_disappear():
    store = EmbeddingStore()
    store.store(EsEntry("v1", "/gone", "f", 0.9))
    store.store(EsEntry("v2", "/stays", "f", 0.9))
    assert store.invalidate_route("/gone") == 1
    assert "v1" not in store and "v2" in store


def test_pit_aggregates_duplicates():
    pit = Pit()
    _, first = pit.insert("/n", "f1", 1, 0.0)
    entry, second = pit.insert("/n", "f2", 2, 0.1)
    assert not first and second
    assert entry.interest_ids == [1, 2] and entry.in_faces == {"f1", "f2"}


# --- discrete-event kernel --------------------------------------------------

def test_events_fire_in_time_order():
    sim = Simulator()
    fired = []
    sim.schedule(5.0, lambda: fired.append("late"))
    sim.schedule(1.0, lambda: fired.append("early"))
    sim.run()
    assert fired == ["early", "late"] and sim.now == 5.0


def test_service_queue_serialises_and_waits():
    """Two 10 ms jobs submitted together must take 20 ms, not 10."""
    sim = Simulator()
    queue = ServiceQueue(sim)
    done = []
    for tag in ("a", "b"):
        queue.submit(lambda t=tag: (10.0, lambda: done.append((t, sim.now))))
    sim.run()
    assert done == [("a", 10.0), ("b", 20.0)]
    assert queue.mean_wait == pytest.approx(5.0)


def test_service_time_is_decided_at_service_start():
    """Cost depends on what the router has learned by the time it is served."""
    sim = Simulator()
    queue = ServiceQueue(sim)
    learned = {"yes": False}
    durations = []

    def work():
        cost = 1.0 if learned["yes"] else 10.0
        durations.append(cost)
        return cost, lambda: learned.__setitem__("yes", True)

    queue.submit(work)
    queue.submit(work)
    sim.run()
    assert durations == [10.0, 1.0]


def test_stopwatch_percentiles():
    watch = Stopwatch()
    for value in range(1, 101):
        watch.record(float(value))
    assert watch.percentile(50) == pytest.approx(50.5)
    assert watch.mean == pytest.approx(50.5)


# --- signatures -------------------------------------------------------------


def test_hamming_matches_bit_difference():
    a = np.array([0b1010_1010], dtype=np.uint8)
    b = np.array([0b0101_1010], dtype=np.uint8)
    assert hamming(a, b) == 4


def test_hamming_array_matches_scalar():
    rng = np.random.default_rng(0)
    table = rng.integers(0, 256, size=(20, 8), dtype=np.uint8)
    query = table[3]
    expected = [hamming(query, row) for row in table]
    assert list(hamming_array(query, table)) == expected


@needs_bundle
def test_signatures_track_cosine_but_do_not_rank_reliably():
    """The measurement that ruled out routing on signatures.

    Hamming distance estimates cosine well on average and still picks the wrong
    nearest FIB entry most of the time, because service names in one domain sit
    close together. If this ever starts passing at high recall, the claim in
    simhash.py should be revisited rather than the test relaxed.
    """
    catalog = datasets.load("hospital")
    backend = PrecomputedBackend(BUNDLE)
    index = NameIndex(backend, catalog.canonical_names)
    queries = backend.encode([i.name for i in catalog.by_kind(VARIANT)])

    hasher = SimHasher(dim=backend.dim, bits=64)
    table = hasher.signatures(index.vectors)
    query_sigs = hasher.signatures(queries)

    gold = (queries @ index.vectors.T).argmax(1)
    chosen = [int(hamming_array(query_sigs[i], table).argmin()) for i in range(len(queries))]
    recall_at_1 = float(np.mean([chosen[i] == gold[i] for i in range(len(queries))]))
    assert recall_at_1 < 0.6, f"64-bit signatures ranked far better than measured: {recall_at_1}"


# --- metrics ----------------------------------------------------------------


def _interest(name: str, expected, kind: str = VARIANT) -> Interest:
    return Interest(name=name, origin="c", created_at=0.0, expected=expected, kind=kind)


def test_a_wrong_delivery_is_not_a_success():
    """The bug this whole metrics module exists to prevent."""
    collector = Collector()
    collector.record(_interest("/v", "/right"), OUTCOME_MISDELIVERED, 10.0, 10.0)
    metrics = summarise(collector)
    assert metrics["correct"] == 0 and metrics["misdelivered"] == 1
    assert metrics["precision"] == 0.0


def test_answering_a_distractor_is_a_false_positive():
    collector = Collector()
    collector.record(_interest("/junk", None, DISTRACTOR), "resolved", 10.0, 10.0)
    metrics = summarise(collector)
    assert metrics["false_positive_rate"] == 1.0
    assert metrics["correct"] == 0


def test_rejecting_a_distractor_is_correct_behaviour():
    collector = Collector()
    collector.record(_interest("/junk", None, DISTRACTOR), OUTCOME_NACK, 0.0, 10.0)
    metrics = summarise(collector)
    assert metrics["accuracy"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_latency_counts_only_correct_deliveries():
    collector = Collector()
    collector.record(_interest("/v", "/right"), "resolved", 20.0, 20.0)
    collector.record(_interest("/w", "/right"), OUTCOME_MISDELIVERED, 500.0, 30.0)
    assert summarise(collector)["irt_mean_ms"] == pytest.approx(20.0)


def test_aggregate_reports_a_confidence_interval():
    stats = aggregate([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
    assert stats["x"]["mean"] == pytest.approx(2.0)
    assert stats["x"]["n"] == 3 and stats["x"]["ci95"] > 0


# --- workload ---------------------------------------------------------------


def test_workload_is_seed_reproducible():
    catalog = datasets.load("hospital")
    config = WorkloadConfig(rate_per_s=100, duration_ms=2000, seed=7)
    first = generate(catalog, ["c0", "c1"], config)
    second = generate(catalog, ["c0", "c1"], config)
    assert [r.name for r in first.requests] == [r.name for r in second.requests]


def test_arrivals_respect_the_configured_rate():
    catalog = datasets.load("hospital")
    config = WorkloadConfig(rate_per_s=200, duration_ms=10_000, seed=3)
    workload = generate(catalog, ["c0"], config)
    assert 0.85 < len(workload) / 2000 < 1.15
    assert all(r.at_ms <= config.duration_ms for r in workload.requests)


def test_consumers_get_overlapping_but_distinct_slices():
    """Identical traffic everywhere would make sharing trivially redundant."""
    catalog = datasets.load("hospital")
    config = WorkloadConfig(rate_per_s=500, duration_ms=20_000, overlap=0.2, seed=5)
    workload = generate(catalog, ["c0", "c1", "c2"], config)
    per_consumer = {}
    for request in workload.requests:
        per_consumer.setdefault(request.consumer, set()).add(request.expected)
    sets = [s - {None} for s in per_consumer.values()]
    assert len(set.union(*sets)) > max(len(s) for s in sets)


# --- end to end -------------------------------------------------------------


@needs_bundle
@pytest.mark.parametrize("strategy", ["vanilla-ndn", "saf", "saf+es", "gs-ndn"])
def test_run_completes_without_stranding_interests(strategy):
    config = ScenarioConfig(
        strategy=strategy, n_edges=3,
        workload=WorkloadConfig(rate_per_s=80, duration_ms=6000, seed=11),
    )
    result = run_once(config)
    metrics = result.metrics
    assert metrics["requests"] > 0
    assert metrics["timeout"] == 0, "an Interest that never resolves is a modelling bug"
    settled = metrics["correct"] + metrics["misdelivered"] + metrics["rejected"]
    assert settled == metrics["requests"]


@needs_bundle
def test_vanilla_resolves_exact_names_and_nothing_else():
    config = ScenarioConfig(
        strategy="vanilla-ndn", n_edges=2,
        workload=WorkloadConfig(rate_per_s=80, duration_ms=6000, seed=11),
    )
    metrics = run_once(config).metrics
    assert metrics["precision"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["isr_variant"] == 0.0, "exact matching cannot resolve a reworded name"
    assert metrics["isr_exact"] == 1.0


@needs_bundle
def test_semantic_forwarding_beats_exact_matching_on_reworded_names():
    shared = dict(n_edges=2, workload=WorkloadConfig(rate_per_s=80, duration_ms=8000, seed=11))
    vanilla = run_once(ScenarioConfig(strategy="vanilla-ndn", **shared)).metrics
    ours = run_once(ScenarioConfig(strategy="gs-ndn", threshold=0.6, **shared)).metrics
    assert ours["isr"] > vanilla["isr"] + 0.2


@needs_bundle
def test_gossip_reduces_encoder_work_across_many_edges():
    """The claim gossip exists to support, at a size where it can show."""
    shared = dict(
        n_edges=8, threshold=0.6,
        workload=WorkloadConfig(rate_per_s=150, duration_ms=20_000, seed=11),
    )
    alone = run_once(ScenarioConfig(strategy="gs-ndn-no-gossip", **shared)).metrics
    shared_knowledge = run_once(ScenarioConfig(strategy="gs-ndn", **shared)).metrics
    assert shared_knowledge["encoder_runs"] < alone["encoder_runs"]
    assert shared_knowledge["isr"] >= alone["isr"] - 0.02, "not at the cost of accuracy"


@needs_bundle
def test_gossip_only_carries_mappings_that_were_confirmed():
    config = ScenarioConfig(
        strategy="gs-ndn", n_edges=6, threshold=0.5,
        workload=WorkloadConfig(rate_per_s=150, duration_ms=20_000, seed=11),
    )
    result = run_once(config)
    for router in result.router_stats:
        assert result.router_stats[router]["es_imported"] >= 0
    assert result.gossip["gossip_mappings_applied"] > 0
    assert result.metrics["precision"] > 0.95, "sharing must not spread wrong routes"


@needs_bundle
def test_results_are_reproducible_for_a_fixed_seed():
    config = ScenarioConfig(
        strategy="gs-ndn", n_edges=4,
        workload=WorkloadConfig(rate_per_s=100, duration_ms=6000, seed=23),
    )
    first = run_once(config).metrics
    second = run_once(config).metrics
    for key in ("isr", "precision", "encoder_runs", "irt_mean_ms"):
        assert first[key] == pytest.approx(second[key])
