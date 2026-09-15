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
from gsndn.datasets import DISTRACTOR, EXACT, VARIANT, ontology  # noqa: E402
from gsndn.des import ServiceQueue, Simulator, Stopwatch  # noqa: E402
from gsndn.embeddings import NameIndex, PrecomputedBackend  # noqa: E402
from gsndn.metrics import CORRECT, MISDELIVERED, Collector, aggregate, summarise  # noqa: E402
from gsndn.packets import Interest, OUTCOME_MISDELIVERED, OUTCOME_NACK  # noqa: E402
from gsndn.churn import ChurnConfig, ChurnDriver  # noqa: E402
from gsndn.risk import (  # noqa: E402
    ACI_EPSILON_MAX,
    ACI_EPSILON_MIN,
    Observation,
    RiskController,
)
from gsndn.runner import ScenarioConfig, run_once  # noqa: E402
from gsndn.topology import AdmissionSpec, multi_edge  # noqa: E402
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


# --- ontology grounding -----------------------------------------------------


@pytest.mark.parametrize("domain", ["hospital", "city"])
def test_grounding_leaves_the_reported_catalogs_untouched(domain):
    """The campaign's numbers must not move because a new dataset was added."""
    base = datasets.load(domain)
    grounded = datasets.load(f"{domain}-grounded")
    assert base.canonical_names == grounded.canonical_names
    base_names = {i.name for i in base.interests}
    assert base_names < {i.name for i in grounded.interests}
    assert base.metadata["variants_per_service"] == 6
    assert grounded.metadata["variants_per_service"] == 7


@pytest.mark.parametrize("domain", ["hospital-grounded", "city-grounded"])
def test_grounded_catalog_reports_how_much_of_it_is_grounded(domain):
    meta = datasets.load(domain).metadata["ontology"]
    assert meta["terms_from_standards"] > 0
    # Fallbacks are counted, not hidden: no published vocabulary names a
    # glucose monitor or a bus timetable.
    assert meta["terms_from_standards"] + meta["invented_fallbacks"] == 50
    assert set(meta["ungrounded_keys"]).isdisjoint(meta["grounded_keys"])


def test_every_ontology_term_names_its_source():
    for key, terms in ontology.TERMS.items():
        assert terms, key
        for term in terms:
            assert term.standard in ontology.STANDARDS
            assert term.source_name, term.term


# --- adaptive conformal inference -------------------------------------------


def test_aci_tightens_the_budget_after_errors():
    """A stream of wrong decisions must push the effective budget down."""
    controller = RiskController(epsilon=0.2, adaptive=True, adapt_rate=0.05)
    assert controller.epsilon_target == 0.2
    for _ in range(3):
        controller.adapt(1.0)
    assert controller.epsilon < 0.2
    # gamma * (target - 1) per step, exactly.
    assert controller.epsilon == pytest.approx(0.2 + 3 * 0.05 * (0.2 - 1.0))


def test_aci_loosens_the_budget_after_clean_decisions():
    controller = RiskController(epsilon=0.05, adaptive=True, adapt_rate=0.1)
    for _ in range(4):
        controller.adapt(0.0)
    assert controller.epsilon > 0.05
    assert controller.epsilon == pytest.approx(0.05 + 4 * 0.1 * 0.05)


def test_aci_settles_where_realised_error_matches_the_target():
    """Feed the target rate back and the budget must stop moving."""
    controller = RiskController(epsilon=0.1, adaptive=True, adapt_rate=0.1)
    for _ in range(50):
        controller.adapt(0.1)
    assert controller.epsilon == pytest.approx(0.1)


def test_aci_stays_inside_its_clip_range():
    low = RiskController(epsilon=0.2, adaptive=True, adapt_rate=0.5)
    for _ in range(100):
        low.adapt(1.0)
    assert low.epsilon == pytest.approx(ACI_EPSILON_MIN)

    high = RiskController(epsilon=0.5, adaptive=True, adapt_rate=0.5)
    for _ in range(100):
        high.adapt(0.0)
    assert high.epsilon == pytest.approx(ACI_EPSILON_MAX)


def test_aci_raises_the_boundary_it_hands_out():
    """The point of the update: a tighter budget must mean a stricter route.

    Without this the arithmetic could be right and change nothing, because the
    budget only matters through the boundary it produces.
    """
    def calibrated(**kwargs):
        controller = RiskController(prior=0.5, confidence=0.9, **kwargs)
        for i in range(60):
            controller.observe(
                "/svc", Observation(score=0.5 + 0.005 * (i % 40), correct=i % 5 != 0)
            )
        return controller

    loose = calibrated(epsilon=0.4)
    tight = calibrated(epsilon=0.05)
    assert tight.boundary_for("/svc") > loose.boundary_for("/svc")


def test_a_fixed_controller_never_moves_its_budget():
    controller = RiskController(epsilon=0.2)
    for i in range(40):
        controller.observe("/svc", Observation(score=0.9, correct=i % 3 != 0))
    assert controller.epsilon == 0.2
    assert controller.adapt_steps == 0


# --- schema drift -----------------------------------------------------------


def test_schema_drift_shrinks_a_declaration_without_touching_routes():
    """The whole point of the event: no route change, no invalidation."""
    catalog = datasets.load("hospital")
    sim = Simulator()
    topology = multi_edge(
        sim, catalog.canonical_names, n_edges=2, n_producers=2,
        admission=AdmissionSpec(catalog=catalog, alias_coverage=1.0, seed=1),
    )
    driver = ChurnDriver(topology, sim, ChurnConfig(interval_s=1.0, drift_share=1.0))

    producer_id = topology.producers[0]
    policy = topology.network.nodes[producer_id].policy
    before = {k: set(v.declared) for k, v in policy.schemas.items()}
    fib_before = {r.id: len(r.fib) for r in topology.routers}

    driver._schema_drift(producer_id)

    after = {k: set(v.declared) for k, v in policy.schemas.items()}
    shrunk = [k for k in before if after[k] < before[k]]
    assert len(shrunk) == 1, "exactly one service should drift per event"
    assert driver.stats.schema_drifts == 1
    assert driver.stats.aliases_dropped == len(before[shrunk[0]] - after[shrunk[0]])

    assert {r.id: len(r.fib) for r in topology.routers} == fib_before
    assert driver.stats.routes_withdrawn == 0
    assert driver.stats.mappings_invalidated == 0


def test_a_drifted_producer_still_answers_its_own_name():
    """Drift must be invisible to exact matching, or it is not silent."""
    catalog = datasets.load("hospital")
    sim = Simulator()
    topology = multi_edge(
        sim, catalog.canonical_names, n_edges=2, n_producers=2,
        admission=AdmissionSpec(catalog=catalog, alias_coverage=1.0, seed=1),
    )
    driver = ChurnDriver(topology, sim, ChurnConfig(interval_s=1.0, drift_share=1.0))
    producer_id = topology.producers[0]
    policy = topology.network.nodes[producer_id].policy
    for _ in range(20):
        driver._schema_drift(producer_id)
    for canonical, schema in policy.schemas.items():
        assert schema.admits(canonical), canonical


# --- cross-encoder evidence -------------------------------------------------


@needs_bundle
def test_evidence_does_not_cross_an_encoder_boundary_unless_told_to():
    shared = dict(
        n_edges=6, epsilon=0.2, alt_embedding_model="lexical-char35-512",
        heterogeneous_share=0.5,
        workload=WorkloadConfig(rate_per_s=120, duration_ms=8000, seed=11),
    )
    guarded = run_once(ScenarioConfig(strategy="rc-ndn", **shared)).metrics
    naive = run_once(ScenarioConfig(strategy="rc-ndn-naive-mix", **shared)).metrics
    assert guarded["gossip_evidence_dropped_encoder"] > 0
    assert guarded["gossip_evidence_mixed_encoder"] == 0
    assert naive["gossip_evidence_dropped_encoder"] == 0
    assert naive["gossip_evidence_mixed_encoder"] > 0


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


def test_reproducible_across_interpreter_hash_seeds():
    """The same seed must give the same numbers in a *different process*.

    Two things broke this and neither showed up in the same-process test above,
    because both depend on the hash salt Python picks per interpreter: the
    gossip digest was folded from ``hash()`` over strings, and PIT in-faces were
    iterated as a set, which set the order replies went out and so the order
    events entered the queue. Same-seed runs then drifted by about 0.03% in the
    gossip counters between processes -- small, but it meant nobody re-running
    the campaign could reproduce the numbers in RESULTS.md exactly.

    Running the child with an explicit differing PYTHONHASHSEED is the point:
    the failure is invisible unless the salt actually differs.
    """
    import json
    import os
    import subprocess
    import sys

    script = (
        "import json, sys;"
        "sys.path.insert(0, %r);"
        "from gsndn.runner import ScenarioConfig, run_once;"
        "from gsndn.workload import WorkloadConfig;"
        "c = ScenarioConfig(strategy='gs-ndn', n_edges=4,"
        " workload=WorkloadConfig(rate_per_s=100, duration_ms=6000, seed=23));"
        "m = run_once(c).metrics;"
        "print(json.dumps({k: m[k] for k in"
        " ('isr', 'encoder_runs', 'gossip_mappings_applied', 'es_imported_total')}))"
    ) % str(ROOT)

    runs = []
    for hash_seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, check=True,
        )
        runs.append(json.loads(out.stdout))

    assert runs[0] == runs[1], (
        "same seed, different interpreter hash salt, different numbers: "
        f"{runs[0]} vs {runs[1]}"
    )


def test_adversary_is_reproducible_across_interpreter_hash_seeds():
    """The poisoning arm must not depend on the interpreter's hash salt.

    ``test_reproducible_across_interpreter_hash_seeds`` guards the honest path
    and missed this one: ``Adversary.compromised`` is a set of router ids, and
    iterating it fixed the order the attacker injected in, which fixed the order
    those events entered the queue. Unlike the gossip-digest and PIT-in-face
    defects that preceded it, this one moved ISR -- by about 0.006 at a fixed
    seed -- so section 9 was not reproducible in a fresh process.

    Verified by reverting the ``sorted`` in ``Adversary._tick``.
    """
    import json
    import os
    import subprocess
    import sys

    script = (
        "import json, sys;"
        "sys.path.insert(0, %r);"
        "from gsndn.runner import ScenarioConfig, run_once;"
        "from gsndn.workload import WorkloadConfig;"
        "from gsndn.adversary import AdversaryConfig;"
        "c = ScenarioConfig(strategy='gs-ndn', n_edges=8, threshold=0.6,"
        " workload=WorkloadConfig(rate_per_s=150, duration_ms=20000, seed=5),"
        " adversary=AdversaryConfig(compromised_share=0.25)).with_seed(5);"
        "m = run_once(c).metrics;"
        "print(json.dumps({k: m[k] for k in"
        " ('isr', 'encoder_runs', 'adv_poison_live', 'gossip_mappings_applied')}))"
    ) % str(ROOT)

    runs = []
    for hash_seed in ("0", "99"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, check=True,
        )
        runs.append(json.loads(out.stdout))

    assert runs[0] == runs[1], (
        "same seed, different interpreter hash salt, different numbers under "
        f"attack: {runs[0]} vs {runs[1]}"
    )


# --- peer reputation --------------------------------------------------------


def test_a_fresh_peer_is_believed():
    """Reputation must not break the bootstrap it is defending.

    A router that distrusts every unaudited peer can never accept the first
    mapping a neighbour teaches it, which is the whole value of gossip. The
    prior is deliberately trusting and the cost of that is bounded and stated:
    MIN_CLAIMS free lies per identity.
    """
    from gsndn.reputation import ReputationTable

    table = ReputationTable()
    assert table.trust("edge-1") == 1.0
    assert table.admits_mapping("edge-1", "/a/temp", "/b/temperature")


def test_a_peer_that_keeps_being_refuted_stops_being_believed():
    from gsndn.reputation import MIN_CLAIMS, ReputationTable

    table = ReputationTable()
    for i in range(MIN_CLAIMS * 3):
        table.debit("liar", f"/v{i}", f"/c{i}")
    assert table.trust("liar") < table.floor
    assert not table.admits_mapping("liar", "/fresh", "/route")
    # An honest peer over the same window is untouched: the table has to
    # separate peers, not simply become paranoid under attack.
    for i in range(MIN_CLAIMS * 3):
        table.credit("honest")
    assert table.trust("honest") > table.floor
    assert table.admits_mapping("honest", "/fresh", "/route")


def test_trust_is_a_lower_bound_not_a_point_estimate():
    """Three clean claims is not proof, for the same reason §6 uses Wilson."""
    from gsndn.reputation import MIN_CLAIMS, ReputationTable

    table = ReputationTable()
    for _ in range(MIN_CLAIMS):
        table.credit("lucky")
    assert table.trust("lucky") < 1.0


def test_a_pair_refuted_first_hand_is_not_reinstalled_from_gossip():
    """The gap that let §9's re-injecting attacker defeat verification.

    GsNdn already recorded refuted prefixes for its own encoder runs, but
    GossipAgent.apply never consulted them, so a mapping the router had
    personally disproved came straight back on the next anti-entropy round.
    """
    from gsndn.reputation import ReputationTable

    table = ReputationTable()
    assert table.admits_mapping("edge-2", "/hospital/temp", "/wrong/route")
    table.note_refuted("/hospital/temp", "/wrong/route")
    assert not table.admits_mapping("edge-2", "/hospital/temp", "/wrong/route")
    # A different route for the same wording is still allowed: the refutation
    # is about the pair, not about the name.
    assert table.admits_mapping("edge-2", "/hospital/temp", "/right/route")


def test_no_single_peer_may_own_the_calibration_window():
    """The evidence channel's defence is arithmetic, not detection."""
    from gsndn.reputation import ReputationTable

    table = ReputationTable(peer_share=0.25)
    window = 256
    assert table.admits_evidence("peer-a", held_from_peer=0, window=window)
    assert table.admits_evidence("peer-a", held_from_peer=63, window=window)
    assert not table.admits_evidence("peer-a", held_from_peer=64, window=window)
    assert table.evidence_blocked_capped == 1


def test_held_from_counts_only_that_source():
    controller = RiskController(epsilon=0.2, prior=0.6)
    controller.observe("/route", Observation(0.8, True, 0.0, source="local"))
    controller.observe("/route", Observation(0.8, True, 0.0, source="edge-3"))
    controller.observe("/route", Observation(0.8, True, 0.0, source="edge-3"))
    assert controller.held_from("/route", "edge-3") == 2
    assert controller.held_from("/route", "local") == 1
    assert controller.held_from("/absent", "edge-3") == 0


@needs_bundle
def test_the_robust_arm_changes_nothing_on_an_honest_network():
    """The defence has to be free when there is nobody to defend against.

    If reputation costs satisfaction on a clean network it cannot be left
    switched on, and every result would need a caveat about which arm was
    measured.
    """
    def isr(strategy):
        config = ScenarioConfig(
            strategy=strategy, n_edges=8, threshold=0.6,
            workload=WorkloadConfig(rate_per_s=150, duration_ms=20_000, seed=11),
        ).with_seed(11)
        return run_once(config).metrics["isr"]

    assert isr("gs-ndn-robust") == pytest.approx(isr("gs-ndn"), abs=0.01)


# --- open vocabulary --------------------------------------------------------


def test_a_closed_vocabulary_is_the_trace_that_was_always_generated():
    """Switching arrivals off must reproduce every published trace exactly."""
    catalog = datasets.load("hospital")
    consumers = ["c0", "c1", "c2", "c3"]
    config = WorkloadConfig(rate_per_s=150, duration_ms=30_000, seed=3)
    closed = generate(catalog, consumers, config)
    explicit = generate(
        catalog, consumers,
        WorkloadConfig(rate_per_s=150, duration_ms=30_000, seed=3,
                       vocabulary_arrival_s=0.0),
    )
    assert [(r.at_ms, r.consumer, r.name) for r in closed.requests] == [
        (r.at_ms, r.consumer, r.name) for r in explicit.requests
    ]


def test_arrivals_change_which_wording_is_asked_and_nothing_else():
    """The sweep is only a controlled comparison if the trace is paired.

    An earlier version drew the arrival order from the trace's own generator,
    which shifted the Poisson gaps drawn afterwards -- so turning arrivals on
    changed the number of requests too, and the comparison had two variables.
    """
    catalog = datasets.load("hospital")
    consumers = ["c0", "c1", "c2", "c3"]
    kw = dict(rate_per_s=150, duration_ms=60_000, seed=3)
    closed = generate(catalog, consumers, WorkloadConfig(**kw))
    open_ = generate(catalog, consumers, WorkloadConfig(vocabulary_arrival_s=2.0, **kw))

    assert len(closed.requests) == len(open_.requests)
    assert [r.at_ms for r in closed.requests] == [r.at_ms for r in open_.requests]
    assert [r.consumer for r in closed.requests] == [r.consumer for r in open_.requests]
    # ... and it has to actually restrict something, or the knob does nothing.
    assert open_.summary()["distinct_names"] < closed.summary()["distinct_names"]


def test_a_wording_is_never_asked_for_before_it_arrives():
    from gsndn.workload import _vocabulary_arrivals

    catalog = datasets.load("hospital")
    config = WorkloadConfig(rate_per_s=150, duration_ms=60_000, seed=3,
                            vocabulary_arrival_s=2.0)
    arrivals = _vocabulary_arrivals(catalog, config)
    workload = generate(catalog, ["c0", "c1", "c2", "c3"], config)
    early = [
        r for r in workload.requests
        if r.kind == VARIANT and r.at_ms < arrivals.get(r.name, 0.0)
    ]
    assert not early, f"{len(early)} requests used a wording that had not arrived"


def test_verification_reaches_imported_mappings_by_default():
    """GS-NDN's first claim has to hold on the path the design is about.

    Until this defect was found, a mapping learned from a peer was never checked
    against a producer however often it was used, so "verify before you trust"
    held only for mappings a router resolved itself. Pinning the default here
    stops that regressing silently.
    """
    from gsndn.strategies import build_strategy

    assert build_strategy("gs-ndn").verify_imported is True
    assert build_strategy("rc-ndn").verify_imported is True
    assert build_strategy("gs-ndn-robust").verify_imported is True
    # ... and the old behaviour stays reachable, so the defect's cost stays
    # measurable and pre-fix results can still be reproduced.
    assert build_strategy("gs-ndn-unverified-import").verify_imported is False


@needs_bundle
def test_imported_poison_is_retracted_now_that_imports_are_verified():
    """The measurement that exposed the defect, as a regression test.

    Before the fix: 608 fabricated mappings injected, 897 gossip hits served
    from them, and ``adv_poison_retracted`` exactly zero for the whole run.
    """
    from gsndn.adversary import AdversaryConfig

    def retracted(strategy):
        config = ScenarioConfig(
            strategy=strategy, n_edges=8, threshold=0.6,
            workload=WorkloadConfig(rate_per_s=150, duration_ms=30_000, seed=5),
            adversary=AdversaryConfig(compromised_share=0.25),
        ).with_seed(5)
        return run_once(config).metrics["imported_retracted"]

    assert retracted("gs-ndn-unverified-import") == 0
    assert retracted("gs-ndn") > 0


@needs_bundle
def test_every_verifying_arm_checks_its_imported_mappings():
    """RiskControlledNdn overrides resolve(), so it can miss the check.

    It did: when imports first became verifiable, gs-ndn-robust retracted
    poisoned mappings and distrusted peers while rc-ndn-robust retracted none
    and distrusted nobody, because the cached branch in the subclass had not
    been given the same treatment. Parameterised so a future strategy that
    overrides resolve() is caught by the same test.
    """
    from gsndn.adversary import AdversaryConfig

    for strategy in ("gs-ndn", "gs-ndn-robust", "rc-ndn", "rc-ndn-robust"):
        config = ScenarioConfig(
            strategy=strategy, n_edges=8, threshold=0.6, epsilon=0.2,
            workload=WorkloadConfig(rate_per_s=150, duration_ms=20_000, seed=5),
            adversary=AdversaryConfig(compromised_share=0.25),
        ).with_seed(5)
        metrics = run_once(config).metrics
        assert metrics["imported_retracted"] > 0, (
            f"{strategy} never retracted an imported mapping under attack, "
            "so its verification is not reaching the gossip path"
        )


# --- refusal reasons and propensity weighting --------------------------------


def test_a_producer_separates_a_wrong_route_from_an_undeclared_wording():
    """The defect that made §16's reason channel carry almost no information.

    The producer asked ``target not in self.names``, where ``target`` is the
    canonical prefix the *router* chose. A misroute arrives precisely because
    this producer publishes that prefix, so the test always said "yes" and every
    misroute was reported as a wording gap: measured at alias_coverage=1.0,
    132 of 157 refusals said "unknown-wording" about a route that was simply
    wrong, and ``no-such-service`` never fired once.
    """
    from gsndn.admission import AdmissionPolicy, ServiceSchema
    from gsndn.packets import REFUSAL_NO_SUCH_SERVICE, REFUSAL_UNKNOWN_WORDING

    schema = ServiceSchema(
        canonical="/h/b-a/f1/temperature/room-101", instance="room-101",
        declared=frozenset({"temperature"}),
    )
    policy = AdmissionPolicy(schemas={schema.canonical: schema})

    # Same instance, a metric this producer does not declare: genuinely
    # ambiguous, and the producer says so.
    assert policy.refusal_reason(
        schema.canonical, "/hospital/thermal-sensor/room-101"
    ) == REFUSAL_UNKNOWN_WORDING
    # A different instance is not this service, whatever the wording. The
    # producer knows which room it is in, so this is a routing fact.
    assert policy.refusal_reason(
        schema.canonical, "/hospital/temperature/room-204"
    ) == REFUSAL_NO_SUCH_SERVICE
    # A prefix this producer does not publish at all.
    assert policy.refusal_reason(
        "/h/b-a/f1/humidity/room-101", "/hospital/humidity/room-101"
    ) == REFUSAL_NO_SUCH_SERVICE


def test_propensity_rises_with_a_route_that_keeps_serving():
    controller = RiskController(epsilon=0.2, prior=0.6)
    # No history: half-believe the producer rather than commit either way.
    assert controller.wording_gap_propensity("/fresh") == pytest.approx(0.5)
    for _ in range(9):
        controller.note_outcome("/live", served=True)
    controller.note_outcome("/live", served=False, wording_refusal=True)
    for _ in range(9):
        controller.note_outcome("/bad", served=False, wording_refusal=True)
    assert controller.wording_gap_propensity("/live") > 0.8
    assert controller.wording_gap_propensity("/bad") < 0.2


def test_a_down_weighted_refusal_counts_as_a_fraction_of_a_trial():
    """Not just a fraction of an error.

    Counting a weighted observation as a partial error while still charging it
    a whole trial would make every boundary look better supported than it is.
    """
    controller = RiskController(epsilon=0.2, prior=0.6)
    controller.observe("/r", Observation(0.9, True, 0.0))
    controller.observe("/r", Observation(0.8, False, 0.0, weight=0.25))
    errors, total = controller.routes["/r"].empirical_error(0.0)
    assert errors == pytest.approx(0.25)
    assert total == pytest.approx(1.25)


def test_unweighted_observations_are_unchanged_by_the_weighted_estimator():
    """Weights must not perturb any arm that does not use them."""
    plain = RiskController(epsilon=0.2, prior=0.6)
    weighted = RiskController(epsilon=0.2, prior=0.6)
    for i in range(40):
        obs = Observation(0.5 + i / 100.0, i % 4 != 0, float(i))
        plain.observe("/r", obs)
        weighted.observe("/r", Observation(obs.score, obs.correct, obs.at_ms, weight=1.0))
    assert plain.boundary_for("/r") == pytest.approx(weighted.boundary_for("/r"))
