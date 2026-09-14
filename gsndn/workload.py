"""Request traces: which name is asked for, by whom, and when.

Three properties of the trace decide whether the results mean anything.

*Popularity.*  Zipf with alpha 1.2, the skew SAF uses, so a few services carry
most of the traffic and a cache has something to exploit.  Uniform is available
as the harder contrast.

*Arrival process.*  Poisson at a configurable rate, rather than a fixed list
processed as fast as the loop runs.  Rate is the independent variable in SAF's
efficiency analysis, and without a real arrival process there is no queueing and
therefore no bottleneck to observe.

*Locality per consumer.*  Each consumer population draws from an overlapping but
distinct slice of the catalog.  This is what makes sharing knowledge between
edge routers a question rather than a foregone conclusion: with identical
traffic everywhere, gossip would be redundant, and with disjoint traffic it
would be useless.  ``overlap`` controls where between those extremes the
experiment sits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .datasets import DISTRACTOR, EXACT, VARIANT, InterestName, NameCatalog


@dataclass
class WorkloadConfig:
    """Everything that shapes the request trace."""

    rate_per_s: float = 100.0          # aggregate Interests per second
    duration_ms: float = 30_000.0
    popularity: str = "zipf"           # "zipf" or "uniform"
    zipf_alpha: float = 1.2

    #: Share of requests that use a reworded name rather than the exact one.
    variation_rate: float = 0.4
    #: Share of requests that cannot be satisfied by any service.
    distractor_rate: float = 0.1

    #: How much of the catalog each consumer population shares with the others.
    #: 1.0 means every consumer asks about everything; 0.0 means disjoint slices.
    overlap: float = 0.5

    #: Mean seconds between one new wording entering the network's vocabulary.
    #:
    #: Zero -- the default, and what every result published before this existed
    #: used -- makes the whole catalog askable from t=0. That is a *closed*
    #: vocabulary, and it is the hidden assumption behind section 2's horizon
    #: result: once every router has met all 300 rewordings there is nothing
    #: left for anyone to learn, so sharing necessarily stops paying and the
    #: measured win decays toward zero. The decay is a property of the closed
    #: catalog, not of the protocol.
    #:
    #: Above zero, wordings arrive over the run in a seeded order and a request
    #: may only use one that has already arrived. This is the realistic case: a
    #: deployment meets new phrasings as client applications, vendors and
    #: integrations appear, and never finishes meeting them. The sweep over this
    #: parameter is what turns "our advantage shrinks with run length" into a
    #: statement about *what* it is proportional to.
    vocabulary_arrival_s: float = 0.0

    seed: int = 0


@dataclass
class Request:
    """One scheduled Interest."""

    at_ms: float
    consumer: str
    name: str
    expected: Optional[str]
    kind: str


@dataclass
class Workload:
    requests: List[Request]
    config: WorkloadConfig
    per_consumer: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.requests)

    def summary(self) -> Dict[str, object]:
        kinds: Dict[str, int] = {}
        for request in self.requests:
            kinds[request.kind] = kinds.get(request.kind, 0) + 1
        return {
            "requests": len(self.requests),
            "distinct_names": len({r.name for r in self.requests}),
            "kinds": kinds,
            "rate_per_s": self.config.rate_per_s,
            "duration_ms": self.config.duration_ms,
        }


def _popularity_weights(n: int, config: WorkloadConfig) -> np.ndarray:
    if config.popularity == "uniform":
        return np.full(n, 1.0 / n)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, config.zipf_alpha)
    return weights / weights.sum()


def _consumer_slices(
    n_services: int, n_consumers: int, overlap: float, rng: np.random.Generator
) -> List[np.ndarray]:
    """Give each consumer population a shared core plus a private remainder."""
    overlap = float(np.clip(overlap, 0.0, 1.0))
    shared_count = int(round(n_services * overlap))
    shared = rng.permutation(n_services)[:shared_count]

    private_pool = np.setdiff1d(np.arange(n_services), shared)
    rng.shuffle(private_pool)
    chunks = np.array_split(private_pool, max(1, n_consumers))

    slices = []
    for i in range(n_consumers):
        combined = np.concatenate([shared, chunks[i % len(chunks)]])
        slices.append(combined if combined.size else np.arange(n_services))
    return slices


def _vocabulary_arrivals(
    catalog: NameCatalog, config: WorkloadConfig
) -> Dict[str, float]:
    """When each rewording first becomes askable, in milliseconds.

    Empty when the vocabulary is closed, which is both the default and the
    condition under which this function must not perturb anything.

    The arrival order is drawn from its own generator rather than the trace's,
    so switching arrivals on changes *which* wording a request uses and nothing
    else -- same number of requests, same arrival times, same consumers, same
    services. Without that separation the Poisson gaps drawn later in
    :func:`generate` shift too, and an arrival sweep would be comparing traces
    of different lengths against each other rather than the one variable.

    Arrivals are evenly spaced rather than Poisson. The quantity under study is
    the *rate* at which the network meets wordings it has not collectively
    resolved, and spacing them evenly measures the response to that rate
    without folding in the variance of a second arrival process.
    """
    if config.vocabulary_arrival_s <= 0.0:
        return {}
    names = [interest.name for interest in catalog.by_kind(VARIANT)]
    if not names:
        return {}
    order = np.random.default_rng(config.seed + 90_210).permutation(len(names))
    step_ms = config.vocabulary_arrival_s * 1000.0
    return {names[int(idx)]: rank * step_ms for rank, idx in enumerate(order)}


def generate(
    catalog: NameCatalog,
    consumers: Sequence[str],
    config: WorkloadConfig,
) -> Workload:
    """Build a Poisson-arrival trace over the catalog."""
    rng = np.random.default_rng(config.seed)
    services = catalog.canonical_names
    n_services = len(services)

    variants_by_service: Dict[str, List[InterestName]] = {}
    for interest in catalog.by_kind(VARIANT):
        variants_by_service.setdefault(interest.expected or "", []).append(interest)
    distractors = catalog.by_kind(DISTRACTOR)
    exact_by_service = {i.expected: i for i in catalog.by_kind(EXACT)}

    weights = _popularity_weights(n_services, config)
    slices = _consumer_slices(n_services, len(consumers), config.overlap, rng)
    arrival_at = _vocabulary_arrivals(catalog, config)

    # Poisson arrivals: exponential gaps at the aggregate rate.
    mean_gap_ms = 1000.0 / max(config.rate_per_s, 1e-9)
    n_expected = max(1, int(config.duration_ms / mean_gap_ms * 1.2))
    gaps = rng.exponential(mean_gap_ms, size=n_expected)
    times = np.cumsum(gaps)
    times = times[times <= config.duration_ms]

    requests: List[Request] = []
    per_consumer: Dict[str, int] = {c: 0 for c in consumers}

    for at_ms in times:
        consumer_idx = int(rng.integers(len(consumers)))
        consumer = consumers[consumer_idx]

        roll = rng.random()
        if roll < config.distractor_rate and distractors:
            pick = distractors[int(rng.integers(len(distractors)))]
            requests.append(Request(float(at_ms), consumer, pick.name, None, DISTRACTOR))
            per_consumer[consumer] += 1
            continue

        allowed = slices[consumer_idx]
        local_weights = weights[allowed]
        local_weights = local_weights / local_weights.sum()
        service = services[int(rng.choice(allowed, p=local_weights))]

        use_variant = rng.random() < config.variation_rate
        options = variants_by_service.get(service, [])
        if use_variant and options:
            # The draw is taken over the full option list whether or not the
            # vocabulary is gated, so that turning arrivals on shifts *which*
            # wording is asked for without shifting the random stream underneath
            # it. With arrivals off this reduces to the original line exactly.
            index = int(rng.integers(len(options)))
            live = (
                [o for o in options if arrival_at.get(o.name, 0.0) <= at_ms]
                if arrival_at else options
            )
            pick = live[index % len(live)] if live else exact_by_service[service]
        else:
            pick = exact_by_service[service]

        requests.append(Request(float(at_ms), consumer, pick.name, pick.expected, pick.kind))
        per_consumer[consumer] += 1

    return Workload(requests=requests, config=config, per_consumer=per_consumer)
