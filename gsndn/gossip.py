"""Anti-entropy gossip of verified name mappings.

This is the part none of the three baselines has.  SAF resolves names at a
single router and shares nothing, so every router that meets a wording pays for
its own encoder run and a router restarted with a cold cache pays again.
INF-NDN does share semantic knowledge, but through a Principal Node that
computes every tag -- one machine whose loss takes the naming layer with it.
SEF shares neighbour state but has no semantic layer at all.

What spreads here is a mapping from a client's wording to a canonical route,
and only after a Data packet has proven that route serves the name.  That
restriction is what makes the exchange safe: a router never repeats a guess, so
a wrong match cannot propagate, and a permissive similarity threshold stops
being dangerous because mistakes are retracted locally before anyone hears
about them.

The protocol is the standard pair, as the Phase 1 deck describes:

*Anti-entropy.*  Periodically, a router picks ``fanout`` neighbours and compares
digests.  Matching digests end the exchange with two packets and no payload.

*Rumour spreading.*  A newly confirmed mapping is pushed to neighbours
immediately rather than waiting for the next round, so a busy name propagates in
one link delay instead of one gossip period.

Only deltas travel.  A mapping on the wire is two names and a score, roughly 90
bytes, against the 1.5 KB a float32 embedding would cost -- the difference that
makes a continuous background exchange affordable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

from .des import Simulator
from .risk import Observation
from .tables import EsEntry

if TYPE_CHECKING:  # pragma: no cover
    from .network import Network, Router

#: Wire cost of one gossiped mapping: two names, a score and a version stamp.
MAPPING_OVERHEAD_BYTES = 12
#: One calibration observation: a prefix reference, a score and a verdict bit.
EVIDENCE_BYTES = 10
#: A digest exchange that finds nothing to do still costs two small packets.
DIGEST_BYTES = 24


@dataclass(frozen=True)
class Mapping:
    """One verified variant-to-canonical mapping, as it travels."""

    variant: str
    canonical: str
    score: float
    version: int
    origin: str

    @property
    def wire_bytes(self) -> int:
        return len(self.variant) + len(self.canonical) + MAPPING_OVERHEAD_BYTES


@dataclass(frozen=True)
class EvidenceBatch:
    """Calibration observations about one route, as they travel.

    Note what is *not* here: an embedding. Evidence is a score and a verdict
    against a named route, so it survives the trip between routers that ran
    different encoders only in the sense that the route name is meaningful to
    both -- the scores are not comparable across models, and
    :meth:`GossipProtocol.on_evidence` refuses to send them where they would be
    misread. Mappings are model-agnostic; scores are not.
    """

    prefix: str
    encoder: str
    observations: Tuple["Observation", ...]
    origin: str

    @property
    def wire_bytes(self) -> int:
        return len(self.prefix) + len(self.encoder) + EVIDENCE_BYTES * len(self.observations)


@dataclass
class GossipStats:
    rounds: int = 0
    digest_exchanges: int = 0
    mappings_sent: int = 0
    mappings_applied: int = 0
    mappings_redundant: int = 0
    conflicts_resolved: int = 0
    bytes_sent: int = 0
    rumours_pushed: int = 0
    evidence_sent: int = 0
    evidence_applied: int = 0
    evidence_dropped_encoder: int = 0
    evidence_bytes: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {
            "gossip_rounds": self.rounds,
            "gossip_digest_exchanges": self.digest_exchanges,
            "gossip_mappings_sent": self.mappings_sent,
            "gossip_mappings_applied": self.mappings_applied,
            "gossip_mappings_redundant": self.mappings_redundant,
            "gossip_conflicts_resolved": self.conflicts_resolved,
            "gossip_bytes": self.bytes_sent,
            "gossip_rumours": self.rumours_pushed,
            "gossip_bytes_per_applied": (
                self.bytes_sent / self.mappings_applied if self.mappings_applied else 0.0
            ),
            "gossip_evidence_sent": self.evidence_sent,
            "gossip_evidence_applied": self.evidence_applied,
            "gossip_evidence_dropped_encoder": self.evidence_dropped_encoder,
            "gossip_evidence_bytes": self.evidence_bytes,
        }


def _encoder_id(router: "Router") -> str:
    """Which embedding space this router's scores live in."""
    index = getattr(router, "name_index", None)
    backend = getattr(index, "backend", None)
    return getattr(backend, "id", "unknown")


class GossipAgent:
    """One router's share of the anti-entropy protocol."""

    def __init__(self, router: "Router", version_seed: int = 0) -> None:
        self.router = router
        self.known: Dict[str, Mapping] = {}
        self._version = version_seed
        self.received_from: Set[str] = set()
        self.evidence_seen = 0

        #: Highest version already handed to each peer. Anti-entropy is supposed
        #: to send what the other side lacks; without this bookkeeping a router
        #: re-sends its whole recent history every round, and gossip overhead
        #: grows with run length instead of with the number of things learned.
        self.sent_upto: Dict[str, int] = {}

    def next_version(self) -> int:
        self._version += 1
        return self._version

    def local_digest(self) -> Tuple[int, int]:
        """A summary of held knowledge: how much, and a hash of what.

        Comparing this pair is enough to skip an exchange between two routers
        that already agree, which is the common case once the network has
        converged and is why steady-state overhead stays flat.
        """
        if not self.known:
            return 0, 0
        folded = 0
        for mapping in self.known.values():
            folded ^= hash((mapping.variant, mapping.canonical, mapping.version))
        return len(self.known), folded & 0xFFFFFFFF

    def publish(self, entry: EsEntry) -> Mapping:
        """Turn a confirmed local mapping into something shareable."""
        mapping = Mapping(
            variant=entry.variant, canonical=entry.canonical,
            score=entry.score, version=self.next_version(), origin=self.router.id,
        )
        self.known[entry.variant] = mapping
        return mapping

    def delta_for(self, peer_id: str, peer_digest: Tuple[int, int], limit: int) -> List[Mapping]:
        """Mappings this peer has not been sent yet, oldest first.

        Bounded by ``limit`` per round so one exchange cannot monopolise a link;
        whatever is left over goes in the next round, and the watermark only
        advances over what was actually sent.
        """
        if peer_digest == self.local_digest():
            return []
        watermark = self.sent_upto.get(peer_id, 0)
        fresh = sorted(
            (m for m in self.known.values() if m.version > watermark),
            key=lambda m: m.version,
        )
        return fresh[:limit]

    def mark_sent(self, peer_id: str, mappings: Sequence[Mapping]) -> None:
        if mappings:
            highest = max(m.version for m in mappings)
            self.sent_upto[peer_id] = max(self.sent_upto.get(peer_id, 0), highest)

    def apply(self, mappings: Sequence[Mapping], now: float, stats: GossipStats) -> int:
        """Install mappings learned from a neighbour.

        A mapping is skipped when this router already holds the same one, and a
        locally verified mapping always wins over a remote one for the same
        wording: first-hand evidence beats hearsay, and this is what stops two
        routers from oscillating between conflicting answers.
        """
        applied = 0
        for mapping in mappings:
            existing = self.known.get(mapping.variant)
            if existing is not None and existing.version >= mapping.version:
                stats.mappings_redundant += 1
                continue

            local = self.router.es.peek(mapping.variant)
            if local is not None and local.source == "local" and local.confirmed:
                stats.conflicts_resolved += 1
                continue

            face = self._face_for(mapping.canonical)
            if face is None:
                # We have no route to that prefix, so the mapping is useless
                # here even though it is true elsewhere.
                continue

            self.known[mapping.variant] = mapping
            self.router.es.store(
                EsEntry(
                    variant=mapping.variant, canonical=mapping.canonical, face=face,
                    score=mapping.score, confirmed=True, learned_at=now,
                    source=mapping.origin,
                )
            )
            self.router.es.imported += 1
            applied += 1
        stats.mappings_applied += applied
        return applied

    def _face_for(self, canonical: str) -> Optional[str]:
        entry = self.router.fib.get(canonical)
        return entry.face if entry else None


class GossipProtocol:
    """Drives anti-entropy rounds across the whole network."""

    def __init__(
        self,
        network: "Network",
        sim: Simulator,
        *,
        interval_ms: float = 500.0,
        fanout: int = 2,
        max_delta: int = 32,
        rumour_push: bool = True,
        apply_cost_ms: float = 0.002,
        seed: int = 0,
    ) -> None:
        self.network = network
        self.sim = sim
        self.interval_ms = interval_ms
        self.fanout = fanout
        self.max_delta = max_delta
        self.rumour_push = rumour_push
        self.apply_cost_ms = apply_cost_ms
        self.rng = random.Random(seed)
        self.stats = GossipStats()
        self.agents: Dict[str, GossipAgent] = {}
        self.history: List[Tuple[float, int, int]] = []   # (t, distinct mappings, bytes)
        self._running = False

    def attach(self) -> None:
        for router in self.network.routers():
            self.agents[router.id] = GossipAgent(router)
            router.gossip = self.agents[router.id]  # type: ignore[attr-defined]

    def start(self) -> None:
        if self.interval_ms <= 0:
            return
        self._running = True
        # Stagger the first round per router so every agent does not transmit on
        # the same tick, which would produce a synchronised burst no real
        # deployment would exhibit.
        self.sim.schedule(self.rng.uniform(0.0, self.interval_ms), self._round)

    def stop(self) -> None:
        self._running = False

    def on_confirmed(self, router: "Router", entry: EsEntry) -> None:
        """A local mapping just proved itself; offer it to neighbours now."""
        agent = self.agents.get(router.id)
        if agent is None:
            return
        mapping = agent.publish(entry)
        if not self.rumour_push:
            return
        for peer_id in router.peers:
            peer = self.agents.get(peer_id)
            if peer is None:
                continue
            self.stats.rumours_pushed += 1
            self.stats.mappings_sent += 1
            self.stats.bytes_sent += mapping.wire_bytes
            agent.mark_sent(peer_id, [mapping])
            delay = self._link_delay(router, peer_id)
            self.sim.schedule(delay, self._receive, peer, [mapping])

    def on_evidence(self, router: "Router", prefix: str, observation: Observation) -> None:
        """Offer one calibration observation to routers that can read it.

        Evidence only travels to neighbours running the same encoder. A score of
        0.62 from MiniLM and a score of 0.62 from a character n-gram model are
        not the same measurement, and pooling them would produce a boundary that
        is calibrated for neither. Mappings still cross that line freely -- a
        route that served a name serves it regardless of how anyone decided to
        ask -- which is the whole reason gossip here carries names and verdicts
        rather than vectors.
        """
        agent = self.agents.get(router.id)
        if agent is None:
            return
        agent.evidence_seen += 1

        encoder = _encoder_id(router)
        batch = EvidenceBatch(
            prefix=prefix, encoder=encoder,
            observations=(observation,), origin=router.id,
        )
        for peer_id in router.peers:
            peer = self.agents.get(peer_id)
            if peer is None:
                continue
            if _encoder_id(peer.router) != encoder:
                self.stats.evidence_dropped_encoder += 1
                continue
            self.stats.evidence_sent += 1
            self.stats.evidence_bytes += batch.wire_bytes
            self.stats.bytes_sent += batch.wire_bytes
            self.sim.schedule(
                self._link_delay(router, peer_id), self._receive_evidence, peer, batch
            )

    def _receive_evidence(self, agent: "GossipAgent", batch: EvidenceBatch) -> None:
        router = agent.router
        strategy = router.strategy
        controller = getattr(strategy, "controllers", {}).get(router.id) if strategy else None
        if controller is None:
            return
        cost = self.apply_cost_ms * len(batch.observations)

        def apply() -> None:
            taken = controller.absorb(
                batch.prefix,
                [
                    Observation(o.score, o.correct, o.at_ms, source=batch.origin)
                    for o in batch.observations
                ],
            )
            self.stats.evidence_applied += taken

        router.cpu.submit(lambda: (cost, apply))

    def _round(self) -> None:
        if not self._running:
            return
        self.stats.rounds += 1
        for router in self.network.routers():
            agent = self.agents.get(router.id)
            if agent is None or not router.peers:
                continue
            peers = self.rng.sample(router.peers, min(self.fanout, len(router.peers)))
            for peer_id in peers:
                self._exchange(agent, router, peer_id)
        self._snapshot()
        self.sim.schedule(self.interval_ms, self._round)

    def _exchange(self, agent: GossipAgent, router: "Router", peer_id: str) -> None:
        peer = self.agents.get(peer_id)
        if peer is None:
            return
        self.stats.digest_exchanges += 1
        self.stats.bytes_sent += 2 * DIGEST_BYTES

        delta = agent.delta_for(peer_id, peer.local_digest(), self.max_delta)
        if not delta:
            return
        agent.mark_sent(peer_id, delta)
        payload = sum(m.wire_bytes for m in delta)
        self.stats.mappings_sent += len(delta)
        self.stats.bytes_sent += payload
        self.sim.schedule(self._link_delay(router, peer_id), self._receive, peer, delta)

    def _receive(self, agent: GossipAgent, mappings: List[Mapping]) -> None:
        router = agent.router
        cost = self.apply_cost_ms * len(mappings)
        router.cpu.submit(lambda: (cost, lambda: agent.apply(mappings, self.sim.now, self.stats)))

    def _link_delay(self, router: "Router", peer_id: str) -> float:
        face = router.faces.get(peer_id)
        return face.delay_ms if face else 1.0

    def _snapshot(self) -> None:
        distinct = len({v for a in self.agents.values() for v in a.known})
        self.history.append((self.sim.now, distinct, self.stats.bytes_sent))

    # -- reporting -------------------------------------------------------

    def coverage(self) -> float:
        """Mean fraction of all known mappings that each router holds.

        This is the convergence measure: 1.0 means every router can answer
        every wording any router has ever resolved, without an encoder.
        """
        if not self.agents:
            return 0.0
        universe = {v for a in self.agents.values() for v in a.known}
        if not universe:
            return 0.0
        return sum(len(a.known) for a in self.agents.values()) / (len(universe) * len(self.agents))

    def report(self) -> Dict[str, float]:
        data = self.stats.as_dict()
        data["gossip_coverage"] = self.coverage()
        data["gossip_distinct_mappings"] = len(
            {v for a in self.agents.values() for v in a.known}
        )
        return data
