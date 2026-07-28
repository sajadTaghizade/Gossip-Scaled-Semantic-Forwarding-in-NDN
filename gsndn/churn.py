"""Producers that come, go and move, so the right answer changes over time.

Everything measured so far assumed a network whose routes never change. That
assumption quietly does a lot of work. A learned mapping is only ever confirmed
or refuted once; a calibration observation from ten seconds ago is as good as
one from now; and the Embedding Store's invalidation path, which SAF specifies
and this code implements, never runs.

It also removes the one setting where verification is unambiguously worth its
cost. On a static network a wrong resolution is wrong forever, so retracting it
saves nothing a re-resolution would not have found anyway -- which is exactly
what the ablation showed, verification costing 6% more work for 0.8 points of
satisfaction. When producers move, a mapping that was right becomes wrong
without anybody's similarity score changing, and no amount of re-encoding
detects that. Only feedback does.

Four events, all of which a smart hospital sees routinely as equipment is
wheeled between wards and as services are reconfigured:

``depart``        a producer stops answering; its routes must be withdrawn
``arrive``        a producer returns, restoring routes
``relocate``      a producer's services move to a different attachment point, so
                  the name is still served but by a different face
``schema_drift``  a producer narrows the set of wordings it will answer to,
                  without moving and without any route changing

Relocation was expected to be the interesting one, and measurement said
otherwise. The reason is visible in :meth:`ChurnDriver._withdraw`: departure and
relocation both pull the FIB route *and* invalidate every Embedding Store entry
that pointed at it, uniformly, for every strategy. That is the consistency rule
SAF specifies and it is correct -- but it means the stale mapping is destroyed by
the route event itself, before any producer gets the chance to refuse it. The
signal verification exists to catch is erased by the mechanism that reports the
event, so no strategy can look different, and section 6 duly found that none did.

``schema_drift`` is the case that experiment was missing. A producer keeps its
routes, its attachment and its FIB entries, and simply stops recognising some of
the wordings it used to answer. Nothing is withdrawn, nothing is invalidated, no
timeout fires and no similarity score changes -- the encoder's view of the world
is exactly what it was. The only observable is a producer's live refusal, which
is precisely the channel verification and calibration are built on. If feedback
never earns its cost here it does not earn it anywhere.

What it should move is precision and realised error rather than satisfaction: a
wording the producer has stopped answering cannot be satisfied by anybody, so no
strategy recovers it. What separates them is whether they keep confidently
delivering it to a producer that will refuse.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .des import Simulator


@dataclass
class ChurnConfig:
    """How unstable the network is."""

    #: Mean seconds between churn events across the whole network. Zero disables.
    interval_s: float = 0.0
    #: Share of events that are relocations rather than departures.
    relocate_share: float = 0.5
    #: How long a departed producer stays away, in seconds.
    outage_s: float = 5.0

    #: Share of events that are schema drifts. Drawn before the
    #: relocate/depart split, so ``drift_share=1.0`` gives a network whose
    #: topology never changes and whose producers quietly narrow what they
    #: answer to -- the arm that isolates feedback from route events.
    drift_share: float = 0.0
    #: Share of a service's declared alias terms dropped per drift event. The
    #: canonical term is never dropped, so exact-match forwarding is untouched
    #: and only semantically resolved wordings are affected.
    drift_fraction: float = 0.5

    seed: int = 0

    @property
    def enabled(self) -> bool:
        return self.interval_s > 0.0


@dataclass
class ChurnStats:
    departures: int = 0
    arrivals: int = 0
    relocations: int = 0
    routes_withdrawn: int = 0
    routes_installed: int = 0
    mappings_invalidated: int = 0
    schema_drifts: int = 0
    aliases_dropped: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {
            "churn_departures": self.departures,
            "churn_arrivals": self.arrivals,
            "churn_relocations": self.relocations,
            "churn_routes_withdrawn": self.routes_withdrawn,
            "churn_routes_installed": self.routes_installed,
            "churn_mappings_invalidated": self.mappings_invalidated,
            "churn_schema_drifts": self.schema_drifts,
            "churn_aliases_dropped": self.aliases_dropped,
        }


class ChurnDriver:
    """Schedules producer departures, returns and relocations during a run."""

    def __init__(
        self,
        topology,
        sim: Simulator,
        config: ChurnConfig,
    ) -> None:
        self.topology = topology
        self.sim = sim
        self.config = config
        self.rng = random.Random(config.seed)
        self.stats = ChurnStats()
        self._running = False

        #: Where each producer is attached, so a relocation has somewhere to go.
        self.attachment: Dict[str, str] = {}
        for producer_id in topology.producers:
            producer = topology.network.nodes[producer_id]
            self.attachment[producer_id] = next(iter(producer.faces), "")

    def start(self) -> None:
        if not self.config.enabled:
            return
        self._running = True
        self.sim.schedule(self._next_gap(), self._tick)

    def stop(self) -> None:
        self._running = False

    def _next_gap(self) -> float:
        return self.rng.expovariate(1.0 / max(self.config.interval_s, 1e-9)) * 1000.0

    def _tick(self) -> None:
        if not self._running:
            return
        producers = [
            p for p in self.topology.producers
            if self.topology.network.nodes[p].available
        ]
        if producers:
            producer_id = self.rng.choice(producers)
            roll = self.rng.random()
            drift = self.config.drift_share
            if roll < drift:
                self._schema_drift(producer_id)
            elif roll < drift + (1.0 - drift) * self.config.relocate_share:
                self._relocate(producer_id)
            else:
                self._depart(producer_id)
        self.sim.schedule(self._next_gap(), self._tick)

    # -- events ----------------------------------------------------------

    def _depart(self, producer_id: str) -> None:
        producer = self.topology.network.nodes[producer_id]
        producer.available = False
        self.stats.departures += 1
        self._withdraw(producer.names)
        self.sim.schedule(self.config.outage_s * 1000.0, self._arrive, producer_id)

    def _arrive(self, producer_id: str) -> None:
        producer = self.topology.network.nodes[producer_id]
        producer.available = True
        self.stats.arrivals += 1
        self._install({producer_id: sorted(producer.names)})

    def _relocate(self, producer_id: str) -> None:
        """Move a producer to a different router, keeping it reachable.

        The names stay available, which is what makes this harder than a
        departure: nothing times out, and a router serving the old face from
        cache has no signal that anything changed except that its Interests stop
        being answered correctly.
        """
        network = self.topology.network
        producer = network.nodes[producer_id]
        candidates = [r for r in self.topology.cores + self.topology.edges]
        current = self.attachment.get(producer_id)
        options = [c for c in candidates if c != current]
        if not options:
            return

        destination = self.rng.choice(options)
        profile_delay = 20.0
        if current:
            face = producer.faces.pop(current, None)
            if face is not None:
                profile_delay = face.delay_ms
            network.nodes[current].faces.pop(producer_id, None)

        network.connect(destination, producer_id, profile_delay)
        self.attachment[producer_id] = destination
        self.stats.relocations += 1

        self._withdraw(producer.names)
        self._install({producer_id: sorted(producer.names)})

    def _schema_drift(self, producer_id: str) -> None:
        """Narrow one service's declared aliases, changing nothing else.

        The producer stays where it is, keeps every route and every FIB entry,
        and keeps answering to its canonical term -- so exact-match forwarding is
        untouched and nothing in the routing plane observes an event at all. What
        changes is that some of the wordings it used to accept now come back
        refused.

        Deliberately nothing else happens here. No ``_withdraw``, no
        ``es.invalidate_route``: the stale mapping stays in every router's
        Embedding Store, still pointing at a real route to a live producer, and
        the only way to find out it has gone bad is to send an Interest and be
        told no. That is the whole design of this event.
        """
        from dataclasses import replace as _replace

        producer = self.topology.network.nodes[producer_id]
        policy = getattr(producer, "policy", None)
        if policy is None:
            return

        candidates = [
            canonical for canonical, schema in policy.schemas.items()
            if len(schema.declared) > 1
        ]
        if not candidates:
            return

        canonical = self.rng.choice(sorted(candidates))
        schema = policy.schemas[canonical]
        keep_always = _canonical_term(canonical)
        droppable = sorted(term for term in schema.declared if term != keep_always)
        if not droppable:
            return

        n_drop = max(1, int(round(len(droppable) * self.config.drift_fraction)))
        dropped = set(self.rng.sample(droppable, min(n_drop, len(droppable))))
        policy.schemas[canonical] = _replace(
            schema, declared=frozenset(schema.declared - dropped)
        )
        self.stats.schema_drifts += 1
        self.stats.aliases_dropped += len(dropped)

    # -- route and cache maintenance -------------------------------------

    def _withdraw(self, names: Sequence[str]) -> None:
        """Pull routes, and with them every mapping that pointed at one.

        This is the consistency rule SAF specifies for the Embedding Store and
        the reason it exists: a cached resolution outlives the route it was
        resolved against unless something removes it.
        """
        for router in self.topology.routers:
            for name in names:
                if router.fib.remove(name):
                    self.stats.routes_withdrawn += 1
                self.stats.mappings_invalidated += router.es.invalidate_route(name)

    def _install(self, ownership: Dict[str, List[str]]) -> None:
        before = sum(len(r.fib) for r in self.topology.routers)
        self.topology.network.install_routes(ownership)
        after = sum(len(r.fib) for r in self.topology.routers)
        self.stats.routes_installed += max(0, after - before)

    def report(self) -> Dict[str, float]:
        return self.stats.as_dict()


def _canonical_term(canonical: str) -> str:
    """The metric component of a canonical name -- the one never dropped.

    ``build_schemas`` seeds every declaration with this term, so keeping it is
    what guarantees a drifting producer still serves its own exact name and the
    event stays invisible to the routing plane.
    """
    components = canonical.strip("/").split("/")
    return components[-2] if len(components) >= 2 else components[-1]
