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

Three events, all of which a smart hospital sees routinely as equipment is
wheeled between wards:

``depart``    a producer stops answering; its routes must be withdrawn
``arrive``    a producer returns, restoring routes
``relocate``  a producer's services move to a different attachment point, so
              the name is still served but by a different face

Relocation is the interesting one. Departure is detectable by any timeout
mechanism; relocation is silent, and a router with a cached mapping keeps
forwarding to a face that no longer leads anywhere useful.
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

    def as_dict(self) -> Dict[str, float]:
        return {
            "churn_departures": self.departures,
            "churn_arrivals": self.arrivals,
            "churn_relocations": self.relocations,
            "churn_routes_withdrawn": self.routes_withdrawn,
            "churn_routes_installed": self.routes_installed,
            "churn_mappings_invalidated": self.mappings_invalidated,
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
            if self.rng.random() < self.config.relocate_share:
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
