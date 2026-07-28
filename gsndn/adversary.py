"""A router that lies, and how far the lie travels.

Sharing what you have learned means trusting what you are told, and nothing so
far has asked what happens when a neighbour is compromised. That question is
sharper here than for ordinary cooperative caching, because two different things
travel: mappings, which say "this wording is served by that route", and
calibration evidence, which says "decisions on that route at this score turned
out fine". Poisoning either one has a different effect and a different defence.

**A poisoned mapping** sends Interests to the wrong producer. Its reach is
bounded by verification: a router that accepts the mapping and forwards on it
gets refused by the producer, drops it, and does not pass it on. The lie
survives one round trip per victim. Without verification nothing stops it, which
is the concrete reason confirmation matters even though the static-network
ablation showed it costing more than it saved.

**Poisoned evidence** is quieter and, it turns out, more dangerous. Fabricated
"this score worked" observations pull a route's boundary *down*, so the victim
starts accepting resolutions it would have refused. No single decision looks
wrong; the error budget is simply no longer honoured. The defence is not
detection but arithmetic -- a router's own observations outnumber any single
neighbour's contribution, so the damage is bounded by the fraction of evidence
an attacker supplies rather than by whether the attack is noticed.

Both effects are measured rather than argued. ``experiments/run_experiments.py``
sweeps the share of compromised routers and reports how far poison spreads and
what it does to the realised error rate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .gossip import GossipAgent, GossipProtocol, Mapping
from .risk import Observation


@dataclass
class AdversaryConfig:
    """Who is compromised and what they do."""

    #: Share of routers under the attacker's control.
    compromised_share: float = 0.0
    #: Fabricated mappings injected per compromised router per gossip round.
    false_mappings_per_round: int = 4
    #: Fabricated calibration observations per round, all claiming success at a
    #: low score so that victims lower their boundaries.
    false_evidence_per_round: int = 8
    #: The score a fabricated observation claims. Low, because the attacker
    #: wants the boundary pulled down to where wrong matches get through.
    false_evidence_score: float = 0.25
    seed: int = 0

    @property
    def enabled(self) -> bool:
        return self.compromised_share > 0.0


@dataclass
class AdversaryStats:
    compromised: int = 0
    false_mappings_injected: int = 0
    false_evidence_injected: int = 0
    victims_holding_poison: int = 0
    poisoned_entries_live: int = 0
    poison_retracted: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {
            "adv_compromised": self.compromised,
            "adv_false_mappings": self.false_mappings_injected,
            "adv_false_evidence": self.false_evidence_injected,
            "adv_victims": self.victims_holding_poison,
            "adv_poison_live": self.poisoned_entries_live,
            "adv_poison_retracted": self.poison_retracted,
        }


class Adversary:
    """Drives compromised routers during a run."""

    def __init__(
        self,
        topology,
        gossip: Optional[GossipProtocol],
        config: AdversaryConfig,
        *,
        interval_ms: float = 500.0,
        targets: Optional[Sequence[str]] = None,
    ) -> None:
        self.topology = topology
        #: Wordings clients actually use. Poisoning a name nobody asks for wastes
        #: cache and nothing else; the attack only bites on live traffic, and an
        #: attacker who has compromised a router can see exactly which names that
        #: is. An earlier version invented names and measured a harmless attack.
        self.targets: List[str] = list(targets or ())
        self.gossip = gossip
        self.config = config
        self.interval_ms = interval_ms
        self.rng = random.Random(config.seed)
        self.stats = AdversaryStats()
        self._running = False

        routers = [r.id for r in topology.routers]
        count = int(round(len(routers) * config.compromised_share))
        self.compromised: Set[str] = set(self.rng.sample(routers, min(count, len(routers))))
        self.stats.compromised = len(self.compromised)

        #: Every (variant, canonical) pair the attacker fabricated, so the
        #: harness can tell poison apart from an honest mistake.
        self.injected: Set[tuple] = set()

    def start(self) -> None:
        if not self.config.enabled or self.gossip is None or not self.compromised:
            return
        self._running = True
        self.gossip.sim.schedule(self.interval_ms, self._tick)

    def stop(self) -> None:
        self._running = False

    def _tick(self) -> None:
        if not self._running or self.gossip is None:
            return
        for router_id in self.compromised:
            agent = self.gossip.agents.get(router_id)
            if agent is not None:
                self._inject_mappings(agent)
                self._inject_evidence(agent)
        self.gossip.sim.schedule(self.interval_ms, self._tick)

    # -- the two attacks -------------------------------------------------

    def _inject_mappings(self, agent: GossipAgent) -> None:
        """Claim that real wordings are served by the wrong real routes.

        Deliberately plausible: both halves exist, so nothing structural marks
        the mapping as forged and only the producer's refusal can expose it.
        """
        router = agent.router
        prefixes = router.fib.prefixes
        if len(prefixes) < 2 or not self.targets:
            return

        for _ in range(self.config.false_mappings_per_round):
            variant = self.rng.choice(self.targets)
            canonical = self.rng.choice(prefixes)
            mapping = Mapping(
                variant=variant, canonical=canonical, score=0.99,
                version=agent.next_version(), origin=router.id,
            )
            agent.known[variant] = mapping
            self.injected.add((variant, canonical))
            self.stats.false_mappings_injected += 1

            for peer_id in router.peers:
                peer = self.gossip.agents.get(peer_id)
                if peer is None:
                    continue
                self.gossip.stats.mappings_sent += 1
                self.gossip.stats.bytes_sent += mapping.wire_bytes
                self.gossip.sim.schedule(
                    self.gossip._link_delay(router, peer_id),  # noqa: SLF001
                    self.gossip._receive, peer, [mapping],     # noqa: SLF001
                )

    def _inject_evidence(self, agent: GossipAgent) -> None:
        """Report successes that never happened, at scores that never worked.

        This is the attack the mapping defence does not cover: no route is named
        as wrong, so nothing gets refused and nothing gets retracted. The victim
        simply comes to believe that low scores are safe.
        """
        router = agent.router
        prefixes = router.fib.prefixes
        if not prefixes:
            return

        for _ in range(self.config.false_evidence_per_round):
            prefix = self.rng.choice(prefixes)
            observation = Observation(
                score=self.config.false_evidence_score,
                correct=True,
                at_ms=self.gossip.sim.now,
                source=router.id,
            )
            self.stats.false_evidence_injected += 1
            self.gossip.on_evidence(router, prefix, observation)

    # -- measuring the damage --------------------------------------------

    def audit(self) -> Dict[str, float]:
        """How much poison is still installed, and where."""
        victims = 0
        live = 0
        for router in self.topology.routers:
            if router.id in self.compromised:
                continue
            held = sum(
                1 for entry in router.es
                if (entry.variant, entry.canonical) in self.injected
            )
            if held:
                victims += 1
                live += held
        self.stats.victims_holding_poison = victims
        self.stats.poisoned_entries_live = live
        self.stats.poison_retracted = max(
            0, self.stats.false_mappings_injected - live
        )
        return self.stats.as_dict()
