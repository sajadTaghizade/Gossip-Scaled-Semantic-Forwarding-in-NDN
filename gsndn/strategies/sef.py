"""SEF: energy-aware next-hop selection by Q-learning.

A reduced implementation of Askar et al., CMC 2024, which forwards toward
whichever neighbour has the best combination of progress toward the producer
and remaining battery, learned online.  In the paper this replaces the FIB
outright with lightweight Neighbours and Producers tables; here the FIB stays
and SEF overrides only the choice of outgoing face, which is the part that
determines its energy behaviour and is what can be compared meaningfully
against the other strategies on the same topology.

One thing to be clear about when reading the results: **SEF has no semantic
capability**.  A name absent from the FIB is dropped, exactly as in Vanilla NDN.
It is included as a baseline on the energy and forwarding-efficiency axis, not
on the name-resolution axis, and reporting it as though it competed on the
latter would misrepresent the paper.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from ..costs import CostModel
from ..packets import OUTCOME_NACK
from .base import ForwardingStrategy

if TYPE_CHECKING:  # pragma: no cover
    from ..network import Resolution, Router
    from ..packets import Interest


class SefQLearning(ForwardingStrategy):
    """Q-learning over (destination prefix, neighbour) pairs.

    The reward follows the paper's two terms: progress toward the destination,
    and the neighbour's remaining energy.  Draining a neighbour makes it a worse
    choice, which is what spreads load and extends network lifetime.
    """

    name = "sef"
    semantic = False

    def __init__(
        self,
        costs: Optional[CostModel] = None,
        *,
        alpha: float = 0.3,
        gamma: float = 0.8,
        epsilon: float = 0.1,
        energy_weight: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__(costs)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.energy_weight = energy_weight
        self.rng = random.Random(seed)
        self.q: Dict[Tuple[str, str, str], float] = defaultdict(float)
        self.updates = 0
        self.explorations = 0

    def resolve(self, router: "Router", interest: "Interest") -> "Resolution":
        from ..network import Resolution

        # No semantic layer: an unknown name is dropped, as in standard NDN.
        return Resolution(outcome=OUTCOME_NACK, cpu_ms=0.0)

    def select_face(self, router: "Router", interest: "Interest", entry) -> str:
        """Pick a neighbour by Q-value, occasionally exploring.

        Only faces that make progress toward the destination are eligible, so
        exploration cannot send a packet backwards -- the paper's Next-Hop-ID
        field serves the same purpose of suppressing pointless transmissions.
        """
        candidates = self._eligible_faces(router, entry)
        if len(candidates) <= 1:
            return entry.face

        if self.rng.random() < self.epsilon:
            self.explorations += 1
            chosen = self.rng.choice(candidates)
        else:
            chosen = max(candidates, key=lambda f: self.q[(router.id, entry.prefix, f)])

        self._update(router, entry.prefix, chosen)
        return chosen

    def _eligible_faces(self, router: "Router", entry) -> list[str]:
        """Neighbours strictly closer to the producer than this router is.

        Merely having a route to the prefix is not enough -- every router does,
        so that test admits the neighbour we just came from and lets exploration
        walk an Interest in circles until it times out. Comparing the installed
        route costs, which carry hop distance to the producer, keeps every
        candidate a step forward. This is the job SEF's Next-Hop-ID field does in
        the paper: suppress transmissions that cannot make progress.
        """
        network = router.network
        if network is None:
            return [entry.face]
        eligible = [entry.face]
        for peer in router.peers:
            if peer == entry.face:
                continue
            neighbour = network.nodes.get(peer)
            peer_fib = getattr(neighbour, "fib", None)
            if peer_fib is None:
                continue
            peer_entry = peer_fib.get(entry.prefix)
            if peer_entry is not None and peer_entry.cost < entry.cost:
                eligible.append(peer)
        return eligible

    def _update(self, router: "Router", prefix: str, face: str) -> None:
        network = router.network
        neighbour = network.nodes.get(face) if network else None
        energy = getattr(neighbour, "energy_fraction", 1.0)
        progress = 1.0

        reward = progress + self.energy_weight * float(energy)
        key = (router.id, prefix, face)
        best_next = max(
            (self.q[(face, prefix, f)] for f in getattr(neighbour, "peers", []) or [face]),
            default=0.0,
        )
        self.q[key] += self.alpha * (reward + self.gamma * best_next - self.q[key])
        self.updates += 1

    def stats(self) -> dict:
        data = super().stats()
        data.update({
            "q_entries": len(self.q),
            "q_updates": self.updates,
            "explorations": self.explorations,
        })
        return data
