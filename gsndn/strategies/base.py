"""The one extension point a forwarding strategy gets.

Routers handle Content Store, PIT and exact FIB lookups identically no matter
which strategy is loaded.  A strategy is consulted at exactly one moment -- the
FIB miss -- and is told about the outcome afterwards.  Everything that
distinguishes Vanilla NDN from SAF from GS-NDN lives behind these three methods,
which is what makes a difference in the results attributable to policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..costs import CostModel
from ..packets import OUTCOME_NACK
from ..tables import PendingMapping

if TYPE_CHECKING:  # pragma: no cover
    from ..network import Resolution, Router
    from ..packets import Interest


class ForwardingStrategy:
    """Base class: reject everything, cost nothing."""

    name = "base"

    #: Whether this strategy can resolve a name the FIB does not contain.
    semantic = False

    def __init__(self, costs: Optional[CostModel] = None) -> None:
        self.costs = costs or CostModel()
        self.encoder_runs = 0
        self.es_hits = 0
        self.gossip_hits = 0
        self.tag_hits = 0
        self.confirmations = 0
        self.retractions = 0

    def resolve(self, router: "Router", interest: "Interest") -> "Resolution":
        """Decide what to do with an Interest that missed the FIB."""
        from ..network import Resolution

        return Resolution(outcome=OUTCOME_NACK, cpu_ms=0.0)

    def select_face(self, router: "Router", interest: "Interest", entry) -> str:
        """Choose the outgoing face for an Interest that *did* match the FIB.

        Almost every strategy takes the route the routing protocol installed.
        The reinforcement-learning baseline overrides this, because choosing
        among next hops by learned cost is the whole of what it does.
        """
        return entry.face

    def on_confirmed(self, router: "Router", mapping: PendingMapping, now: float) -> None:
        """Data came back: the speculative mapping was right."""
        self.confirmations += 1

    def on_rejected(self, router: "Router", mapping: PendingMapping) -> None:
        """A Nack came back: the speculative mapping was wrong."""
        self.retractions += 1

    def attach(self, router: "Router") -> None:
        """Hook for per-router setup once the topology exists."""

    def stats(self) -> dict:
        return {
            "encoder_runs": self.encoder_runs,
            "es_hits": self.es_hits,
            "gossip_hits": self.gossip_hits,
            "tag_hits": self.tag_hits,
            "confirmations": self.confirmations,
            "retractions": self.retractions,
        }
