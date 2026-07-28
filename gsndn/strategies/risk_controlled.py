"""Risk-controlled semantic forwarding.

Everything GS-NDN does, with the similarity threshold taken out and replaced by
a per-route decision boundary that is learned from the network's own feedback
and held to an operator-chosen error budget. See :mod:`gsndn.risk` for why a
single global threshold cannot serve every route, and for the relation to
verified semantic caching and federated conformal prediction.

The forwarding path changes in one place. Where GS-NDN asks

    is the best score above 0.7?

this asks

    is the best score above the boundary this route has earned, given that at
    most epsilon of decisions on it may be wrong?

Two consequences are worth stating before the results are read.

*The encoder still runs.* Calibration decides whether to trust a resolution, not
whether to compute one, so the inference savings come from the same caching and
gossip as before. What changes is which resolutions get acted on.

*A route can end up refusing everything.* If no cutoff on a route's observations
meets the budget, the boundary goes above 1.0 and the route stops being
selected. That is the correct behaviour for a guarantee and it costs recall; the
alternative -- quietly reverting to the prior -- would make the budget a
suggestion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..costs import CostModel
from ..packets import (
    OUTCOME_ES_HIT,
    OUTCOME_GOSSIP_HIT,
    OUTCOME_NACK,
    OUTCOME_SEMANTIC,
    SemanticTag,
)
from ..risk import Observation, RiskController
from ..tables import EsEntry, PendingMapping
from .semantic import GsNdn

if TYPE_CHECKING:  # pragma: no cover
    from ..network import Resolution, Router


class RiskControlledNdn(GsNdn):
    """GS-NDN with a learned, per-route decision boundary."""

    name = "rc-ndn"

    def __init__(
        self,
        threshold: float = 0.6,
        costs: Optional[CostModel] = None,
        *,
        epsilon: float = 0.05,
        confidence: float = 0.9,
        explore_rate: float = 0.05,
        verify: bool = True,
        gossip: bool = True,
        share_evidence: bool = True,
        adaptive: bool = False,
        adapt_rate: float = 0.05,
        mix_across_encoders: bool = False,
    ) -> None:
        # The threshold survives only as the prior for routes with no evidence,
        # which is what makes this strictly a superset of the fixed-threshold
        # behaviour rather than a different system.
        super().__init__(threshold, costs, verify=verify, gossip=gossip)
        self.epsilon = epsilon
        #: Whether calibration observations travel between routers. Off for the
        #: ablation that separates sharing answers from sharing evidence.
        self.share_evidence = share_evidence
        #: Pool scores across an encoder boundary anyway. Wrong by construction
        #: -- two encoders' cosine scores are different measurements -- and here
        #: only so that refusing to pool has something to be measured against.
        self.mix_across_encoders = mix_across_encoders

        #: One controller per router: boundaries are local state, even when the
        #: evidence behind them came from elsewhere.
        self.controllers: dict[str, RiskController] = {}
        self._defaults = dict(
            epsilon=epsilon, prior=threshold, confidence=confidence,
            explore_rate=explore_rate, adaptive=adaptive, adapt_rate=adapt_rate,
        )
        self.blocked_by_risk = 0
        self.explorations = 0

    def controller(self, router: "Router") -> RiskController:
        controller = self.controllers.get(router.id)
        if controller is None:
            controller = RiskController(**self._defaults)
            self.controllers[router.id] = controller
        return controller

    def attach(self, router: "Router") -> None:
        self.controller(router)

    # -- forwarding ------------------------------------------------------

    def resolve(self, router: "Router", interest: "Interest") -> "Resolution":
        from ..network import Resolution

        tagged = self._tagged(router, interest)
        if tagged is not None:
            return tagged
        if interest.tag is not None:
            self.tag_misses += 1

        cached = router.es.lookup(interest.name)
        if cached is not None and (self.gossip or cached.source == "local"):
            outcome = OUTCOME_ES_HIT if cached.source == "local" else OUTCOME_GOSSIP_HIT
            if cached.source == "local":
                self.es_hits += 1
            else:
                self.gossip_hits += 1
            return Resolution(
                outcome=outcome, canonical=cached.canonical, face=cached.face,
                score=cached.score, cpu_ms=self.costs.es_lookup_ms,
                tag=SemanticTag(cached.canonical, cached.score, router.id),
            )

        refuted = self.refuted[router.id].get(interest.name) if self.verify else None
        if refuted:
            self.reresolutions += 1

        # The candidate is chosen without a threshold -- take the best route the
        # encoder offers, then ask whether this route has earned the right to be
        # used at that score.
        canonical, score, cpu_ms = self._best_candidate(
            router, interest.name, exclude=refuted
        )
        cpu_ms += self.costs.es_lookup_ms
        if canonical is None:
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=cpu_ms)

        controller = self.controller(router)
        allowed, exploring = controller.allows(canonical, score)
        if not allowed:
            self.blocked_by_risk += 1
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=cpu_ms)
        if exploring:
            self.explorations += 1

        face = self._face_for(router, canonical)
        if face is None:
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=cpu_ms)
        if refuted:
            self.recoveries += 1

        mapping = PendingMapping(
            variant=interest.name, canonical=canonical, face=face, score=score
        )
        entry = EsEntry(
            variant=interest.name, canonical=canonical, face=face, score=score,
            confirmed=not self.verify, learned_at=router.sim.now, source="local",
        )
        router.es.store(entry)
        if not self.verify and self.gossip and router.gossip_protocol is not None:
            router.gossip_protocol.on_confirmed(router, entry)

        return Resolution(
            outcome=OUTCOME_SEMANTIC, canonical=canonical, face=face, score=score,
            cpu_ms=cpu_ms, tag=SemanticTag(canonical, score, router.id),
            learn=mapping if self.verify else None,
        )

    def _best_candidate(self, router: "Router", name: str, exclude=None):
        """The encoder's best route and its score, with no threshold applied.

        A fixed threshold would decide twice -- once here and once in the risk
        controller -- and the tighter of the two would silently win, making the
        error budget unmeasurable.
        """
        index = router.name_index
        cpu_ms = self.costs.semantic_resolve_ms(len(index) if index else 0)
        self.encoder_runs += 1
        router.encoder_runs += 1
        if index is None or len(index) == 0:
            return None, 0.0, cpu_ms
        query = index.backend.encode([name])[0]
        match = index.best_match(query, -1.0, exclude=exclude)
        if match is None:
            return None, 0.0, cpu_ms
        canonical, score = match
        return canonical, score, cpu_ms

    # -- learning --------------------------------------------------------

    def on_confirmed(self, router: "Router", mapping: PendingMapping, now: float) -> None:
        self._record(router, mapping, correct=True, now=now)
        super().on_confirmed(router, mapping, now)

    def on_rejected(self, router: "Router", mapping: PendingMapping) -> None:
        self._record(router, mapping, correct=False, now=router.sim.now)
        super().on_rejected(router, mapping)

    def _record(self, router: "Router", mapping: PendingMapping, correct: bool, now: float) -> None:
        """Turn one answered forwarding decision into calibration evidence."""
        observation = Observation(
            score=mapping.score, correct=correct, at_ms=now, source="local"
        )
        self.controller(router).observe(mapping.canonical, observation)
        if self.share_evidence and router.gossip_protocol is not None:
            router.gossip_protocol.on_evidence(router, mapping.canonical, observation)

    # -- reporting -------------------------------------------------------

    def stats(self) -> dict:
        data = super().stats()
        data["blocked_by_risk"] = self.blocked_by_risk
        data["explorations"] = self.explorations
        if not self.controllers:
            return data

        merged: dict[str, float] = {}
        for controller in self.controllers.values():
            for key, value in controller.stats().items():
                merged[key] = merged.get(key, 0.0) + float(value)
        count = len(self.controllers)
        for key in (
            "risk_epsilon", "risk_insample_error", "risk_boundary_global",
            "risk_boundary_mean", "risk_boundary_std", "risk_boundary_range",
        ):
            merged[key] /= count
        data.update(merged)
        return data
