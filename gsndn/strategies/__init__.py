"""Forwarding strategies, and the registry experiments select them by."""

from __future__ import annotations

from typing import Optional

from ..costs import CostModel
from .base import ForwardingStrategy
from .sef import SefQLearning
from .risk_controlled import RiskControlledNdn
from .semantic import GsNdn, Saf, SafWithEs, VanillaNdn

#: Registry keys used on the command line and in result tables.
STRATEGIES = (
    "vanilla-ndn",
    "saf",
    "saf+es",
    "gs-ndn",
    "sef",
    # Ablations, each removing one of GS-NDN's three mechanisms so the
    # contribution of that mechanism is attributable rather than assumed.
    "gs-ndn-no-gossip",
    "gs-ndn-no-verify",
    # Risk-controlled forwarding and its own ablations: no shared evidence
    # (each router calibrates alone) and no mapping gossip.
    "rc-ndn",
    # The same controller with an adaptive rather than fixed budget.
    "rc-ndn-aci",
    # ... and the same controller with the cross-encoder guard removed, as the
    # naive baseline the heterogeneity result is measured against.
    "rc-ndn-naive-mix",
    "rc-ndn-no-evidence",
    "rc-ndn-no-gossip",
    "rc-ndn-no-explore",
)


def build_strategy(
    name: str,
    *,
    threshold: float = 0.7,
    costs: Optional[CostModel] = None,
    seed: int = 0,
    epsilon: float = 0.05,
    confidence: float = 0.9,
    explore_rate: float = 0.05,
    adapt_rate: float = 0.05,
) -> ForwardingStrategy:
    """Instantiate a strategy by registry key."""
    if name == "vanilla-ndn":
        return VanillaNdn(costs)
    if name == "saf":
        return Saf(threshold, costs)
    if name == "saf+es":
        return SafWithEs(threshold, costs)
    if name == "sef":
        return SefQLearning(costs, seed=seed)
    if name == "gs-ndn":
        return GsNdn(threshold, costs)
    if name == "gs-ndn-no-gossip":
        return GsNdn(threshold, costs, gossip=False)
    if name == "gs-ndn-no-verify":
        return GsNdn(threshold, costs, verify=False)
    if name == "rc-ndn":
        return RiskControlledNdn(
            threshold, costs, epsilon=epsilon, confidence=confidence,
            explore_rate=explore_rate,
        )
    if name == "rc-ndn-aci":
        # Adaptive Conformal Inference: the budget itself moves in response to
        # realised coverage, so a distribution that shifts under the controller
        # is answered rather than assumed away.
        strategy = RiskControlledNdn(
            threshold, costs, epsilon=epsilon, confidence=confidence,
            explore_rate=explore_rate, adaptive=True, adapt_rate=adapt_rate,
        )
        strategy.name = "rc-ndn-aci"
        return strategy
    if name == "rc-ndn-naive-mix":
        # The naive baseline for the heterogeneity experiment: gossip evidence
        # across an encoder boundary as though a MiniLM 0.62 and an n-gram 0.62
        # were the same number. Note this has to be a risk-controlled arm --
        # gs-ndn gossips no scores at all, so there is nothing for it to mix.
        strategy = RiskControlledNdn(
            threshold, costs, epsilon=epsilon, confidence=confidence,
            explore_rate=explore_rate, mix_across_encoders=True,
        )
        strategy.name = "rc-ndn-naive-mix"
        return strategy
    if name == "rc-ndn-no-explore":
        return RiskControlledNdn(
            threshold, costs, epsilon=epsilon, confidence=confidence, explore_rate=0.0
        )
    if name == "rc-ndn-no-evidence":
        return RiskControlledNdn(
            threshold, costs, epsilon=epsilon, confidence=confidence,
            explore_rate=explore_rate, share_evidence=False,
        )
    if name == "rc-ndn-no-gossip":
        return RiskControlledNdn(
            threshold, costs, epsilon=epsilon, confidence=confidence,
            explore_rate=explore_rate, gossip=False, share_evidence=False,
        )
    raise ValueError(f"unknown strategy {name!r}; expected one of {STRATEGIES}")


__all__ = [
    "ForwardingStrategy",
    "GsNdn",
    "STRATEGIES",
    "Saf",
    "RiskControlledNdn",
    "SafWithEs",
    "SefQLearning",
    "VanillaNdn",
    "build_strategy",
]
