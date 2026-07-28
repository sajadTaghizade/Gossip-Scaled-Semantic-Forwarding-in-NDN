"""Forwarding strategies, and the registry experiments select them by."""

from __future__ import annotations

from typing import Optional

from ..costs import CostModel
from .base import ForwardingStrategy
from .sef import SefQLearning
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
    "gs-ndn-no-edge-tag",
)


def build_strategy(
    name: str,
    *,
    threshold: float = 0.7,
    costs: Optional[CostModel] = None,
    seed: int = 0,
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
    if name == "gs-ndn-no-edge-tag":
        return GsNdn(threshold, costs, edge_only=False)
    raise ValueError(f"unknown strategy {name!r}; expected one of {STRATEGIES}")


__all__ = [
    "ForwardingStrategy",
    "GsNdn",
    "STRATEGIES",
    "Saf",
    "SafWithEs",
    "SefQLearning",
    "VanillaNdn",
    "build_strategy",
]
