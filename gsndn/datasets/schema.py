"""Core data types shared by every name catalog.

A catalog is the ground truth of an experiment: the set of service names that
producers advertise (and that therefore end up in router FIBs), plus the set of
Interest names that clients actually emit.  Every Interest carries the canonical
service it *should* resolve to, or ``None`` when the correct behaviour is to
reject it.  Without that label a simulator can only measure how often it
forwarded something, not how often it forwarded something correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

# Interest name kinds.
EXACT = "exact"            # verbatim copy of a canonical service name
VARIANT = "variant"        # a semantically equivalent rewording
DISTRACTOR = "distractor"  # no service in the catalog can satisfy it


@dataclass(frozen=True)
class Service:
    """One IoT service, advertised by exactly one producer."""

    canonical: str
    category: str
    producer: str

    def __post_init__(self) -> None:
        if not self.canonical.startswith("/"):
            raise ValueError(f"service name must be absolute: {self.canonical!r}")


@dataclass(frozen=True)
class InterestName:
    """One name a client may ask for, with its ground-truth resolution."""

    name: str
    kind: str
    expected: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in (EXACT, VARIANT, DISTRACTOR):
            raise ValueError(f"unknown interest kind: {self.kind!r}")
        if self.kind == DISTRACTOR and self.expected is not None:
            raise ValueError("distractors must not have an expected service")
        if self.kind != DISTRACTOR and self.expected is None:
            raise ValueError(f"{self.kind} interests need an expected service")


@dataclass
class NameCatalog:
    """A complete evaluation dataset for one application domain."""

    domain: str
    services: List[Service]
    interests: List[InterestName]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical = {s.canonical for s in self.services}
        if len(canonical) != len(self.services):
            raise ValueError("duplicate canonical service names in catalog")
        seen: set[str] = set()
        for interest in self.interests:
            if interest.name in seen:
                raise ValueError(f"duplicate interest name: {interest.name!r}")
            seen.add(interest.name)
            if interest.expected is not None and interest.expected not in canonical:
                raise ValueError(
                    f"interest {interest.name!r} expects unknown service "
                    f"{interest.expected!r}"
                )

    # -- convenience views ------------------------------------------------

    @property
    def canonical_names(self) -> List[str]:
        return [s.canonical for s in self.services]

    @property
    def all_names(self) -> List[str]:
        """Every distinct string in the catalog, for batch embedding."""
        return self.canonical_names + [i.name for i in self.interests]

    def by_kind(self, kind: str) -> List[InterestName]:
        return [i for i in self.interests if i.kind == kind]

    def producer_of(self, canonical: str) -> str:
        for service in self.services:
            if service.canonical == canonical:
                return service.producer
        raise KeyError(canonical)

    def summary(self) -> Dict[str, int]:
        return {
            "services": len(self.services),
            "producers": len({s.producer for s in self.services}),
            "interests": len(self.interests),
            EXACT: len(self.by_kind(EXACT)),
            VARIANT: len(self.by_kind(VARIANT)),
            DISTRACTOR: len(self.by_kind(DISTRACTOR)),
        }


def dedupe_preserving_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def name_to_text(ndn_name: str) -> str:
    """Render a hierarchical NDN name as the plain text an encoder sees.

    ``/smart-hospital/building-a/temperature/room-101``
    becomes ``smart hospital building a temperature room 101``.
    """
    return " ".join(ndn_name.replace("/", " ").replace("-", " ").replace("_", " ").split())


def check_variant_budget(variants: Sequence[str], wanted: int, canonical: str) -> None:
    if len(variants) < wanted:
        raise ValueError(
            f"only {len(variants)} distinct variants generated for {canonical!r}, "
            f"need {wanted}; widen the lexicon"
        )
