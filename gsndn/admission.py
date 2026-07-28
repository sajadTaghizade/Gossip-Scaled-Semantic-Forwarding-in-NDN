"""What a producer knows about its own services, and how it uses it.

The weakest assumption in the first version of this work was that a producer
could tell whether a semantically-resolved Interest was really meant for it.
The simulation answered that question with the catalog's ground truth, which is
an oracle: it knows the right answer for every name in the network. Every
confirmation result rested on it, and a reviewer would be right to say the
feedback loop had been assumed rather than modelled.

This module replaces the oracle with something a real service actually has: a
**declared schema**. When a producer publishes ``/smart-hospital/.../temperature
/room-101`` it also declares the terms it answers to -- ``temp-sensor``,
``thermal-sensor``, and so on -- exactly as a service registry, an ontology
fragment or an OpenAPI description would. That declaration is local. A producer
knows its own aliases and nothing about anybody else's, which is the difference
between local knowledge and an oracle.

Two consequences follow, and both are the point rather than a limitation:

*Declarations are incomplete.* A producer that declares three of the six ways
clients word its service will refuse the other three, even though they were
meant for it. The feedback signal is therefore wrong sometimes, in the direction
that matters most -- a correct resolution reported as a failure.
``alias_coverage`` controls how incomplete.

*Declarations can be wrong.* ``noise`` flips a verdict outright, modelling a
misconfigured schema or a service that answers a request it should not have.

Everything downstream -- verification, calibration, gossip -- now has to work
with a feedback channel that is merely informative rather than correct, which is
the situation any deployment would actually be in.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Sequence, Set

from .datasets.schema import name_to_text


def tokens(ndn_name: str) -> Set[str]:
    """The bag of words a name reduces to, as a schema check would see it."""
    return set(name_to_text(ndn_name).split())


def _stable_fraction(key: str, salt: str) -> float:
    """A deterministic value in [0, 1) keyed on a string.

    Used so that which aliases a producer happens to declare is fixed by the
    scenario rather than by evaluation order, while still varying between
    services and between seeds.
    """
    digest = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


@dataclass
class ServiceSchema:
    """One service's public declaration of what it answers to."""

    canonical: str
    instance: str                       # e.g. "room-101" -- the thing measured
    declared: FrozenSet[str]            # metric terms the producer admits
    required_instance: bool = True

    def admits(self, requested_name: str) -> bool:
        """Does this request name one of my services?

        Both halves must match: a term I answer to, and the specific instance I
        am attached to. Requiring the instance is what stops a temperature
        producer for room 101 from happily answering for room 204 -- the failure
        mode that made the first version's precision meaningless.
        """
        words = tokens(requested_name)
        if self.required_instance and not tokens(self.instance) <= words:
            return False
        return any(tokens(term) <= words for term in self.declared)


@dataclass
class AdmissionPolicy:
    """The producer-side check, with its imperfections made explicit."""

    schemas: Dict[str, ServiceSchema] = field(default_factory=dict)

    #: Probability that a verdict is flipped, modelling a misconfigured schema.
    noise: float = 0.0
    #: Salt so noise is reproducible per scenario without a shared RNG.
    salt: str = "admission"

    accepted: int = 0
    refused: int = 0
    flipped: int = 0

    def admits(self, resolved_prefix: str, requested_name: str) -> bool:
        schema = self.schemas.get(resolved_prefix)
        if schema is None:
            # Not a name this producer publishes at all: an ordinary NDN miss,
            # nothing semantic about it.
            self.refused += 1
            return False

        verdict = schema.admits(requested_name)
        if self.noise > 0.0 and _stable_fraction(
            f"{resolved_prefix}|{requested_name}", self.salt
        ) < self.noise:
            verdict = not verdict
            self.flipped += 1

        if verdict:
            self.accepted += 1
        else:
            self.refused += 1
        return verdict

    @property
    def decisions(self) -> int:
        return self.accepted + self.refused


def build_schemas(
    catalog,
    names: Sequence[str],
    *,
    alias_coverage: float = 1.0,
    seed: int = 0,
) -> Dict[str, ServiceSchema]:
    """Derive declared schemas for the services a producer publishes.

    ``alias_coverage`` is the share of a service's known wordings that its
    operator bothered to declare. At 1.0 the producer recognises every rewording
    the catalog can generate and feedback is perfect; at 0.5 it recognises about
    half, and the network must cope with being told "no" for requests that were
    in fact correct.

    Which aliases survive is chosen by a stable hash rather than a sequential
    draw, so a service's declaration does not depend on the order producers were
    constructed in.
    """
    from .datasets import VARIANT

    #: canonical -> every wording the catalog can produce for it
    wordings: Dict[str, Set[str]] = {}
    for interest in catalog.by_kind(VARIANT):
        if interest.expected:
            wordings.setdefault(interest.expected, set()).add(interest.name)

    schemas: Dict[str, ServiceSchema] = {}
    for canonical in names:
        components = canonical.strip("/").split("/")
        instance = components[-1]
        metric = components[-2] if len(components) >= 2 else components[-1]

        # A service always answers to its own canonical metric term; the
        # question is how many of the alternatives it also declares.
        declared: Set[str] = {metric}
        for wording in sorted(wordings.get(canonical, ())):
            term = _metric_term(wording, instance)
            if term is None:
                continue
            if _stable_fraction(f"{canonical}|{term}", f"alias-{seed}") < alias_coverage:
                declared.add(term)

        schemas[canonical] = ServiceSchema(
            canonical=canonical, instance=instance, declared=frozenset(declared)
        )
    return schemas


def _metric_term(wording: str, instance: str) -> Optional[str]:
    """The component of a reworded name that names the measurement.

    Everything except the root authority, the structural components and the
    instance. Structural components are recognised by shape rather than by a
    list, so the same rule works for both domains.
    """
    components = [c for c in wording.strip("/").split("/") if c]
    instance_words = tokens(instance)
    candidates = [
        c for c in components[1:]
        if not (tokens(c) & instance_words)
        and c not in _STRUCTURAL
        and not _looks_structural(c)
    ]
    return candidates[-1] if candidates else None


#: Path components that describe where a service sits rather than what it is.
_STRUCTURAL = frozenset({
    "sensors", "devices", "monitoring", "readings", "services", "realtime",
    "live", "sensor",
})


def _looks_structural(component: str) -> bool:
    """Building/floor/zone-style components, matched by shape not by listing."""
    prefixes = ("building-", "bldg-", "floor-", "fl-", "district-", "zone-", "b-")
    return component.startswith(prefixes) or component in ("d5", "d9", "oldtown")
