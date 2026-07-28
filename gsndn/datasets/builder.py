"""Deterministic catalog construction from a hand-authored lexicon.

Both evaluation domains follow the shape used by SAF (Amadeo et al., IEEE IoT
Magazine 2026): every service contributes one exact Interest name plus a fixed
number of synonymous rewordings.  Rather than typing out several hundred names
by hand we compose them from an explicit lexicon of domain synonyms and
structural rewrites, so the dataset is reproducible, auditable and easy to grow.

Six rewrite families are applied per service, mirroring the ways real IoT
namespaces actually diverge:

1. ``synonym``      -- different root authority and a different word for the metric
2. ``abbreviated``  -- the same hierarchy with abbreviated path components
3. ``flattened``    -- intermediate hierarchy (building/floor) dropped entirely
4. ``alt_synonym``  -- a second, more distant wording of the metric
5. ``reordered``    -- instance placed before the metric
6. ``verbose``      -- a longer, more descriptive phrasing

Each family draws from the lexicon with a deterministic rotation keyed on the
service index, so distinct services get distinct wordings while the whole
catalog is a pure function of the lexicon.

A seventh family, ``ontology``, is available and off by default:

7. ``ontology``     -- the metric named with a class name transcribed from
                       Brick Schema, SAREF, Project Haystack or W3C SSN/SOSA

It exists because the six above are ours, and a reviewer is entitled to ask
whether the recognition results measure the method or measure our lexicon. See
:mod:`gsndn.datasets.ontology` for what it does and does not fix. It is off by
default deliberately: switching it on changes the catalog, and therefore every
number in ``RESULTS.md``, so the grounded catalogs are separate domains
(``hospital-grounded``, ``city-grounded``) measured separately rather than a
silent redefinition of the ones the campaign already ran on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import ontology
from .schema import (
    DISTRACTOR,
    EXACT,
    VARIANT,
    InterestName,
    NameCatalog,
    Service,
    check_variant_budget,
)

VARIANTS_PER_SERVICE = 6


@dataclass(frozen=True)
class MetricFamily:
    """One measurable quantity and the words people use for it.

    ``synonyms[0]`` is canonical; the rest are what clients might type instead.
    """

    key: str
    synonyms: Sequence[str]

    def __post_init__(self) -> None:
        if len(self.synonyms) < 5:
            raise ValueError(f"metric {self.key!r} needs at least 5 synonyms")

    @property
    def canonical(self) -> str:
        return self.synonyms[0]

    def pick(self, offset: int) -> str:
        """Choose a non-canonical synonym by deterministic rotation."""
        alternatives = self.synonyms[1:]
        return alternatives[offset % len(alternatives)]


@dataclass(frozen=True)
class Placement:
    """Where a service physically sits, and how that location is abbreviated."""

    group: str           # e.g. "building-a"   / "district-5"
    group_short: str     # e.g. "bldg-a"       / "d5"
    sub: str             # e.g. "floor-1"      / "zone-north"
    sub_short: str       # e.g. "fl-1"         / "north"
    instance: str        # e.g. "room-101"     / "junction-a12"


@dataclass(frozen=True)
class DomainLexicon:
    """Everything needed to generate one domain's catalog."""

    domain: str
    root: str                      # canonical root authority component
    root_synonyms: Sequence[str]   # alternative root authorities
    metrics: Sequence[MetricFamily]
    placements: Sequence[Placement]
    verbose_templates: Sequence[str]
    distractor_roots: Sequence[Tuple[str, Sequence[str], Sequence[str]]]

    def __post_init__(self) -> None:
        if len(self.root_synonyms) < 3:
            raise ValueError("need at least 3 root synonyms")
        if not self.verbose_templates:
            raise ValueError("need at least one verbose template")


def _join(*components: str) -> str:
    parts = [c.strip("/") for c in components if c]
    return "/" + "/".join(parts)


# --- the six rewrite families ------------------------------------------------


def _synonym(lex: DomainLexicon, m: MetricFamily, p: Placement, i: int) -> str:
    root = lex.root_synonyms[i % len(lex.root_synonyms)]
    return _join(root, m.pick(i), p.instance)


def _abbreviated(lex: DomainLexicon, m: MetricFamily, p: Placement, i: int) -> str:
    root = lex.root_synonyms[(i + 1) % len(lex.root_synonyms)]
    return _join(root, p.group_short, p.sub_short, m.pick(i + 1), p.instance)


def _flattened(lex: DomainLexicon, m: MetricFamily, p: Placement, i: int) -> str:
    return _join(lex.root, m.pick(i + 2), p.instance)


def _alt_synonym(lex: DomainLexicon, m: MetricFamily, p: Placement, i: int) -> str:
    root = lex.root_synonyms[(i + 2) % len(lex.root_synonyms)]
    return _join(root, p.group_short, m.pick(i + 3), p.instance)


def _reordered(lex: DomainLexicon, m: MetricFamily, p: Placement, i: int) -> str:
    root = lex.root_synonyms[(i + 3) % len(lex.root_synonyms)]
    return _join(root, p.instance, m.pick(i + 4))


def _verbose(lex: DomainLexicon, m: MetricFamily, p: Placement, i: int) -> str:
    template = lex.verbose_templates[i % len(lex.verbose_templates)]
    return template.format(
        root=lex.root_synonyms[(i + 4) % len(lex.root_synonyms)],
        metric=m.pick(i + 5),
        group=p.group,
        group_short=p.group_short,
        sub=p.sub,
        sub_short=p.sub_short,
        instance=p.instance,
    )


def _ontology(lex: DomainLexicon, m: MetricFamily, p: Placement, i: int) -> str:
    """Name the metric with a class name somebody else standardised.

    Falls back to the invented lexicon when no published vocabulary names this
    quantity -- which is the case for every clinical metric family, since Brick,
    SAREF, Haystack and SSN/SOSA are building and sensing vocabularies and none
    of them models a glucose monitor. The fallback is counted separately in the
    catalog metadata so the grounded share is never overstated.
    """
    term = ontology.pick(m.key, i)
    metric = term.term if term is not None else m.pick(i + 6)
    root = lex.root_synonyms[(i + 5) % len(lex.root_synonyms)]
    return _join(root, p.group_short, metric, p.instance)


REWRITES: Sequence[Tuple[str, Callable[[DomainLexicon, MetricFamily, Placement, int], str]]] = (
    ("synonym", _synonym),
    ("abbreviated", _abbreviated),
    ("flattened", _flattened),
    ("alt_synonym", _alt_synonym),
    ("reordered", _reordered),
    ("verbose", _verbose),
)

#: Appended to ``REWRITES`` only when a catalog is built grounded.
ONTOLOGY_REWRITE: Tuple[str, Callable[[DomainLexicon, MetricFamily, Placement, int], str]] = (
    "ontology", _ontology,
)


def build_catalog(
    lex: DomainLexicon,
    n_distractors: Optional[int] = None,
    *,
    grounded: bool = False,
) -> NameCatalog:
    """Compose a full catalog: services, exact names, variants and distractors.

    ``n_distractors`` defaults to the number of resolvable Interest names, which
    reproduces SAF's harder configuration where only half of the offered traffic
    corresponds to an actually available service.
    """
    rewrites = tuple(REWRITES) + ((ONTOLOGY_REWRITE,) if grounded else ())
    wanted_variants = VARIANTS_PER_SERVICE + (1 if grounded else 0)

    services: List[Service] = []
    interests: List[InterestName] = []
    rewrite_counts: Dict[str, int] = {name: 0 for name, _ in rewrites}
    family: Dict[str, str] = {}
    ontology_terms = 0

    index = 0
    for metric in lex.metrics:
        for placement in lex.placements:
            canonical = _join(
                lex.root, placement.group, placement.sub,
                metric.canonical, placement.instance,
            )
            services.append(
                Service(
                    canonical=canonical,
                    category=metric.key,
                    producer=f"{metric.key}@{placement.group}",
                )
            )
            interests.append(InterestName(canonical, EXACT, canonical))

            generated: List[str] = []
            for label, rewrite in rewrites:
                candidate = rewrite(lex, metric, placement, index)
                if candidate != canonical and candidate not in generated:
                    generated.append(candidate)
                    rewrite_counts[label] += 1
                    # Which family produced a name is not recoverable from the
                    # name itself, and the grounded experiment has to compare
                    # families against each other, so record it here.
                    family[candidate] = label
                    if label == "ontology":
                        if ontology.pick(metric.key, index) is not None:
                            ontology_terms += 1
                        else:
                            family[candidate] = "ontology-fallback"
            check_variant_budget(generated, wanted_variants, canonical)
            for variant in generated[:wanted_variants]:
                interests.append(InterestName(variant, VARIANT, canonical))
            index += 1

    resolvable = len(interests)
    wanted = resolvable if n_distractors is None else n_distractors
    interests.extend(_build_distractors(lex, wanted, taken={i.name for i in interests}))

    metadata: Dict[str, object] = {
        "variants_per_service": wanted_variants,
        "rewrite_counts": rewrite_counts,
        "resolvable_fraction": round(resolvable / len(interests), 4),
        "grounded": grounded,
        "rewrite_family": family,
    }
    if grounded:
        # Carried on the catalog so that any result computed from it states its
        # own grounded share rather than relying on a claim made elsewhere.
        metadata["ontology"] = {
            "terms_from_standards": ontology_terms,
            "invented_fallbacks": rewrite_counts.get("ontology", 0) - ontology_terms,
            **ontology.coverage([m.key for m in lex.metrics]),
        }

    return NameCatalog(
        domain=lex.domain,
        services=services,
        interests=interests,
        metadata=metadata,
    )


def _build_distractors(
    lex: DomainLexicon, wanted: int, taken: set[str]
) -> List[InterestName]:
    """Names from unrelated domains that no service in the catalog can satisfy.

    These are what make precision measurable.  Some are deliberately near-misses
    (weather temperature against hospital room temperature, for instance) so the
    similarity threshold is forced to discriminate rather than just accept
    everything, which is exactly the regime SAF's threshold tuning explored.
    """
    if not lex.distractor_roots or wanted <= 0:
        return []

    # Full cross product per root, then round-robin across roots so that a
    # truncated distractor set still spans every unrelated domain evenly.
    per_root: List[List[str]] = []
    for root, topics, instances in lex.distractor_roots:
        per_root.append(
            [_join(root, topic, instance) for instance in instances for topic in topics]
        )

    available = sum(len(names) for names in per_root)
    if available < wanted:
        raise ValueError(
            f"distractor lexicon yields only {available} names, need {wanted}; "
            f"add roots, topics or instances to the domain lexicon"
        )

    out: List[InterestName] = []
    for column in range(max(len(names) for names in per_root)):
        for names in per_root:
            if column >= len(names):
                continue
            name = names[column]
            if name in taken:
                continue
            taken.add(name)
            out.append(InterestName(name, DISTRACTOR, None))
            if len(out) == wanted:
                return out
    return out
