"""Evaluation name catalogs for the two application domains."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from . import city, hospital
from .schema import (
    DISTRACTOR,
    EXACT,
    VARIANT,
    InterestName,
    NameCatalog,
    Service,
    name_to_text,
)

from functools import partial

_BUILDERS: Dict[str, Callable[[Optional[int]], NameCatalog]] = {
    "hospital": hospital.catalog,
    "city": city.catalog,
    # The same two domains with a seventh rewording per service taken from a
    # published ontology where one exists. Separate domains rather than a flag
    # on the originals, because switching it on changes the catalog and
    # therefore every number the campaign has already reported; see
    # :mod:`gsndn.datasets.ontology`.
    "hospital-grounded": partial(hospital.catalog, grounded=True),
    "city-grounded": partial(city.catalog, grounded=True),
}

#: The domains the reported campaign runs on. Kept to the two originals so that
#: adding a grounded variant does not silently double every sweep.
DOMAINS = ("hospital", "city")
GROUNDED_DOMAINS = ("hospital-grounded", "city-grounded")
ALL_DOMAINS = tuple(_BUILDERS)


def load(domain: str, n_distractors: Optional[int] = None) -> NameCatalog:
    """Build the catalog for ``domain`` -- see :data:`ALL_DOMAINS`."""
    try:
        builder = _BUILDERS[domain]
    except KeyError:
        raise ValueError(
            f"unknown domain {domain!r}; expected one of {ALL_DOMAINS}"
        ) from None
    return builder(n_distractors)


__all__ = [
    "ALL_DOMAINS",
    "DISTRACTOR",
    "DOMAINS",
    "GROUNDED_DOMAINS",
    "EXACT",
    "VARIANT",
    "InterestName",
    "NameCatalog",
    "Service",
    "city",
    "hospital",
    "load",
    "name_to_text",
]
