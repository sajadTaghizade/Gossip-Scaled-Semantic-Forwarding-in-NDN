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

_BUILDERS: Dict[str, Callable[[Optional[int]], NameCatalog]] = {
    "hospital": hospital.catalog,
    "city": city.catalog,
}

DOMAINS = tuple(_BUILDERS)


def load(domain: str, n_distractors: Optional[int] = None) -> NameCatalog:
    """Build the catalog for ``domain`` ("hospital" or "city")."""
    try:
        builder = _BUILDERS[domain]
    except KeyError:
        raise ValueError(
            f"unknown domain {domain!r}; expected one of {DOMAINS}"
        ) from None
    return builder(n_distractors)


__all__ = [
    "DISTRACTOR",
    "DOMAINS",
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
