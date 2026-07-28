"""The per-router tables: FIB, PIT, Content Store and Embedding Store.

Phase 2 of the project fixes an architectural rule that this module enforces:
semantic matching lives in the FIB only.  The Content Store and PIT stay strict
exact-match, because a semantic hit in the CS would hand a client data it did
not ask for, and a semantic hit in the PIT would return data down the wrong
chain.  Similarity is allowed to choose a *route*; it is never allowed to decide
that two names are the same object.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np


@dataclass
class FibEntry:
    """A route to a name prefix, with the embedding used to match it."""

    prefix: str
    face: str
    cost: float = 1.0
    embedding_index: int = -1


class Fib:
    """Forwarding Information Base with exact and longest-prefix lookup."""

    def __init__(self) -> None:
        self._entries: Dict[str, FibEntry] = {}
        self.lookups = 0
        self.exact_hits = 0

    def add(self, prefix: str, face: str, cost: float = 1.0) -> FibEntry:
        entry = FibEntry(prefix=prefix, face=face, cost=cost)
        self._entries[prefix] = entry
        return entry

    def remove(self, prefix: str) -> bool:
        return self._entries.pop(prefix, None) is not None

    def exact(self, name: str) -> Optional[FibEntry]:
        self.lookups += 1
        entry = self._entries.get(name)
        if entry is not None:
            self.exact_hits += 1
        return entry

    def longest_prefix(self, name: str) -> Optional[FibEntry]:
        """Standard NDN lookup: the longest registered prefix of ``name``."""
        self.lookups += 1
        components = name.strip("/").split("/")
        for length in range(len(components), 0, -1):
            candidate = "/" + "/".join(components[:length])
            entry = self._entries.get(candidate)
            if entry is not None:
                self.exact_hits += 1
                return entry
        return None

    def __contains__(self, prefix: str) -> bool:
        return prefix in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[FibEntry]:
        return iter(self._entries.values())

    @property
    def prefixes(self) -> List[str]:
        return list(self._entries)

    def get(self, prefix: str) -> Optional[FibEntry]:
        return self._entries.get(prefix)


@dataclass
class PitEntry:
    """One outstanding Interest, and everyone waiting on it."""

    name: str
    in_faces: Set[str] = field(default_factory=set)
    out_face: Optional[str] = None
    created_at: float = 0.0
    interest_ids: List[int] = field(default_factory=list)
    resolved_prefix: Optional[str] = None
    pending_learn: Optional["PendingMapping"] = None


@dataclass
class PendingMapping:
    """A speculative variant-to-canonical mapping awaiting confirmation.

    A semantic match is a guess.  In NDN the guess is checked for free: if the
    chosen producer really serves the name, Data comes back; if not, nothing
    does.  Promoting a mapping only once Data has returned is what makes it safe
    to share the mapping with other routers -- a router never gossips a route it
    has not seen work.
    """

    variant: str
    canonical: str
    face: str
    score: float
    signature: Optional[np.ndarray] = None


class Pit:
    """Pending Interest Table. Exact-match only, by architectural rule."""

    def __init__(self) -> None:
        self._entries: Dict[str, PitEntry] = {}
        self.aggregated = 0

    def find(self, name: str) -> Optional[PitEntry]:
        return self._entries.get(name)

    def insert(self, name: str, in_face: str, interest_id: int, now: float) -> Tuple[PitEntry, bool]:
        """Add an Interest. Returns the entry and whether it aggregated."""
        entry = self._entries.get(name)
        if entry is not None:
            entry.in_faces.add(in_face)
            entry.interest_ids.append(interest_id)
            self.aggregated += 1
            return entry, True
        entry = PitEntry(name=name, in_faces={in_face}, created_at=now, interest_ids=[interest_id])
        self._entries[name] = entry
        return entry, False

    def satisfy(self, name: str) -> Optional[PitEntry]:
        return self._entries.pop(name, None)

    def expire_before(self, deadline: float) -> List[PitEntry]:
        stale = [e for e in self._entries.values() if e.created_at < deadline]
        for entry in stale:
            self._entries.pop(entry.name, None)
        return stale

    def __len__(self) -> int:
        return len(self._entries)


class ContentStore:
    """LRU cache of Data packets. Exact-match only, by architectural rule."""

    def __init__(self, capacity: int = 0) -> None:
        self.capacity = capacity
        self._items: "OrderedDict[str, object]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, name: str) -> Optional[object]:
        if self.capacity <= 0:
            self.misses += 1
            return None
        item = self._items.get(name)
        if item is None:
            self.misses += 1
            return None
        self._items.move_to_end(name)
        self.hits += 1
        return item

    def put(self, name: str, data: object) -> None:
        if self.capacity <= 0:
            return
        self._items[name] = data
        self._items.move_to_end(name)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class EsEntry:
    """A learned mapping from a client's wording to a canonical route."""

    variant: str
    canonical: str
    face: str
    score: float
    signature: Optional[np.ndarray] = None
    confirmed: bool = False
    learned_at: float = 0.0
    source: str = "local"      # "local" or the id of the router that taught us
    hits: int = 0


class EmbeddingStore:
    """SAF's Embedding Store: an LRU cache of resolved names.

    Two things here are not in the friend's Phase 2 prototype and matter for the
    result to mean anything.

    *Bounded capacity with LRU eviction.*  An unbounded cache over a closed set
    of variant names converges to a 100% hit ratio by construction, which
    measures the workload rather than the design.  SAF specifies LRU; with a
    capacity smaller than the working set, the hit ratio becomes a property of
    locality and eviction policy.

    *Consistency with the FIB.*  When a route disappears, every mapping that
    pointed at it is invalidated; when a route appears, mappings may have a
    better target available.  SAF calls this out explicitly, and without it a
    long run silently accumulates mappings to producers that are gone.
    """

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._items: "OrderedDict[str, EsEntry]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        self.imported = 0

    def lookup(self, variant: str) -> Optional[EsEntry]:
        entry = self._items.get(variant)
        if entry is None:
            self.misses += 1
            return None
        self._items.move_to_end(variant)
        entry.hits += 1
        self.hits += 1
        return entry

    def peek(self, variant: str) -> Optional[EsEntry]:
        """Read without disturbing recency or hit statistics."""
        return self._items.get(variant)

    def store(self, entry: EsEntry) -> None:
        if self.capacity <= 0:
            return
        self._items[entry.variant] = entry
        self._items.move_to_end(entry.variant)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
            self.evictions += 1

    def confirm(self, variant: str, now: float) -> Optional[EsEntry]:
        """Mark a mapping as proven by a returned Data packet.

        Returns the entry only on the transition from unconfirmed to confirmed.
        Every subsequent Data for the same wording returns ``None``, which is
        what keeps a popular name from being re-announced to the gossip layer on
        every single hit.
        """
        entry = self._items.get(variant)
        if entry is None or entry.confirmed:
            return None
        entry.confirmed = True
        entry.learned_at = now
        return entry

    def drop(self, variant: str) -> None:
        self._items.pop(variant, None)

    def invalidate_route(self, canonical: str) -> int:
        """Remove every mapping that pointed at a now-unavailable route."""
        doomed = [v for v, e in self._items.items() if e.canonical == canonical]
        for variant in doomed:
            del self._items[variant]
        self.invalidations += len(doomed)
        return len(doomed)

    def confirmed_entries(self) -> List[EsEntry]:
        """The mappings this router is willing to teach others."""
        return [e for e in self._items.values() if e.confirmed]

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, variant: str) -> bool:
        return variant in self._items

    def __iter__(self) -> Iterator[EsEntry]:
        return iter(self._items.values())

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def memory_bytes(self) -> int:
        """Rough footprint: the names plus an 8-byte signature per entry.

        Embeddings themselves are not stored per mapping -- only the resolved
        route and its signature are needed to serve a hit, which is why this
        stays far below the 1.5 KB per cached embedding SAF reports.
        """
        return sum(len(e.variant) + len(e.canonical) + 24 for e in self._items.values())


def iter_names(entries: Iterable[EsEntry]) -> Iterator[str]:
    for entry in entries:
        yield entry.variant
