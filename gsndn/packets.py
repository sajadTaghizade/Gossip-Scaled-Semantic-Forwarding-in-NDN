"""Interest, Data and Nack, plus the semantic tag an Interest may carry.

The only departure from stock NDN is ``SemanticTag``.  When an edge router
resolves a name semantically it annotates the Interest with the canonical
prefix it settled on and a compact signature of the name's embedding.  Core
routers downstream match on that annotation instead of re-running an encoder,
which is what keeps the expensive work at the edge -- the arrangement SAF
approximates by prepending the discovered FIB name with a delimiter, and what
the project's Phase 2 deck calls edge tagging.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional

_interest_ids = itertools.count(1)

#: Wire sizes in bytes, used by the energy model.  An NDN Interest is name plus
#: framing; the tag adds a fixed 8-byte signature and the resolved prefix.
INTEREST_OVERHEAD_BYTES = 40
DATA_OVERHEAD_BYTES = 60
DATA_PAYLOAD_BYTES = 256


@dataclass(frozen=True)
class SemanticTag:
    """An edge router's resolution decision, carried with the Interest.

    The tag names the canonical prefix outright rather than a signature of it.
    That is a measured choice: signature matching picks the right FIB entry only
    a third of the time at 64 bits and 84% at 1024 (see
    ``experiments/bench_simhash.py``), because service names within a domain sit
    too close together for locality-sensitive hashing to separate.  Carrying the
    prefix costs a few dozen bytes and is exact, so downstream routers resolve
    it with an ordinary FIB lookup and never load an encoder.
    """

    resolved_prefix: str
    score: float
    resolver: str

    @property
    def wire_bytes(self) -> int:
        return len(self.resolved_prefix.encode()) + 4


@dataclass
class Interest:
    """A request for a named piece of data."""

    name: str
    origin: str                       # consumer node that emitted it
    created_at: float
    tag: Optional[SemanticTag] = None
    hop_count: int = 0
    id: int = field(default_factory=lambda: next(_interest_ids))
    path: List[str] = field(default_factory=list)

    #: Ground truth, carried alongside the packet purely for scoring. A router
    #: never reads this -- it exists so the harness can tell a correct
    #: resolution from a confident wrong one.
    expected: Optional[str] = None
    kind: str = "exact"

    @property
    def wire_bytes(self) -> int:
        size = len(self.name.encode()) + INTEREST_OVERHEAD_BYTES
        return size + (self.tag.wire_bytes if self.tag else 0)

    def with_tag(self, tag: SemanticTag) -> "Interest":
        self.tag = tag
        return self

    @property
    def lookup_name(self) -> str:
        """The name a router should match against its FIB.

        Once an edge router has resolved the Interest, downstream routers match
        the resolved prefix; before that, the name as the client wrote it.
        """
        return self.tag.resolved_prefix if self.tag else self.name


@dataclass
class Data:
    """A response travelling back down the chain of PIT entries."""

    name: str                         # the canonical name the producer serves
    interest_id: int
    produced_at: float
    producer: str
    payload_bytes: int = DATA_PAYLOAD_BYTES

    @property
    def wire_bytes(self) -> int:
        return len(self.name.encode()) + DATA_OVERHEAD_BYTES + self.payload_bytes


@dataclass
class Nack:
    """No route, or no producer willing to serve the name."""

    name: str
    interest_id: int
    reason: str
    origin: str

    @property
    def wire_bytes(self) -> int:
        return len(self.name.encode()) + INTEREST_OVERHEAD_BYTES


#: Reasons a request ends, recorded per Interest for the outcome breakdown.
OUTCOME_EXACT = "exact"                # matched the FIB verbatim, fast path
OUTCOME_ES_HIT = "es_hit"              # served from a learned mapping
OUTCOME_GOSSIP_HIT = "gossip_hit"      # served from a mapping learned elsewhere
OUTCOME_SEMANTIC = "semantic"          # resolved by running the encoder
OUTCOME_TAGGED = "tagged"              # forwarded on an upstream router's tag
OUTCOME_NACK = "nack"                  # rejected, no sufficiently similar route
OUTCOME_MISDELIVERED = "misdelivered"  # forwarded confidently to the wrong producer
OUTCOME_TIMEOUT = "timeout"            # PIT entry expired

RESOLVED_OUTCOMES = frozenset(
    {OUTCOME_EXACT, OUTCOME_ES_HIT, OUTCOME_GOSSIP_HIT, OUTCOME_SEMANTIC, OUTCOME_TAGGED}
)
