"""Where every latency number in the simulation comes from.

The Phase 2 prototype hard-coded an 8 ms embedding cost and a 0.5 ms search
cost.  Both are plausible and neither was measured, which means the comparison
between strategies was decided in the configuration file rather than by the
experiment.  This module instead holds costs that
``experiments/bench_micro.py`` measures on the machine running the study, and
records whether each one was measured or assumed so that a reader can tell the
difference.

The one cost that must be a function rather than a constant is the similarity
search: it is a matrix-vector product over the FIB, so it grows with FIB size,
and a strategy that searches a 1000-entry FIB should not be charged the same as
one searching 50 entries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class CostModel:
    """Per-operation processing costs at a router, in milliseconds."""

    #: One encoder inference on a single name. The dominant cost of the slow path.
    encode_ms: float = 6.7

    #: Cosine search: a fixed overhead plus a per-FIB-entry term.
    search_base_ms: float = 0.01
    search_per_entry_ms: float = 0.0002

    #: Hash-table probe of the Embedding Store.
    es_lookup_ms: float = 0.001

    #: Projecting one embedding onto the signature basis.
    signature_ms: float = 0.02

    #: Reading a semantic tag off an Interest and matching it in the FIB.
    tag_match_ms: float = 0.005

    #: One Q-table update, for the reinforcement-learning baseline.
    qtable_update_ms: float = 0.002

    #: Applying a batch of gossiped mappings.
    gossip_apply_ms_per_entry: float = 0.002

    #: Which of the above were measured rather than assumed.
    measured: Dict[str, bool] = field(default_factory=dict)
    machine: str = ""

    def search_ms(self, fib_size: int) -> float:
        return self.search_base_ms + self.search_per_entry_ms * max(0, fib_size)

    def semantic_resolve_ms(self, fib_size: int) -> float:
        """Full slow path: encode the name, then compare it to the whole FIB."""
        return self.encode_ms + self.search_ms(fib_size)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "CostModel":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def load_or_default(cls, path: Optional[str | Path]) -> "CostModel":
        """Load measured costs, falling back to documented defaults.

        The fallback is loud on purpose: an unmeasured cost model is a caveat
        that belongs in the results, not a silent default.
        """
        if path is not None and Path(path).exists():
            return cls.load(path)
        model = cls()
        model.measured = {k: False for k in model.to_dict() if k not in ("measured", "machine")}
        model.machine = "defaults (not measured on this machine)"
        return model

    @property
    def fully_measured(self) -> bool:
        return bool(self.measured) and all(self.measured.values())


#: Link and producer delays for the topology.  SAF's edge-network measurements
#: put the client-to-router hop at 10 ms and the router-to-provider hop at
#: 20 ms; the same figures are used here so latency results are comparable.
@dataclass
class LinkProfile:
    consumer_to_edge_ms: float = 10.0
    router_to_router_ms: float = 5.0
    router_to_producer_ms: float = 20.0
    bandwidth_mbps: float = 100.0
    producer_service_ms: float = 1.0

    #: How long a consumer waits before declaring an Interest lost.
    interest_lifetime_ms: float = 4000.0
