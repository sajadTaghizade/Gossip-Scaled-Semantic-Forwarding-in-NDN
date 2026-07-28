"""Random-projection signatures over name embeddings.

A signature is one bit per random hyperplane: the sign of the embedding's
projection onto it.  For unit vectors, the probability that two names disagree
on a bit is their angle divided by pi, so Hamming distance estimates cosine
similarity at the cost of an XOR and a popcount.

What signatures are good for here is narrower than it first appears, and the
measurement is in ``experiments/bench_simhash.py``.  On the hospital catalog,
picking the nearest FIB entry by Hamming distance alone agrees with the true
cosine argmax only 33% of the time at 64 bits and 84% at 1024 bits.  Hospital
service names are mutually close -- they differ in one metric word out of five
-- and locality-sensitive hashing does not have the resolution to separate
them.  A design that replaces the cosine comparison in the forwarding plane
with a signature lookup would therefore misroute constantly, and it could not
avoid the dominant cost regardless, since producing a signature requires the
embedding the encoder spent milliseconds computing.

Two roles survive that measurement:

*Anti-entropy digests.*  Comparing what two routers know without shipping
embeddings.  A learned mapping travels as two names plus 32 bytes instead of a
1.5 KB float32 vector -- a factor of about 48 on the wire, which is what makes
continuous gossip affordable.

*Candidate prefiltering*, where a FIB is large enough for the cosine to matter.
At 256 bits the true nearest entry is in the top 8 candidates 95% of the time.
At the FIB sizes studied here that buys nothing, and the code says so rather
than claiming a speedup it cannot demonstrate.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BITS = 256

#: popcount for every byte value, so Hamming distance is a table lookup and a
#: sum. numpy only grew ``bitwise_count`` in 2.0 and this works everywhere.
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


class SimHasher:
    """Projects embeddings onto a fixed random basis to produce bit signatures.

    The basis is derived from a seed, so every router in a run generates the
    same one.  In a deployment it would be a published constant: two routers
    hashing against different bases produce incomparable signatures.

    Signatures are ``uint8`` arrays rather than Python ints so that widths above
    64 bits work and so that a whole table can be compared at once.
    """

    def __init__(self, dim: int, bits: int = DEFAULT_BITS, seed: int = 0x5EED) -> None:
        if bits % 8:
            raise ValueError(f"signature width must be a whole number of bytes, got {bits}")
        self.dim = dim
        self.bits = bits
        self.nbytes = bits // 8
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.planes = rng.standard_normal((bits, dim)).astype(np.float32)

    def signature(self, vector: np.ndarray) -> np.ndarray:
        return self.signatures(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]

    def signatures(self, matrix: np.ndarray) -> np.ndarray:
        """Signatures for a stack of row vectors, as an ``(n, nbytes)`` uint8 array."""
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.size == 0:
            return np.zeros((0, self.nbytes), dtype=np.uint8)
        bits = (matrix @ self.planes.T) > 0.0
        return np.packbits(bits, axis=1)

    def estimate_cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Recover an approximate cosine from a Hamming distance."""
        return float(np.cos(np.pi * hamming(a, b) / self.bits))

    def distance_for_cosine(self, cosine: float) -> float:
        """The Hamming distance a given cosine corresponds to, in expectation."""
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return float(np.arccos(cosine) / np.pi * self.bits)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


def hamming_array(query: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Hamming distance from one signature to every row of a table."""
    if table.size == 0:
        return np.zeros(0, dtype=np.int32)
    return _POPCOUNT[np.bitwise_xor(table, query[None, :])].sum(axis=1, dtype=np.int32)


def digest(signatures: np.ndarray) -> int:
    """A cheap order-independent summary of a set of signatures.

    Used by anti-entropy to decide in one comparison whether two routers hold
    the same knowledge, before either of them offers to send anything.
    """
    if signatures.size == 0:
        return 0
    folded = np.bitwise_xor.reduce(signatures, axis=0)
    return int.from_bytes(bytes(folded.tolist()), "little")


class SignatureTable:
    """Signature-to-route map with nearest-neighbour lookup by Hamming distance.

    Provided so the prefilter claim can be measured rather than asserted;
    ``candidates`` returns the top-k for an exact cosine pass, which is the only
    form in which signature matching is accurate enough to route on.
    """

    def __init__(
        self,
        hasher: SimHasher,
        max_distance: Optional[int] = None,
        cosine_threshold: float = 0.7,
    ) -> None:
        self.hasher = hasher
        self.max_distance = (
            max_distance
            if max_distance is not None
            else int(round(hasher.distance_for_cosine(cosine_threshold)))
        )
        self._table = np.zeros((0, hasher.nbytes), dtype=np.uint8)
        self._routes: List[Tuple[str, str]] = []   # (canonical prefix, face)
        self.lookups = 0
        self.hits = 0

    def add(self, signature: np.ndarray, prefix: str, face: str) -> None:
        self._table = np.vstack([self._table, signature[None, :]])
        self._routes.append((prefix, face))

    def bulk_add(self, signatures: np.ndarray, routes: Sequence[Tuple[str, str]]) -> None:
        if len(signatures) != len(routes):
            raise ValueError("signatures and routes must be the same length")
        self._table = np.vstack([self._table, signatures]) if len(signatures) else self._table
        self._routes.extend(routes)

    def candidates(self, signature: np.ndarray, k: int = 8) -> List[Tuple[str, str, int]]:
        """The k nearest routes by Hamming distance, closest first."""
        if not self._routes:
            return []
        distances = hamming_array(signature, self._table)
        order = np.argsort(distances)[:k]
        return [(self._routes[i][0], self._routes[i][1], int(distances[i])) for i in order]

    def lookup(self, signature: np.ndarray) -> Optional[Tuple[str, str, int]]:
        """Nearest route within ``max_distance``, as (prefix, face, distance)."""
        self.lookups += 1
        best = self.candidates(signature, k=1)
        if not best or best[0][2] > self.max_distance:
            return None
        self.hits += 1
        return best[0]

    def __len__(self) -> int:
        return len(self._routes)

    @property
    def memory_bytes(self) -> int:
        return len(self._routes) * (self.hasher.nbytes + 8)
