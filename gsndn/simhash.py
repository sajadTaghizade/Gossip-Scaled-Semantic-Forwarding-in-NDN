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
continuous gossip affordable.  (:mod:`gsndn.gossip` currently summarises its own
state directly; signatures are what a cross-vendor deployment would exchange,
and the width/accuracy trade is measured here.)

Candidate prefiltering is the other role signatures could fill, where a FIB is
large enough for the cosine to matter -- at 256 bits the true nearest entry is
in the top 8 candidates 95% of the time.  At the FIB sizes studied here that
buys nothing, so no prefilter is implemented; ``experiments/bench_micro.py``
measures the recall curve that would justify one.
"""

from __future__ import annotations

from typing import Sequence

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


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


def hamming_array(query: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Hamming distance from one signature to every row of a table."""
    if table.size == 0:
        return np.zeros(0, dtype=np.int32)
    return _POPCOUNT[np.bitwise_xor(table, query[None, :])].sum(axis=1, dtype=np.int32)
