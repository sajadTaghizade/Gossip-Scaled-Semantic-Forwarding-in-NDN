# Design notes — superseded

An earlier version of this file proposed three architectures (SHEF, NERVE, CAFÉ)
and recommended SHEF, which would have replaced the runtime
embedding-then-search pipeline with locality-sensitive hash lookup in the
forwarding plane. Building the simulator and measuring the pieces refuted the
premise of all three. The record is kept here because the measurements are
useful and the reasoning is worth not repeating.

## What was proposed, and what measurement showed

**SHEF — SimHash signatures as a forwarding-plane primitive.** The claim was
that bitwise matching would cut semantic lookup from 5–15 ms to 10–50 µs.

Two problems, both fatal.

A signature is derived *from* an embedding, so producing one still requires the
transformer inference the proposal was trying to avoid. The stated speedup could
never have been realised, because the dominant cost was never removed.

More seriously, signatures cannot pick the right route. Measured on the 300
reworded hospital names (`experiments/bench_micro.py`), nearest-by-Hamming
agrees with the true cosine argmax **32.7% of the time at 64 bits** and 84.0%
even at 1024 bits — 128 bytes per FIB entry. Service names within one domain
differ in a single component out of five, and locality-sensitive hashing does
not separate them. A router built this way would misroute most reworded
Interests.

**NERVE — HNSW approximate nearest-neighbour search at each router.** Aimed at
replacing an O(n) cosine scan with O(log n) graph traversal.

This optimises something that is not a cost. A cosine search over a 50-entry FIB
takes **0.0019 ms**; one MiniLM-L6 inference takes **7.05 ms** — a factor of
3,700. Extrapolating the measured fit, even a 10,000-entry FIB stays under
0.1 ms. Approximate search would trade accuracy for a saving that does not
appear in any total.

**CAFÉ — a cascade of progressively more expensive filters.** Sound in shape,
but its stages inherit the problems above: the cheap stages cannot decide, and
the expensive stage is the only one that costs anything.

## What the measurements pointed to instead

If the search is free and the encoder is everything, the only quantity worth
reducing is **how often the encoder runs**. That reframing is what the
implemented design follows, and it is why the contribution ended up being about
distribution rather than about data structures:

- SAF already avoids re-encoding at every hop, by attaching the resolved prefix
  to the Interest. That is theirs, and it is inherited rather than claimed.
- SAF's Embedding Store already avoids re-encoding a repeated wording at one
  router — and the scaling experiment shows its benefit eroding by 60% as the
  same traffic splits across 16 edge routers.
- What is left, and what this project contributes, is making one router's
  resolution usable by the others: verification via producer feedback, so a
  guess becomes a fact, and anti-entropy gossip of the facts.

Signatures did survive, in a smaller role: as anti-entropy digests, where 32
bytes against a 1.5 KB float32 embedding is exactly the property needed.

See [`README.md`](README.md) for the design as built and
[`RESULTS.md`](RESULTS.md) for the full measurements.
