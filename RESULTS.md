# Results

Twenty-seed means with 95% confidence intervals, from
`experiments/run_experiments.py --all --seeds 20`. Processing costs come from
`data/costs.json`, measured on the machine that ran the campaign by
`experiments/bench_micro.py`; no latency in the simulation is assumed.

Two independent domains throughout — a smart hospital and a smart city, 50
services each, 350 resolvable Interest names and 350 unsatisfiable ones. Where
they disagree, that is said.

---

## 1. What the forwarding path costs

| Operation | Cost | Relative |
|---|---:|---:|
| MiniLM-L6 inference, one name | 7.05 ms | 1× |
| Character n-gram encoder, one name | 0.096 ms | 73× cheaper |
| Cosine search over a 50-entry FIB | 0.0019 ms | 3,700× cheaper |
| Cosine search over a 1000-entry FIB | 0.027 ms | 260× cheaper |
| Embedding Store probe | 0.0001 ms | 70,000× cheaper |

**This decides the design.** The similarity search is not the bottleneck and
cannot become one at any FIB size a router would hold. Approximate
nearest-neighbour indexes and hash-based lookup optimise the 0.0019 ms. The only
quantity worth attacking is how often the 7.05 ms is paid.

## 2. Signatures cannot select routes

Nearest FIB entry by Hamming distance over random projections, against the true
cosine argmax on 300 reworded hospital names:

| Width | Wire | Ranked 1st | in top 3 | in top 8 |
|---:|---:|---:|---:|---:|
| 64 bits | 8 B | 0.327 | 0.563 | 0.793 |
| 256 bits | 32 B | 0.597 | 0.823 | 0.953 |
| 1024 bits | 128 B | 0.840 | 0.983 | 1.000 |

A plane routing on the nearest signature misroutes two Interests in three at 64
bits, and one in six at 128 bytes per entry. Service names within one domain
differ in a single component out of five; locality-sensitive hashing does not
have the resolution to separate them. This is why the Interest tag carries the
resolved prefix as a string. Signatures are kept only for anti-entropy digests,
where 32 bytes against a 1.5 KB embedding is the point.

## 3. The threshold problem

Precision, recall and F1 against the cosine threshold:

| Threshold | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Recall (city) | 0.985 | 0.984 | 0.976 | 0.957 | 0.877 | **0.809** | 0.725 | 0.681 |
| Recall (hospital) | — | — | — | 0.939 | 0.915 | **0.866** | 0.790 | 0.718 |
| Precision (city) | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | **0.999** | 1.000 | 1.000 |

SAF selects 0.7 by tuning on its own catalog. Here it costs 18 points of recall
against the best operating point while buying essentially no precision — the
curve is flat above 0.99 throughout, because the distractors sit far enough away
that even a permissive cutoff rejects them. Best F1 is near 0.45–0.50 on city
and 0.55–0.60 on hospital: **different from SAF's, and different from each
other.**

## 4. An error budget instead of a threshold

The operator sets ε, the share of semantic forwarding decisions that may turn
out wrong. Each route learns its own boundary from observed outcomes,
hierarchically — shrinking from a pooled boundary toward its own as evidence
accumulates. Realised error is measured **out of sample**, over exactly the
decisions the budget governs: semantic resolutions, fresh or cached or learned
from a neighbour. Exact FIB matches are excluded; they were never judgement
calls and including them would dilute any rate towards zero.

**Hospital, 8 edge routers:**

| ε | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| realised error | 0.006 | 0.006 | 0.010 | 0.008 | 0.015 | 0.020 | 0.029 |
| within budget | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| satisfaction | 0.827 | 0.827 | 0.853 | 0.903 | 0.926 | 0.960 | 0.968 |
| coverage | 0.208 | 0.208 | 0.219 | 0.236 | 0.246 | 0.263 | 0.269 |

The budget holds at every setting, and the coverage/risk curve is monotone: a
tighter budget forwards less and errs less. At ε = 0.2, risk-controlled
forwarding beats a tuned-threshold GS-NDN on both axes simultaneously —
satisfaction 0.946 against 0.933, realised error 0.0074 against 0.0189 — with no
threshold to tune.

Learned boundaries spread as evidence accumulates (standard deviation 0 → 0.073
as ε loosens from 0.02 to 0.35). That spread is the direct evidence that a
single global number could not have served every route.

### Why a flat per-route estimator does not work

The first version estimated each route's boundary from its own observations
alone and returned "refuse everything" for every calibrated route. A Wilson
bound at ε = 0.05 needs upwards of thirty clean observations before it concedes
anything, and no single route accumulates them. Pooling every route's evidence
gives one well-supported boundary that each route then shrinks away from at a
rate set by its own evidence. This is not a refinement; without it the mechanism
does not function.

## 5. Risk control deadlocks without exploration

A boundary set too high blocks exactly the decisions that would have produced
the evidence to lower it. The system stops forwarding, stops learning, and stays
there.

| | ε ≤ 0.10 | evidence gathered | satisfaction |
|---|---:|---:|---:|
| no exploration | boundary 1.00, refuse all | 104 | 0.745 |
| 5% exploration | boundary 0.87 | 476–740 | 0.853 |

This is a property of learning from your own choices, not an implementation
fault, and it is the sharpest argument for gossip in this design: **pooled
evidence means each router has to explore less.** Exploratory forwards are
decisions the budget explicitly did not cover, and are counted separately rather
than folded into the reported rate.

## 6. Producer churn

Producers depart, return and relocate. Relocation is the hard case: the name
stays available, nothing times out, and no similarity score changes — a router
with a cached mapping keeps forwarding to a face that no longer leads anywhere
useful. Only feedback detects it.

Interest satisfaction, hospital, 8 edge routers:

| mean seconds between events | static | 5 s | 2 s |
|---|---:|---:|---:|
| SAF+ES | 0.928 | 0.774 | 0.577 |
| GS-NDN | 0.933 | 0.778 | 0.579 |
| GS-NDN, no verification | 0.928 | 0.773 | 0.577 |
| **Risk-controlled** | **0.946** | **0.797** | **0.634** |

The advantage widens as the network becomes unstable — from 1.8 points over
SAF+ES when static to 5.7 points at two seconds between events. This is the
setting the static ablation could not show, where the earlier finding that
verification costs 6% more work for 0.8 points of satisfaction reverses.

## 7. Compromised routers

Two things travel over gossip and each is poisoned differently. A **false
mapping** sends Interests to the wrong producer and is exposed by that
producer's refusal. **False calibration evidence** is quieter: fabricated "this
score worked" observations pull a boundary down so the victim accepts
resolutions it would have refused, and no individual decision looks wrong.

Hospital, share of routers compromised, attacker re-injecting every gossip round
against names clients actually request:

| compromised | 0% | 12.5% | 25% | 50% |
|---|---:|---:|---:|---:|
| GS-NDN satisfaction | 0.933 | 0.855 | 0.815 | 0.750 |
| Risk-controlled satisfaction | 0.946 | 0.855 | 0.821 | 0.767 |
| Risk-controlled realised error | 0.007 | 0.033 | 0.060 | 0.080 |

**Degradation is graceful, not prevented.** Against a persistent attacker,
retraction does not contain the attack — poison retracted after one round trip
is re-injected on the next. What limits the damage is that a router's own
confirmed mappings outrank anything it is told, so a router that has learned the
truth first-hand resists. The realised error rises above the 0.2 budget only
past 50% compromise, but the budget is no longer a guarantee once any evidence
is adversarial: it is a guarantee conditional on honest reporting, and this
table is the measurement of what that condition is worth. Provenance and
reputation are the obvious next step and are not implemented.

## 8. Encoder heterogeneity

Routers running different encoders in one network — MiniLM-L6 alongside the
character n-gram control. Mappings pool across the boundary because a route that
served a name serves it however anyone chose to ask. **Scores do not**: 0.62
from MiniLM and 0.62 from an n-gram model are not the same measurement, and the
gossip layer refuses to send evidence across an encoder boundary rather than
pooling incomparable numbers.

This is the structural argument for gossiping names and verdicts rather than
vectors, and it is the second reason signatures were the wrong currency —
they are not portable either.

## 9. Where the cost goes as the network grows

Encoder inferences per run as edge routers increase:

| Edge routers | 1 | 2 | 4 | 8 | 16 | Growth |
|---|---:|---:|---:|---:|---:|---:|
| SAF | 2,022 | 2,045 | 2,062 | 2,073 | 2,079 | +3% |
| SAF+ES | 833 | 901 | 1,006 | 1,155 | 1,330 | **+60%** |
| GS-NDN | 868 | 886 | 868 | 905 | 952 | **+10%** |

SAF caches nothing, so it pays one inference per FIB miss wherever the miss
happens. SAF+ES caches per router: as traffic splits across more routers each
cache sees a thinner slice, and the same total traffic produces 60% more
inferences — **the Embedding Store's benefit erodes as the network grows**,
which is the setting SAF's conclusion names as future work. Sharing restores it.

## 10. Latency under load

95th-percentile resolution time on SAF's single-router topology:

| Interests/s | 30 | 100 | 200 | 300 |
|---|---:|---:|---:|---:|
| SAF | 79.1 | 84.6 | 98.3 | **172.2** |
| SAF+ES | 77.9 | 77.9 | 79.7 | **82.2** |
| GS-NDN | 77.9 | 77.9 | 80.1 | **82.8** |

SAF reaches 0.90 processor utilisation at 300 Interests/s and its tail more than
doubles — the bottleneck the paper reports past roughly 210 Interests/s,
reproduced. **This result belongs to the Embedding Store, not to us**: on one
router there is nobody to gossip with, and the two cached strategies are
indistinguishable. Caching fixes latency at one router; sharing fixes cost
across many.

## 11. Does a transformer earn its cost?

| Encoder | ε / Th | ISR (hospital) | ISR (city) |
|---|---:|---:|---:|
| MiniLM-L6 | 0.6 | 0.939 | 0.957 |
| MiniLM-L6 | 0.7 | 0.866 | 0.803 |
| Character n-gram | 0.6 | 0.695 | 0.730 |
| Character n-gram | 0.7 | 0.652 | 0.643 |

Decisively yes. The lexical encoder handles abbreviation and morphology and
fails on genuine synonymy — `cardiac-monitor` against `heart-rate-monitor`
shares almost no character n-grams. It recovers about a third of the gap at 1/73
of the cost, which makes it a reasonable fallback for a router that cannot host
a transformer and not a replacement for one.

## 12. Energy

Eight edge routers, 30 s, idle charged over the same window for every strategy:

| Strategy | Total | Radio | Compute | Compute share | ISR |
|---|---:|---:|---:|---:|---:|
| Vanilla NDN | 549.1 J | 6.27 J | 3.3 J | 0.6% | 0.598 |
| SAF | 734.8 J | 9.57 J | 213.7 J | 29.1% | 0.933 |
| SAF+ES | 654.4 J | 9.57 J | 120.9 J | 18.5% | 0.933 |
| GS-NDN | 632.7 J | 9.60 J | 95.8 J | 15.1% | 0.938 |

Radio energy is identical across the semantic strategies — the same Interests
cross the same links — so compute is the entire difference. Semantic forwarding
is not cheap: it costs 15–29% of the whole budget on a task where exact matching
spends 0.6%. The question is what that buys (satisfaction 0.60 → 0.94) and
whether the bill can be reduced without giving it up.

SEF is compared on the energy axis only. It has no semantic layer, resolves
exactly what Vanilla NDN does, and spends almost nothing on compute; its
contribution is energy-aware next-hop choice, which is orthogonal.

## 13. Is the simulator right?

The transport layer is cross-checked against ndnSIM on the same topology, links
and load, exact-match forwarding only:

| | n | mean | p50 | p95 |
|---|---:|---:|---:|---:|
| ndnSIM | 4,351 | 53.72 ms | 70.08 ms | 70.18 ms |
| This model | 4,545 | 63.49 ms | **71.19 ms** | 71.20 ms |

Medians differ by 1.11 ms, of which 1.00 ms is the producer service time this
model charges and ndnSIM does not. **Unexplained residual: 0.11 ms.** Means
differ more because ndnSIM's Zipf consumer concentrates all consumers on one
popularity distribution and PIT-aggregates far more; the median is the quantity
both setups define identically. See [`ndnsim/`](ndnsim/).

## 14. What this does not show

- **Feedback is modelled, not assumed — but it is still a model.** Producers
  decide from schemas they declare, never from the catalog oracle, and those
  declarations are deliberately incomplete: at `alias_coverage=0.7` a producer
  refuses 29% of requests meant for it, and satisfaction falls 0.936 → 0.879 as
  coverage drops to 0.5. What is not modelled is a producer whose schema is
  adversarially wrong.
- **The budget is conditional on honest reporting.** Section 7 measures what
  that condition is worth; it does not remove it.
- **Exchangeability.** Conformal-style bounds assume the calibration and test
  distributions match. Zipf popularity and producer churn both violate this. The
  hierarchical estimator and the bounded observation window mitigate it; a
  weighted or adaptive variant would be the principled fix and is not
  implemented.
- **The catalogs are generated from a lexicon**, not collected from a
  deployment. The rewrite families mirror how IoT namespaces diverge, but they
  are still our idea of that.
- **One transformer.** SAF compares MiniLM-L6, MiniLM-L12 and MPNet and finds
  recognition quality nearly identical; only L6 is used here, alongside the
  lexical control.
