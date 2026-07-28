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
| realised error | 0.009 | 0.009 | 0.009 | 0.010 | 0.014 | 0.025 | 0.035 |
| within budget | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| satisfaction | 0.818 | 0.818 | 0.841 | 0.885 | 0.919 | 0.959 | 0.967 |
| coverage | 0.194 | 0.194 | 0.205 | 0.222 | 0.238 | 0.258 | 0.265 |
| calibrated routes | 235 | 235 | 264 | — | 348 | — | 402 |
| boundary spread (σ) | 0.000 | 0.000 | 0.012 | — | 0.042 | — | 0.096 |

City behaves the same way: 0.010 realised at ε = 0.02 rising to 0.020 at 0.40,
within budget throughout.

**It is a trade, not a free win.** At ε = 0.2 the realised error is half
GS-NDN's — 0.014 against 0.027 — for two points less satisfaction, 0.919 against
0.940. What ε buys is movement along that curve without retuning, and knowledge
of where on it you are. An early single-seed run showed a win on both axes
simultaneously; twenty seeds did not reproduce it and the claim was withdrawn
rather than kept at a favourable seed.

Learned boundaries spread as evidence accumulates — standard deviation 0.000 at
ε = 0.02, where almost every route still sits on the pooled boundary, rising to
0.096 at ε = 0.40. That spread is the direct evidence that a single global
number could not have served every route.

### Against a threshold tuned on the other domain

The comparison above is not the one a sceptic would ask for. A fixed threshold
traces its own frontier: sweep it and every point is a (realised error,
satisfaction) pair, so an operator who can measure realised error on their own
catalog can simply pick the point they want. Section 3's flat precision curve
does not answer that; it only says where the best F1 sits.

What the budget can claim over such an operator is narrower, and it is about
*transfer*. A threshold is placed using the catalog available at tuning time,
and the network it then runs on is not that catalog. So: build the frontier on
one domain, take for each budget the most permissive threshold whose realised
error stays inside it, carry that threshold to the other domain, and stand it
against rc-ndn given the same budget and calibrating live on the domain it was
moved to. Same configuration as the sweep above, twenty seeds
(`--experiment threshold_transfer`).

**Tuned on city, evaluated on hospital:**

| ε | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|
| threshold chosen on city | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |
| its realised error on hospital | **0.032** | 0.032 | 0.032 | 0.032 | 0.032 | 0.032 |
| within budget | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ |
| its satisfaction | 0.971 | 0.971 | 0.971 | 0.971 | 0.971 | 0.971 |
| rc-ndn realised error | 0.009 | 0.009 | 0.009 | 0.014 | 0.025 | 0.035 |
| rc-ndn satisfaction | 0.818 | 0.818 | 0.841 | 0.919 | 0.959 | 0.967 |

**Tuned on hospital, evaluated on city:**

| ε | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|
| threshold chosen on hospital | 0.70 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |
| its realised error on city | 0.003 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 |
| its satisfaction | 0.804 | 0.983 | 0.983 | 0.983 | 0.983 | 0.983 |
| rc-ndn realised error | 0.010 | 0.010 | 0.011 | 0.012 | 0.016 | 0.020 |
| rc-ndn satisfaction | 0.813 | 0.813 | 0.845 | 0.956 | 0.977 | 0.979 |

**rc-ndn does not dominate the transferred frontier, and the earlier claim of
better efficiency is withdrawn.** Across the fourteen comparisons -- seven
budgets in each direction -- rc-ndn Pareto-dominates in **none**. The
transferred threshold dominates in four, all at ε ≥ 0.2 on the hospital-tuned
side where 0.45 reaches 0.983 satisfaction inside budget and rc-ndn's best is
0.979. The other ten are trades: less error for less satisfaction, which is the
same trade section 4 already reports.

Two things survive, and they are worth less than the withdrawn claim.

*The transferred threshold missed the budget once, and it was the tightest one.*
Tuned on city at ε = 0.02, 0.45 realises 0.011 there and 0.032 on hospital --
over budget by 61%, with the confidence interval (±0.009) nowhere near covering
the gap. rc-ndn held its budget in all fourteen. One violation in fourteen is a
weak result and is reported as one; what makes it worth reporting at all is
where it landed. Every budget from 0.05 up is slack -- no threshold in the sweep
realises more than 0.032 error on either domain -- so "tuning" degenerates to
"take the most permissive threshold" and transfer cannot fail. The one budget
that actually binds is the one transfer broke.

*The two knobs are not equally available.* Placing a threshold on the frontier
requires having measured realised error, which requires a labelled catalog for
the domain you are about to run on. That is the assumption the transfer
experiment removes, and removing it is the whole point: rc-ndn gets its labels
from producer feedback at run time, on the domain it is actually deployed in.
The defensible statement is therefore **not** that the budget forwards more
efficiently, but that it is *tunable without access to the test domain* and
reports where on the curve it ended up. On these two catalogs an operator who
does have a labelled sample of the target domain should tune a threshold on it
and will do slightly better.

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

At ε ≤ 0.10, hospital:

| | observations | calibrated routes | satisfaction | realised error |
|---|---:|---:|---:|---:|
| no exploration | 104 | 68 | 0.738 | 0.002 |
| 5% exploration | 598 | 235 | 0.818 | 0.009 |

Tightening ε below 0.10 changes nothing for the unexploring controller: it is
already refusing everything it can, so there is no behaviour left to tighten.
Exploratory forwards are decisions the budget explicitly did not cover and are
counted separately rather than folded into the reported rate.

**Sharing evidence is the precondition for calibration, not an optimisation.**

| at ε = 0.2 | observations | of which remote | calibrated routes |
|---|---:|---:|---:|
| with evidence gossip | 1,261 | 945 (75%) | 348 |
| without | 245 | 0 | 147 |

Three quarters of what a router calibrates on was gathered by somebody else. No
router sees enough of its own traffic to fit per-route boundaries; this is what
gossip is for in this design, and it is a stronger reason than the encoder
savings that motivated the earlier version.

## 6. Producer churn

Producers depart, return and relocate. Relocation is the hard case: the name
stays available, nothing times out, and no similarity score changes — a router
with a cached mapping keeps forwarding to a face that no longer leads anywhere
useful. Only feedback detects it.

Interest satisfaction, hospital, 8 edge routers:

| mean seconds between events | static | 10 s | 5 s | 2 s | 1 s |
|---|---:|---:|---:|---:|---:|
| SAF+ES | 0.932 | 0.851 | 0.780 | 0.582 | 0.385 |
| GS-NDN | 0.940 | 0.851 | 0.787 | 0.587 | 0.380 |
| GS-NDN, no verification | 0.932 | 0.844 | 0.780 | 0.584 | 0.380 |
| Risk-controlled | 0.919 | 0.828 | 0.764 | 0.579 | 0.368 |

**This is a negative result and is reported as one.** Churn was the setting
where verification was expected to finally earn its cost: a relocation changes
the right answer without changing any similarity score, so re-encoding cannot
detect it and only feedback can. Over twenty seeds no such advantage appears.
Every strategy degrades together, and at one second between events all four land
within 0.02 of each other. A single-seed run during development suggested
risk-controlled forwarding pulled ahead under churn; it does not.

What churn does establish is that the Embedding Store's invalidation path --
specified by SAF, implemented here, and never exercised by any static experiment
-- is now under load, and that no strategy in this family tolerates a network
reorganising itself every second.

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
| GS-NDN satisfaction | 0.940 | 0.832 | 0.783 | 0.747 |
| Risk-controlled satisfaction | 0.919 | 0.822 | 0.780 | 0.754 |
| GS-NDN realised error | 0.027 | 0.103 | 0.142 | 0.165 |
| Risk-controlled realised error | 0.014 | 0.105 | 0.151 | 0.170 |

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

Satisfaction degrades gracefully as more routers run the weaker encoder --
hospital 0.940 → 0.931 → 0.914 as the lexical share goes 0 → 25% → 50%, city
0.956 → 0.949 → 0.930 -- which is what should happen when a mixed network keeps
sharing what it can and stops sharing what it cannot.

This is the structural argument for gossiping names and verdicts rather than
vectors, and it is the second reason signatures were the wrong currency: they
are not portable across encoders either.

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
