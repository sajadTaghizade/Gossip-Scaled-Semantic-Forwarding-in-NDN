# Results

Twenty-seed means with 95% confidence intervals, from
`experiments/run_experiments.py --all --seeds 20`. Processing costs come from
`data/costs.json`, measured on the machine that ran the campaign by
`experiments/bench_micro.py`; no latency in the simulation is assumed.

Two independent domains throughout — a smart hospital and a smart city, 50
services each, 350 resolvable Interest names and 350 unsatisfiable ones. Where
they disagree, that is said.

Every number below is twenty seeds. Where an effect is small enough that the
per-arm intervals overlap, the comparison is repeated **paired by seed** — all
arms see the same workload, the same events and the same producers, so the
seed-to-seed variation those intervals are made of is shared and cancels. That
is said explicitly wherever it is done, and the unpaired means are printed
alongside so nothing rests on the paired form alone.

Two results carry this document, and they are not equally strong. §2–3 is
the stronger one and holds regardless of any threshold or budget: sharing a
verified resolution by gossip, instead of caching it per router, slows how
fast recognition cost grows with the network. It carries two scope conditions
that §2 measures rather than assumes — the effect shrinks by about two thirds
as the run lengthens from 60 to 600 seconds, and it reverses below roughly four
edge routers — so what it supports is a claim about large networks still
learning their catalogs, not a standing property of the two designs.

§5–10 is narrower and reports a claim that was tested and withdrawn: replacing
the similarity threshold with an online error budget does not forward more
efficiently than a threshold tuned on a labelled target domain (§6), and §16
shows that correcting a real bias in its evidence does not rescue that
comparison either. What survives is that it needs no labelled catalog to tune,
that it keeps most of verification's advantage under a churn event a
threshold-only design cannot even see (§8), and that where evidence is scarce
or biased the correction in §16 is worth up to 0.08 satisfaction.

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

## 2. Where the cost goes as the network grows

Encoder inferences per run as edge routers increase:

| Edge routers | 1 | 2 | 4 | 8 | 16 | Growth |
|---|---:|---:|---:|---:|---:|---:|
| SAF | 2,022 | 2,045 | 2,062 | 2,073 | 2,079 | +3% |
| SAF+ES | 833 | 901 | 1,006 | 1,155 | 1,330 | **+60%** |
| GS-NDN | 875 | 893 | 877 | 914 | 959 | **+10%** |

SAF caches nothing, so it pays one inference per FIB miss wherever the miss
happens. SAF+ES caches per router: as traffic splits across more routers each
cache sees a thinner slice, and the same total traffic produces 60% more
inferences — **the Embedding Store's benefit erodes as the network grows**,
which is the setting SAF's conclusion names as future work. Sharing restores it.

### The horizon this is measured over, and why it matters

That table is one 60-second run, and the figure moves with the run length. It
has to. A cold cache costs one inference per router per distinct wording: N
routers pay it N times, but they pay it **once**. Run for longer and that fixed
cost is spread over more traffic, the per-router caches warm up, and the gap
closes. Reporting a single growth figure without its horizon overstates a
transient as a steady state, so here is the sweep — 1 → 16 edge routers at three
horizons, twenty seeds, both domains:

| Horizon | SAF+ES growth | GS-NDN growth | Gossip saves at 16 edges |
|---|---:|---:|---:|
| 60 s | +55% / +61% | +9% / +15% | **26% / 27%** |
| 240 s | +31% / +37% | +6% / +12% | **16% / 17%** |
| 600 s | +14% / +21% | +1% / +9% | **7.5% / 9.5%** |

*hospital / city.*

**The ordering never reverses at scale, and the magnitude falls by roughly two
thirds.** GS-NDN grows less than SAF+ES at every horizon on both domains, so the
qualitative claim in the table above survives; what does not survive is reading
+60% against +10% as a standing property of the two designs. It is the 60-second
figure.

**And below about four edge routers, gossip loses outright.** At one edge router
there is nobody to share with, so anti-entropy is pure overhead: 4.4–5.5% *more*
inferences than a plain per-router cache on hospital, 0.9–2.3% more on city, at
every horizon. The benefit appears from roughly four edge routers upward. A
reader of the first table alone would not know that.

### Would an unbounded sync layer do better?

The obvious objection to all of this is that NDN already has anti-entropy
dataset synchronisation — ChronoSync, PSync, State Vector Sync — and NLSR
already floods name-prefix reachability over such a layer. If sharing is what
helps, why not simply turn that on and let it run without the fanout and delta
limits this gossip layer imposes?

`gs-ndn-full-sync` answers the narrow version of that question. It is this
protocol with the limits opened up: every router reconciles with every peer it
has each round instead of two, and no cap on how many entries one exchange may
carry. **It is a bound, not an implementation of NDN Sync** — those protocols
differ from each other and from this one in how state is digested and how much
of a dataset one exchange names, and NLSR carries prefix LSAs rather than
resolutions. What it can say is what an upper limit on sharing aggression buys
and costs, at 16 edge routers:

| Horizon | GS-NDN saves | full-sync saves | GS-NDN gossip | full-sync gossip |
|---|---:|---:|---:|---:|
| 60 s | **26% / 27%** | 21% / 23% | 743 / 766 kB | 871 / 897 kB (1.2×) |
| 240 s | **16% / 17%** | 13% / 15% | 1516 / 1541 kB | 2181 / 2183 kB (1.4×) |
| 600 s | **7.5% / 9.5%** | 6.2% / 8.3% | 2777 / 2768 kB | 4461 / 4456 kB (1.6×) |

*hospital / city, twenty seeds.*

**Syncing harder is worse on both axes.** It removes *fewer* encoder inferences
than the bounded protocol — 21% against 26% at 60 seconds, and growth 1 → 16
edge routers of +16% against +9% — while sending 1.2× to 1.6× the bytes. The
reason is that neither is limited by how fast a mapping can reach everyone.
Anti-entropy at fanout 2 already reaches N routers in O(log N) rounds, so
raising the fanout mostly buys redundant deliveries, and lifting the delta cap
lets a single exchange carry history the peer will discard. What limits the
saving is how many *distinct* wordings the network has yet to see, which no
amount of sync changes.

That is the honest form of the answer to "why not just announce the prefix":
not that sync would fail, but that the sharing benefit saturates early, so the
part of the design doing the work is *what* is shared and *when* — a mapping,
only once a Data packet has proven it — rather than how aggressively. It also
means an implementation over PSync or SVS is a reasonable engineering choice
this work does not argue against, and did not test.

So the scope of this result is: **sharing pays a network that is large and still
learning its catalog, and pays progressively less as it settles.** The regimes
where it earns its cost are the ones a static 60-second measurement flatters —
networks being brought up, catalogs still growing, producers still arriving —
and section 8's churn experiments are the closest thing here to measuring one of
them directly. A large, stable, long-running deployment with a fixed catalog is
the case where a per-router cache eventually catches up on its own, and this
work has no result showing otherwise.

## 3. Latency under load

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

## 4. Signatures cannot select routes

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

## 5. The threshold problem

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

## 6. An error budget instead of a threshold

This section and the next four ask a narrower question than §2–3: whether the
fixed similarity threshold — the second cost identified above — can be
replaced by something that does not need retuning per catalog. Nothing here
changes how the forwarding path scales; that was already settled by gossip.

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
catalog can simply pick the point they want. Section 5's flat precision curve
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

| ε | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| threshold chosen on city | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |
| its realised error on hospital | **0.032** | 0.032 | 0.032 | 0.032 | 0.032 | 0.032 | 0.032 |
| within budget | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| its satisfaction | 0.971 | 0.971 | 0.971 | 0.971 | 0.971 | 0.971 | 0.971 |
| rc-ndn realised error | 0.009 | 0.009 | 0.009 | 0.010 | 0.014 | 0.025 | 0.035 |
| rc-ndn satisfaction | 0.818 | 0.818 | 0.841 | 0.885 | 0.919 | 0.959 | 0.967 |

**Tuned on hospital, evaluated on city:**

| ε | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| threshold chosen on hospital | 0.70 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |
| its realised error on city | 0.003 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 |
| its satisfaction | 0.804 | 0.983 | 0.983 | 0.983 | 0.983 | 0.983 | 0.983 |
| rc-ndn realised error | 0.010 | 0.010 | 0.011 | 0.009 | 0.012 | 0.016 | 0.020 |
| rc-ndn satisfaction | 0.813 | 0.813 | 0.845 | 0.916 | 0.956 | 0.977 | 0.979 |

**rc-ndn does not dominate the transferred frontier, and the earlier claim of
better efficiency is withdrawn.** Across the fourteen comparisons -- seven
budgets in each direction -- rc-ndn Pareto-dominates in **none**. The
transferred threshold dominates in four, all at ε ≥ 0.2 on the hospital-tuned
side where 0.45 reaches 0.983 satisfaction inside budget and rc-ndn's best is
0.979. The other ten are trades: less error for less satisfaction, which is the
same trade section 6 already reports.

Two things survive, and they are worth less than the withdrawn claim.

*The transferred threshold missed the budget once, and it was the tightest one.*
Tuned on city at ε = 0.02, 0.45 realises 0.011 there and 0.032 on hospital --
over budget by 61%, and the miss is resolved rather than borderline: the
interval is ±0.009, so even its lower end, 0.023, sits above the 0.020 it was
asked for. rc-ndn held its budget in all fourteen. One violation in fourteen is a
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
efficiently, but that it is *tunable without access to the test domain* --
call this the zero-tuning property -- and reports where on the curve it ended
up. On these two catalogs an operator who does have a labelled sample of the
target domain should tune a threshold on it and will do slightly better.

### Why a flat per-route estimator does not work

The first version estimated each route's boundary from its own observations
alone and returned "refuse everything" for every calibrated route. A Wilson
bound at ε = 0.05 needs upwards of thirty clean observations before it concedes
anything, and no single route accumulates them. Pooling every route's evidence
gives one well-supported boundary that each route then shrinks away from at a
rate set by its own evidence. This is not a refinement; without it the mechanism
does not function.

## 7. Risk control deadlocks without exploration

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

## 8. Producer churn

Producers depart, return and relocate. Relocation looks like the hard case: the
name stays available, nothing times out, and no similarity score changes — a
router with a cached mapping keeps forwarding to a face that no longer leads
anywhere useful.

Interest satisfaction, hospital, 8 edge routers:

| mean seconds between events | static | 10 s | 5 s | 2 s | 1 s |
|---|---:|---:|---:|---:|---:|
| SAF+ES | 0.932 | 0.851 | 0.780 | 0.582 | 0.385 |
| GS-NDN | 0.940 | 0.851 | 0.787 | 0.587 | 0.380 |
| GS-NDN, no verification | 0.932 | 0.844 | 0.780 | 0.584 | 0.380 |
| Risk-controlled | 0.919 | 0.828 | 0.764 | 0.579 | 0.368 |
| Risk-controlled, adaptive ε | 0.931 | 0.838 | 0.771 | 0.586 | 0.375 |

**This is a negative result and is reported as one.** Churn was the setting
where verification was expected to finally earn its cost. Over twenty seeds no
such advantage appears. Every strategy degrades together, and at one second
between events all five land within 0.02 of each other. A single-seed run during
development suggested risk-controlled forwarding pulled ahead under churn; it
does not.

**And the experiment could not have shown otherwise, which is a fault in the
experiment.** Departure and relocation both withdraw the producer's FIB route,
and withdrawing a route also invalidates every Embedding Store entry that
pointed at it — the consistency rule SAF specifies, implemented here, applied
identically for every strategy. So the stale mapping is destroyed *by the event*
before any producer can be given the chance to refuse it. The signal
verification exists to catch is erased by the same mechanism that reports the
event. Only feedback can detect a relocation, but nothing was left for feedback
to detect.

What this arm does establish is that the invalidation path is now under load,
and that no strategy in this family tolerates a network reorganising itself
every second.

### Schema drift: the event that leaves only feedback

A producer keeps its routes, its attachment and its FIB entries, and quietly
stops recognising a subset of the wordings it used to answer — its declared
alias set shrinks. No route is withdrawn, nothing is invalidated, no timeout
fires, no similarity score changes; the encoder's view of the world is exactly
what it was. The canonical term is never dropped, so exact matching is
untouched and the routing plane observes nothing at all. A producer's live
refusal is the only observable in the system.

What drift moves is **wasted work, not satisfaction**. A wording its producer
has stopped answering cannot be satisfied by anyone, so no strategy recovers it;
what separates them is how long they keep spending a round trip to be told no.
Hospital, share of Interests a producer refused:

| mean seconds between drifts | static | 10 s | 5 s | 2 s | 1 s |
|---|---:|---:|---:|---:|---:|
| SAF+ES | 0.045 | 0.050 | 0.056 | 0.074 | 0.103 |
| GS-NDN, no verification | 0.045 | 0.050 | 0.056 | 0.074 | 0.102 |
| GS-NDN | 0.034 | 0.039 | 0.046 | 0.064 | **0.094** |
| Risk-controlled | 0.023 | 0.027 | 0.032 | 0.045 | **0.066** |
| Risk-controlled, adaptive ε | 0.027 | 0.029 | 0.035 | 0.048 | 0.071 |

with satisfaction 0.880 / 0.880 / 0.886 / 0.855 / 0.864 at one second.

Those satisfaction gaps are small next to their own confidence intervals, but
every arm sees the same workload, the same drift events and the same producers,
so the seed-to-seed variation the intervals are made of is shared and cancels.
Paired seed by seed at the hardest point (`--experiment drift_paired`,
twenty seeds, differences against GS-NDN without verification):

| | ISR | producer refusals | seeds won |
|---|---:|---:|---:|
| GS-NDN, hospital | **+0.0052 ± 0.0013** | −0.0083 ± 0.0016 | **20 / 20** |
| GS-NDN, city | **+0.0101 ± 0.0036** | −0.0171 ± 0.0039 | **19 / 20** |
| SAF+ES, hospital | −0.0000 ± 0.0002 | +0.0001 ± 0.0002 | 6 / 20 |
| Risk-controlled, hospital | −0.0252 ± 0.0085 | −0.0361 ± 0.0082 | 0 / 20 |

**What drift restores is verification's advantage, which relocation erases.**
The advantage over not verifying is +0.0080 satisfaction on a static hospital
network — the 0.8 points the ablation already reports. Under relocation at one
second it is +0.0005: gone, because the route withdrawal destroyed the mapping
first. Under schema drift at one second it is **+0.0052 ± 0.0013, on twenty
seeds out of twenty**, with 8% fewer Interests wasted on a producer that will
refuse; city gives +0.0101 ± 0.0036 on nineteen of twenty and 14% fewer. So
drift keeps about two thirds of the static advantage under a churn rate that
otherwise destroys it, and the claim this experiment can support is that
**feedback survives an event nothing else can see** — not that churn is where
verification finally becomes large. It does not become large anywhere. This is
the second leg of the zero-tuning contribution in §6: a threshold-only design
has no live signal to detect drift with at all, verified or not.

SAF+ES lands on top of GS-NDN-without-verification to four decimal places, which
is the sanity check the arm needed: both cache on resolution and neither
retracts, so under this event they are the same system, and they measure as it.

The risk controller trades in the other direction, and hard: 36% fewer refused
Interests than SAF+ES at one second (0.066 against 0.103) for two and a half
points of satisfaction. That is the same trade section 6 reports, and drift
sharpens it rather than changing it. City behaves the same way — verification
+0.0101 ± 0.0036 satisfaction on 19 of 20 seeds, refusals 0.101 against 0.118.

### Does an adaptive budget help?

Section 14 has been listing exchangeability as a known hole: conformal-style
bounds assume the calibration and test distributions match, and churn violates
that. Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021) is the
principled fix, and `rc-ndn-aci` implements it — the budget itself moves,
ε<sub>t+1</sub> = ε<sub>t</sub> + γ(ε<sub>target</sub> − err<sub>t</sub>), where
err<sub>t</sub> is the miscoverage indicator of the decision just judged. Only
decisions the boundary actually covered drive it; exploratory forwards below the
boundary are decisions the budget did not cover (section 7) and feeding them in
would have the controller punish itself for gathering evidence.

**The mechanism does what it says.** Effective ε with a target of 0.2, hospital:

| mean seconds between events | static | 10 s | 5 s | 2 s | 1 s |
|---|---:|---:|---:|---:|---:|
| under schema drift | 0.197 | 0.182 | 0.174 | 0.160 | **0.149** |
| under relocation churn | 0.197 | 0.242 | 0.266 | 0.341 | **0.364** |

Drift produces real coverage errors, so the budget tightens with the drift rate
— exactly the intended direction, and the first direct evidence in this work
that the adaptive loop is doing anything at all. City tightens the same way,
0.186 → 0.135.

**The benefit does not follow the mechanism.** Paired against static rc-ndn on
identical seeds, schema drift at one second:

| | ISR | producer refusals | seeds won |
|---|---:|---:|---:|
| hospital | +0.0091 ± 0.0046 | +0.0045 ± 0.0029 | 17 / 20 |
| city | +0.0007 ± 0.0055 | +0.0017 ± 0.0025 | **10 / 20** |

Nine tenths of a point of satisfaction on hospital — resolved, but tiny, and
paid for with slightly more wasted round trips. On city it is a coin flip — ten
seeds out of twenty, an interval eight times the effect. **Adaptation does not
give a consistent edge over a static budget under drift, and this is reported as
the weak result it is.** One domain of two, one point of satisfaction, and the
sign of the refusal difference is against it in both.

**And under relocation churn it drifts the wrong way, which is the more
interesting finding.** Route withdrawals destroy the mappings before they can be
refuted (above), so the decisions that survive to be judged are mostly correct,
and the update rule reads a clean stretch and loosens. At one second between
relocations the effective budget has climbed to 0.364 on hospital and 0.501 on
city, the latter past the top of the range section 6 sweeps at all. Realised error
rises with it, 0.017 → 0.027 on hospital, and stays far inside 0.2 only because
this catalog cannot produce much error at any setting. **The number the operator
set has stopped being the number in force**, and nothing in the mechanism
reports that. ACI's guarantee is asymptotic coverage, not a per-run bound, and
this is what that distinction costs on a network reorganising itself every
second. Clipping the range is a floor and a ceiling, not a fix.

## 9. Compromised routers

### Threat model

**What the attacker controls.** A share of the routers, chosen uniformly at
random and fixed for the run. A compromised router is a full participant, not
an outsider: it holds the keys and the peer relationships of an honest router,
so nothing here is defeated by authenticating the channel. It behaves correctly
on the forwarding path — it forwards Interests, returns Data, and answers
digests — and lies only in what it contributes to gossip. That is deliberate: a
router that simply drops traffic is a denial-of-service problem the ICN
literature already treats, and it would confound the measurement of what the
*learning* channel is worth.

**What it can do.** Inject false mappings (a wording bound to a producer that
does not serve it) and false calibration evidence (fabricated observations that
a score succeeded), and re-inject both every round, so retraction after one
round trip does not settle anything. It knows the catalog well enough to target
names clients actually request, which is the strong assumption here and makes
the attack far more effective than random poisoning.

**What it cannot do.** Forge another router's identity, partition the network,
observe or alter traffic it does not carry, or make a producer answer a name it
does not publish. A producer is trusted about its own services throughout; the
attack is on what routers tell *each other*, not on what a producer says. That
boundary matters for §16 in particular: the refusal reason is authored by the
producer, so a compromised **router** cannot forge one, and nothing here
measures what a compromised **producer** could do with it.

**What limits the damage, structurally.** A router's own confirmed mappings
outrank anything it is told. That is not a defence added against this attack;
it falls out of only gossiping what a returned Data packet proved, and it is
why the degradation below is graceful rather than total.

**What is not implemented, and why it is a limitation and not a design choice.**
Provenance (signing an observation with the router that made it, so a lie can be
attributed) and reputation (down-weighting a peer whose contributions are
repeatedly refuted) are the two obvious defences, and neither is here. The
budget in §6 is a guarantee conditional on honest reporting, and this section
measures what that condition is worth rather than removing it. A deployment
facing this threat model needs those mechanisms; this work does not show they
would be sufficient, only that the attack lands and by how much.

### What the attack does

Two things travel over gossip and each is poisoned differently. A **false
mapping** sends Interests to the wrong producer and is exposed by that
producer's refusal. **False calibration evidence** is quieter: fabricated "this
score worked" observations pull a boundary down so the victim accepts
resolutions it would have refused, and no individual decision looks wrong.

Hospital, share of routers compromised, attacker re-injecting every gossip round
against names clients actually request:

| compromised | 0% | 12.5% | 25% | 50% |
|---|---:|---:|---:|---:|
| GS-NDN satisfaction | 0.940 | 0.832 | 0.783 | 0.745 |
| Risk-controlled satisfaction | 0.919 | 0.822 | 0.780 | 0.754 |
| GS-NDN realised error | 0.027 | 0.103 | 0.142 | 0.168 |
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

## 10. Encoder heterogeneity

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

### What the guard is actually worth

That argument was asserted for one version of this document and never tested:
only the arm that refuses to pool was run, so "graceful" had nothing to be
graceful against. `rc-ndn-naive-mix` is the missing baseline — the same
controller with the guard removed, gossiping scores across the encoder boundary
as though a MiniLM 0.62 and an n-gram 0.62 were the same number. It has to be a
risk-controlled arm: GS-NDN gossips mappings and no scores at all, so a naive
version of it would have nothing to mix.

Hospital, 8 edge routers, twenty seeds:

| lexical share | 0% | 25% | 50% |
|---|---:|---:|---:|
| ISR, guard on | 0.919 | 0.911 | 0.882 |
| ISR, guard off | 0.919 | 0.911 | **0.882** |
| realised error, guard on | 0.014 | 0.014 | 0.014 |
| realised error, guard off | 0.014 | 0.014 | **0.015** |
| mean learned boundary, guard on | 0.776 | 0.829 | **0.866** |
| mean learned boundary, guard off | 0.776 | 0.781 | **0.807** |
| evidence dropped / mixed | 0 / 0 | 389 / 0 | 320 / 0 |

**Pooling incomparable scores measurably corrupts the boundaries and does not
measurably hurt anything else.** The learned boundary is where the two arms
separate cleanly: at 50% heterogeneity it sits 0.06 lower on hospital (0.807
against 0.866) and 0.09 lower on city (0.681 against 0.768), because n-gram
scores are distributed higher and drag the pooled estimate down. That is the
predicted failure, visible and unambiguous.

It does not reach the outcome. Satisfaction is identical to three decimals on
hospital (0.8817 ± 0.0166 against 0.8821 ± 0.0161 at 50%) and within noise on
city; realised error differs by 0.0013 at 50% hospital with confidence intervals
of ±0.005 on both. **The naive baseline does not degrade worse, and the
expectation that it would is not borne out.** The guard prevents a
mis-calibration that these two catalogs are too forgiving to punish — every
threshold on either domain realises under 0.035 error (section 6), so a boundary
displaced by 0.06 still lands somewhere safe. On a catalog where the boundary
had to be precise the corruption would presumably cost something; that is a
conjecture, and it is not what was measured here.

The guard stays, because a mechanism that is right for a stated reason and free
in practice is worth keeping, and because a deployment cannot know in advance
that its catalog is one of the forgiving ones. But the earlier phrasing invited
a reader to think refusing to pool had been shown to matter, and it has not.

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
  declarations are deliberately incomplete: measured in section 15, at
  `alias_coverage=0.7` a producer leaves 29% of the wordings it can be asked by
  undeclared, and satisfaction falls 0.940 → 0.826 as coverage drops to 0.5.
  What is not modelled is a producer whose schema is adversarially wrong.
- **The budget is conditional on honest reporting.** Section 9 measures what
  that condition is worth; it does not remove it.
- **Exchangeability.** Conformal-style bounds assume the calibration and test
  distributions match. Zipf popularity and producer churn both violate this. The
  hierarchical estimator and the bounded observation window mitigate it. The
  adaptive variant is now implemented and measured (section 8): it moves the
  budget in the right direction under schema drift, gives an edge on one domain
  of two, and under relocation churn drifts the effective budget to 0.36–0.50
  against a target of 0.2. It replaces a per-run bound with asymptotic coverage,
  and on a fast-churning network that is a real loss of what the operator's
  number meant.
- **The catalogs are generated from a lexicon**, not collected from a
  deployment. The rewrite families mirror how IoT namespaces diverge, but they
  are still our idea of that. Part of this has now been measured rather than
  conceded — see below — and the measurement is not flattering.
- **One transformer.** SAF compares MiniLM-L6, MiniLM-L12 and MPNet and finds
  recognition quality nearly identical; only L6 is used here, alongside the
  lexical control.
- **Source weighting against poisoned evidence is not implemented.** Section 9
  shows a compromised router's fabricated observations moving a victim's
  boundary, and names provenance and reputation as the obvious next step. A
  k-source quorum -- accept a boundary shift only when independent routers
  agree -- is the specific mechanism, and it remains unwritten.

### How much of the vocabulary is ours

The synonym objection can be partly measured instead of only conceded. The
grounded catalogs (`hospital-grounded`, `city-grounded`) add a seventh rewording
per service, taken from **Brick Schema 1.3, ETSI SAREF, Project Haystack 4 or
W3C/OGC SSN/SOSA** where those vocabularies name the quantity at all
(`gsndn/datasets/ontology.py`, one citation per entry, transcribed by hand — no
ontology file is fetched at build time). They are separate domains rather than a
redefinition of the reported ones, so nothing above moves.

Coverage is uneven and the catalog reports its own share. These are building and
sensing vocabularies: **20 of 50 hospital services** get a standardised wording
(temperature, humidity, air quality, occupancy) and **45 of 50 city services**
do. Heart rate, blood pressure, oxygen, respiration, glucose, infusion and bus
timetables have no class in any of the four, keep their invented synonyms and
are counted separately as fallbacks rather than averaged in.

Rank-1 recognition — does a cosine search over the 50-entry FIB put the right
service first — by rewrite family, restricted to the services a standard
actually names so that the same services are being compared:

| | synonym | abbreviated | flattened | alt_synonym | reordered | verbose | **ontology** |
|---|---:|---:|---:|---:|---:|---:|---:|
| hospital (n = 20 each) | 0.950 | 0.750 | 0.850 | 0.900 | 0.800 | 0.850 | **0.800** |
| city (n = 45 each) | 0.911 | 0.844 | 0.933 | 0.844 | 0.933 | 0.889 | **0.578** |

**On the city domain the standardised wordings are far harder than any we
invented** — 0.578 against 0.844–0.933, a gap of at least 27 points on exactly
the same services, and the mean score to the true service drops to 0.620 from
0.626–0.743. The reason is visible in the terms: Brick calls a crowd sensor
`Occupancy_Count_Sensor`, a waste-bin monitor `Level_Sensor`, a traffic detector
`MotionSensor`. Those are correct and they are nothing like what a city
integrator writes, which is precisely the kind of synonymy our lexicon does not
contain. On the hospital domain the ontology family sits at 0.800 against
0.750–0.950 for the invented ones — inside the range, at the lower end.

**So the reviewer's suspicion is confirmed on one domain of two.** The invented
rewrite families are easier than published class names wherever the standard's
vocabulary diverges from the domain's colloquial one, and every recognition
number above is measured on the easier set. End to end on the grounded catalogs
the ordering between strategies is unchanged — hospital-grounded ISR 0.920
(SAF+ES) / 0.938 (GS-NDN) / 0.902 (rc-ndn at ε = 0.2), city-grounded 0.906 /
0.922 / 0.874, twenty seeds — but those are different datasets with a seventh
wording per service and are not comparable line-for-line with section 1.

**And this fixes only half of what it was aimed at.** It grounds *class
synonyms*: the words for the thing being measured. It does nothing for how a
client actually phrases a request — word order, abbreviation, local convention,
whatever the integrator typed at three in the morning — because no published
ontology describes that and no public NDN trace exists to supply it. The
phrasing axis is exactly as invented as it was.

## 15. What an incomplete producer declaration costs

The feedback channel every verified strategy calibrates against is a producer
deciding from the schema it declared for itself. That declaration is allowed to
be incomplete, and section 14 leans on how incomplete. This section measures it
rather than asserting it. Twenty seeds, eight edge routers, ε = 0.2.

`undeclared` is a property of the declarations alone — the share of the wordings
a service can be asked by that it did not declare — read off the schemas without
running any traffic. It tracks 1 − coverage closely, which is what the stable-hash
draw in `admission.build_schemas` should produce, and is the check that the knob
does what its name says.

| coverage | undeclared | SAF+ES | GS-NDN | RC-NDN |
|---|---:|---:|---:|---:|
| 1.0 | 0.000 | 0.932 | **0.940** | 0.919 |
| 0.9 | 0.100 | 0.911 | **0.917** | 0.864 |
| 0.7 | 0.292 | 0.871 | **0.874** | 0.801 |
| 0.5 | 0.505 | 0.827 | 0.826 | 0.762 |

City, same protocol: undeclared 0.000 / 0.103 / 0.301 / 0.510; GS-NDN 0.956 →
0.917 → 0.849 → 0.776.

**The cost is producer-side, and it lands on everyone.** SAF+ES never consults
feedback at all and loses almost exactly what GS-NDN loses (0.932 → 0.827 against
0.940 → 0.826), which is the sign that this is a producer refusing requests, not
a strategy mislearning from them. Verification's margin over the cache narrows as
coverage drops — +0.008 at full coverage, −0.001 at 0.5 — because what
verification exploits is a producer's "no" being *informative*, and a producer
that has declared half its vocabulary says "no" to correct routes too.

**Risk control pays the most for it, and the mechanism is visible in the refusal
rate.** RC-NDN falls furthest (0.919 → 0.762 on hospital, 0.956 → 0.710 on city)
while its producer refusal rate stays *lowest* of the three (0.051 against 0.162
and 0.150 at coverage 0.5). It is not being refused more; it is forwarding less.
False refusals enter the calibration windows as genuine errors, the Wilson upper
bound rises, every route's boundary tightens, and the controller declines
decisions that would in fact have been served. This is the honest limit of
calibrating from live feedback: **the guarantee is against the feedback channel,
not against the ground truth, and a systematically incomplete declaration biases
the two apart.** ε is still held — the realised error is computed against the
same channel — but what the operator gets for it is worth less.

Not modelled: a producer whose declaration is adversarially wrong rather than
merely narrow. Section 9 is the closest thing, and it attacks the gossip channel
rather than the declaration.

## 16. The free labels are biased, not just noisy

Section 15 measured what an incomplete producer declaration costs and found the
risk-controlled arm losing most: 0.919 → 0.762 on hospital as coverage drops to
0.5, while its *refusal rate stays lowest of the three*. It was not being refused
more. It was forwarding less. This section is why, and what happens when the
cause is removed.

### The cause

A producer refuses for two reasons that mean different things about the route.
It may not publish the name at all — the mapping that sent the Interest here is
wrong, and that is a routing fact. Or it may publish the name and not recognise
the client's *phrasing*, because the phrasing is outside the schema it declared.
Then the route was right and only the vocabulary failed.

Only the producer can tell these apart, and it was discarding the distinction:
`Producer.on_interest` refused both with the same reason. Every gap in a
producer's declared vocabulary therefore arrived at the forwarding plane as
evidence against a route that was in fact correct. For `gs-ndn` that blacklists
the right prefix, so the next attempt is forced to exclude the correct producer
and pick a worse one. For `rc-ndn` it is worse: the false refusal enters the
route's calibration window as a genuine error, the Wilson upper bound rises, the
boundary tightens, and the controller starts declining decisions that would have
been served.

The fix is a refusal reason on the Nack — `no-such-service` against
`unknown-wording` — and a controller that does not count the second as evidence
about the route's score. HTTP separates 404 from 406 and DNS separates NXDOMAIN
from NODATA for the same reason. NDN's Nack has no such distinction, and
semantic forwarding is what makes one necessary. The arms below are
`gs-ndn-reasons` and `rc-ndn-reasons`; everything else is held fixed.

### Where it pays

Satisfaction against declared alias coverage, eight edge routers, ε = 0.2:

| coverage | undeclared | rc-ndn | rc-ndn-reasons | Δ | gs-ndn | gs-ndn-reasons | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.000 | 0.919 | 0.947 | +0.027 | 0.940 | 0.932 | −0.008 |
| 0.9 | 0.100 | 0.864 | 0.924 | +0.061 | 0.917 | 0.911 | −0.006 |
| 0.7 | 0.292 | 0.801 | 0.880 | **+0.079** | 0.874 | 0.870 | −0.004 |
| 0.5 | 0.505 | 0.762 | 0.839 | +0.076 | 0.826 | 0.827 | +0.001 |

City, same protocol: rc-ndn 0.956 / 0.854 / 0.759 / 0.710 against reason-aware
0.939 / 0.900 / 0.841 / 0.772 — deltas −0.017, +0.046, **+0.082**, +0.062.

**The effect has a regime, and that is the evidence it is the mechanism claimed.**
The gain grows as declarations get worse and peaks where roughly a third of
wordings are undeclared, because the correction can only recover what false
refusals were costing. Where there is nothing to correct — city at full coverage
— it loses 0.017, since skipping those observations only costs the controller
evidence. A correction that helped uniformly would be weaker evidence, not
stronger: it would not be behaving like a correction to this particular bias.

`gs-ndn-reasons` is the control and behaves like one, within ±0.008 throughout.
It has no calibration windows to protect, so reading the reason buys it nothing
and giving up the blacklist costs it a little. **The damage was in the evidence
path, not the refutation path.**

The same shape appears against the budget rather than against coverage. Hospital
satisfaction, full coverage:

| ε | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rc-ndn | 0.818 | 0.818 | 0.841 | 0.885 | 0.919 | **0.959** | **0.967** |
| rc-ndn-reasons | 0.821 | 0.848 | **0.928** | **0.946** | **0.947** | 0.949 | 0.952 |

Tight budgets gain up to 0.087, loose budgets lose up to 0.015, and city agrees.
Same account: a tight budget is where one false refusal does most damage, because
it pushes a controller already close to refusing everything over the edge. A
loose budget has nothing for the correction to unlock. **The realised error stays
inside the budget in all fourteen settings on both domains**, so the guarantee
survives the correction.

### Where it does not

Against the opponent of §6 — a threshold tuned on one domain and carried to the
other — reading the reason makes things **worse**, over fourteen comparisons:

| | trade | transferred threshold dominates | rc dominates |
|---|---:|---:|---:|
| rc-ndn | 10 | 4 | 0 |
| rc-ndn-reasons | 3 | **11** | 0 |

The mechanism is visible in the error column. Not counting unknown-wording
refusals makes the controller more permissive, so satisfaction *and* realised
error both rise — and the opponent is already sitting at satisfaction 0.97 with
error 0.011. Section 5 is why it is that strong: precision stays above 0.99
across the entire threshold sweep, so on these catalogs a permissive fixed
threshold is very nearly free.

**So §6's withdrawal stands.** This corrects a real bias in the label channel and
is worth up to 0.08 satisfaction where declarations are incomplete or budgets are
tight. It does not make a learned budget beat a tuned threshold, and on the
transfer comparison it gives up ground. The honest scope is narrower than the
first five-seed run suggested, and it is reported here rather than at the seed
count that flattered it.

### What this does not settle

The catalogs here are generated, so `alias_coverage` is a knob rather than a
measurement of how completely real operators declare their services. Nothing
here says where a real deployment sits on that axis, and the whole value of the
correction depends on it. That is the first thing a real name corpus would
settle.
