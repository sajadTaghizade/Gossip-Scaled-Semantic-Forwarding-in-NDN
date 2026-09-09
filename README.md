# Gossip-Scaled Semantic Forwarding in NDN

When a client asks for `/hospital/temp-sensor/room-101` and the network only
knows `/smart-hospital/building-a/floor-1/temperature/room-101`, exact-match
forwarding drops the request. Embedding the name and matching it by cosine
similarity fixes that — at the cost of running a transformer in the forwarding
path.

That cost does not disappear with per-router caching. The Embedding Store of
Amadeo et al. caches a resolution where it was made; split the same traffic
across more routers and each cache sees a thinner slice of it, so over a short
horizon the network re-runs the encoder more as it grows, not less — a failure
mode their own conclusion names as future work. This work shares a resolution,
once verified, over anti-entropy gossip, so the rest of the network does not
re-derive what one router has already confirmed.

**Two things this repository reports, and they are not the same kind of thing.**
The first is a scaling result with its scope measured rather than assumed:
sharing slows how fast recognition cost grows with the network, by an amount
that falls by roughly two thirds as the run lengthens and that reverses
entirely below about four edge routers (§2). The second is a defect found in
the feedback channel every verified strategy here calibrates against: a
producer's refusal conflates "this name is not mine" with "this name is mine
but I do not know that wording", so an incomplete declaration arrives as
evidence against routes that were correct. Separating the two reasons is worth
up to 0.08 satisfaction where declarations are incomplete or budgets are tight
(§16).

Riding on the shared substrate is a third thing that is **not** presented as a
contribution: an error budget replacing the similarity threshold. It was tested
against a threshold tuned on one catalog and carried to another, it lost, and
the efficiency claim was withdrawn (§6) — correcting the feedback defect above
does not rescue that comparison either. It is kept here as a measured component
with a narrow surviving property (it needs no labelled target catalog to tune)
and is reported as such.

**Authors:** Mohammad Mahdi Yari, Sajjad Taghizadeh · **Advisor:** Dr. Mohammadreza Shakournia

> **On the name "SAF".** Earlier revisions used SAF as shorthand for the
> semantic name-based forwarding of Amadeo et al. That collides with SAF,
> Stochastic Adaptive Forwarding (Posch, Rainer and Hellwagner, IEEE/ACM
> Transactions on Networking 25(2), 2017), which is what an NDN reader will
> assume. Prose now says "semantic forwarding" or names the authors; the
> strategy keys in code (`saf`, `saf+es`) are unchanged so that the result
> files stay readable, and mean the semantic scheme throughout.

---

## The argument

**Caching alone scales worse than sharing, on a horizon that has to be stated.**
SAF's Embedding Store caches a resolved name at the router that resolved it.
Split the same traffic across more edge routers and each cache sees a thinner
slice: over a 60-second run, encoder calls rise 60% from 1 to 16 edge routers
even though total traffic is unchanged — the erosion SAF's own conclusion names
as future work. Gossiping a verified resolution to every router, once, holds
that growth to 10% over the same range.

That much is independent of any threshold or error budget: it is a property of
how many times the encoder runs, not of where the cutoff is set. What it is
*not* independent of is the horizon and the network size, and both bound it.
A cold cache costs one inference per router per wording — N routers pay it N
times, but they pay it once — so over 600 seconds the same comparison is +14%
against +1%, and gossip's saving at 16 edge routers falls from 26% to 7.5%.
Below about four edge routers it is a net loss, because there is nobody to
share with. The claim that survives is narrower than "sharing scales and
caching does not": **sharing pays a network that is large and still learning
its catalog, and pays less the longer that network has been up.**

**A tuned threshold does not transfer, and this is the narrower problem
tackled on top.** SAF selects 0.7 on its own catalog. On the two catalogs
here that setting gives recall of 0.66 and 0.50, while the best operating
points sit at 0.55 and 0.45 — different from SAF's and different from each
other. A threshold is a property of the catalog it was tuned on.

**One number cannot serve every route anyway.** Some services sit alone in
embedding space and a loose score is safe; others have five near-identical
siblings and a high score is still a coin flip. A single cutoff is
simultaneously too strict for the first and too permissive for the second.

**The labels are free.** Every semantic forwarding decision is an experiment the
network answers: Data comes back when the producer recognises the request, a
Nack when it does not. That is exactly a calibration set, produced as a
by-product of forwarding. Verified semantic caching for LLM prompts needs a
judge model to obtain the same signal, which is expensive enough to be the thing
you were avoiding.

So the operator can set an error budget ε instead of a threshold: each route
learns its own decision boundary from observed outcomes, and the same gossip
channel that shares resolutions also pools the evidence that calibrates them.

**That was proposed as a second contribution and it did not survive testing.**
Against a threshold tuned on one catalog and carried to the other it
Pareto-dominates in none of fourteen comparisons (§6). What is reported instead
is what the attempt *exposed*: those free labels are not merely noisy, they are
**biased**, because a producer's refusal cannot distinguish a wrong route from
a wording it never declared. That is the finding this repository stands on
alongside the scaling result, and it is an instance of a pattern with a name —
missing-not-at-random feedback, familiar from implicit-feedback recommendation
(Saito et al., WSDM 2020) — located here in an NDN forwarding plane and traced
to a specific protocol gap.

## What comes out

**Sharing slows how fast cost grows with the network — the primary result,
and it is horizon-scoped.** Encoder inferences for the same workload over a
60-second run, edge routers 1 → 16:

| Edge routers | 1 | 2 | 4 | 8 | 16 | Growth |
|---|---:|---:|---:|---:|---:|---:|
| SAF (no cache) | 2,022 | 2,045 | 2,062 | 2,073 | 2,079 | +3% |
| SAF+ES (per-router cache) | 833 | 901 | 1,006 | 1,155 | 1,330 | **+60%** |
| GS-NDN (gossiped) | 875 | 893 | 877 | 914 | 959 | **+10%** |

SAF pays for every FIB miss, so there is nothing cached to erode and it barely
grows. SAF+ES caches locally and thins as routers multiply. Sharing what one
router has already resolved holds growth to a sixth of that. This claim is
independent of ε or any threshold — it is measured at a single fixed operating
point and holds regardless of it.

**Two scope conditions, both measured, both easy to miss from that table.**
It is a 60-second run, and the gap is partly a warm-up cost that amortises: over
600 seconds SAF+ES grows +14% rather than +55%, GS-NDN +1% rather than +9%, and
gossip's saving at 16 edge routers falls from 26% to 7.5%. The ordering never
reverses at scale, but the magnitude does — the headline is the short-horizon
figure. And **below about four edge routers gossip is a net loss**: at one edge
router there is nobody to share with and anti-entropy costs 4–5% more inferences
than a plain per-router cache. Sharing pays a network that is large and still
learning its catalog; it pays progressively less as that network settles. See
[`RESULTS.md`](RESULTS.md) §2.

**The error budget holds, but the claim that it forwards more efficiently was
tested and withdrawn.** Measured out of sample:

| ε | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| realised error | 0.009 | 0.009 | 0.009 | 0.010 | 0.014 | 0.025 | 0.035 |
| satisfaction | 0.818 | 0.818 | 0.841 | 0.885 | 0.919 | 0.959 | 0.967 |

Put against a fixed threshold tuned on one domain and carried to the other,
rc-ndn Pareto-dominates in none of fourteen comparisons; the transferred
threshold dominates in four, and reading the refusal reason (§16) makes that
tally 11 to 3 against, not better. What survives is narrower than efficiency —
call it **zero-tuning**: rc-ndn held its budget in all fourteen tests, where the
transferred threshold missed once, at the tightest budget (ε = 0.02, tuned on
city, realising 0.032 on hospital), and rc-ndn needs no labelled sample of the
domain it runs on to get there, because it calibrates from live producer
feedback instead. An operator who does have a labelled target-domain catalog
should tune a threshold on it and will do slightly better. See
[`RESULTS.md`](RESULTS.md) §6.

**Under schema drift — a producer quietly narrowing what it answers to,
without any route or cache event — the budget keeps most of the advantage
that route churn otherwise erases.** Paired seed by seed against a
non-verifying baseline: +0.0052 ± 0.0013 satisfaction on the hospital
catalog, 20 of 20 seeds; +0.0101 ± 0.0036 on the city catalog, 19 of 20.
Nothing in the routing plane observes a schema drift; feedback is the only
signal that can. See [`RESULTS.md`](RESULTS.md) §8.

**Risk control alone deadlocks.** A boundary set too high blocks exactly the
decisions that would produce the evidence to lower it, so the system stops
forwarding, stops learning, and stays there: left unexplored at ε ≤ 0.10 it
settles at refuse-everything on 104 observations and 0.738 satisfaction, and
tightening ε below 0.10 changes nothing because it is already refusing what it
can. Spending 5% of refused decisions on evidence gathers 598 and reaches
0.818.

Two further results, both in [`RESULTS.md`](RESULTS.md):

- **Encoder cost is the only cost.** One MiniLM-L6 inference takes 7.05 ms; a
  cosine search over a 50-entry FIB takes 0.0019 ms. Every scheme for speeding
  up the *search* optimises a rounding error.
- **Locality-sensitive hashing cannot route.** Nearest-by-Hamming agrees with
  the true cosine argmax 33% of the time at 64 bits, 84% at 1024. Signatures are
  kept for gossip digests and never for choosing a route.

## Honesty about what is new

**Not ours: edge tagging.** Attaching the resolved prefix to the Interest so
later hops skip the encoder is SAF's mechanism.

**Not ours: the shape of the risk-control idea.** Learning per-item boundaries
against an error bound instead of a fixed threshold is what
[vCache](https://arxiv.org/abs/2502.03771) does for LLM prompt caching, and
pooling calibration across parties is
[federated conformal prediction](https://arxiv.org/abs/2305.17564).

**Ours:** that sharing a verified resolution by gossip, rather than caching it
per router, is what keeps recognition cost from growing with the network —
measured against the erosion SAF's own conclusion leaves open, not asserted;
that the same verification signal lets a forwarding plane generate its own
calibration labels for free; that per-route error budgets can therefore be
held online without a coordinator, and that doing so deadlocks without
controlled exploration; and that mappings and scores have different
portability — a route that served a name serves it however anyone asked, but a
score means nothing outside the embedding space that produced it.

## Layout

```
gsndn/
  datasets/      two labelled name catalogs, built from an explicit lexicon,
                 plus grounded variants using Brick/SAREF/Haystack/SSN class names
  admission.py   what a producer knows about its own services
  risk.py        per-route boundaries from an error budget
  strategies/    Vanilla NDN, SAF, SAF+ES, SEF, GS-NDN, RC-NDN and ablations
  gossip.py      anti-entropy over verified mappings and calibration evidence
  churn.py       producers that depart, return, relocate and narrow their schema
  adversary.py   compromised routers, and how far their lies travel
  embeddings.py  MiniLM via ONNX, precomputed vectors, a lexical control
  des.py         discrete-event kernel with a single-server queue per router
  network.py     routers, links, producers, consumers, packet movement
  tables.py      FIB, PIT, Content Store, LRU Embedding Store
  simhash.py     signatures, with their measured limits documented
  energy.py      SEF's radio model, plus the cost of running an encoder
  metrics.py     correctness-aware scoring
  runner.py      assemble a scenario, run it, score it
experiments/     model fetch, embedding export, microbenchmarks, campaign, figures
ndnsim/          ns-3 cross-validation of the transport layer
tests/           50 tests, most guarding a specific mistake made while building this
```

## Running it

```bash
pip install -r requirements.txt

python experiments/fetch_model.py                                # ~80 MB, cached outside the repo
python experiments/export_embeddings.py --all                    # encode every catalog name once
python experiments/export_embeddings.py --all --backend lexical  # the no-transformer control
python experiments/export_embeddings.py --domain hospital-grounded \
    --cost-from data/embeddings/hospital.all-MiniLM-L6-v2-onnx.json   # ontology-grounded variant
python experiments/bench_micro.py                                # measure this machine's costs

python experiments/run_experiments.py --all --seeds 20
python experiments/make_figures.py
python -m pytest tests/ -q
```

Encoding is separated from simulation deliberately: it mirrors the deployment,
where FIB-entry embeddings are computed when routes are installed, and it means
every reported number reproduces without a model download or a GPU. The
transport model is cross-checked against ndnSIM — median round-trip times agree
to within 0.11 ms once the producer service time this model charges is
accounted for. See [`ndnsim/README.md`](ndnsim/README.md).

## How to read the results honestly

*Producer feedback is modelled, not assumed.* Each producer declares the terms
it answers to and the instance it serves, and decides on that alone — never on
the catalog's ground truth. Declarations are deliberately incomplete: at
`alias_coverage=0.7` a producer leaves 29% of the wordings it can be asked by
undeclared, and GS-NDN's satisfaction falls from 0.940 to 0.826 (city: 0.956 to
0.776) as coverage drops to 0.5. The feedback channel is informative, not
correct. See [`RESULTS.md`](RESULTS.md) §15.

*Exploration is a real cost.* The 5% of refused decisions spent on evidence are
decisions the budget explicitly did not cover, and they are counted separately
rather than folded into the reported rate.

*Poisoning degrades gracefully; it is not prevented, and risk control does not
help.* Against a persistent attacker re-injecting every gossip round,
GS-NDN's satisfaction falls from 0.940 to 0.747 at 50% compromise and its
realised error rises from 0.027 to 0.165 (risk-controlled: 0.919 to 0.754, and
0.014 to 0.170) — the budget is a guarantee conditional on honest reporting, and
that condition is exactly what the attack removes. All three strategies degrade
alike; what limits the damage is that a router's own confirmed mappings outrank
anything it is told. Provenance and reputation are left as future work.

*Relocation churn hurts everything equally; schema drift does not.* Producer
mobility was expected to be where verification finally earns its cost. Over
twenty seeds, departure and relocation are not: FIB withdrawal invalidates the
stale mapping before any producer gets the chance to refuse it, so at one
second between events every strategy lands within 0.02 of the others (0.368 to
0.385). Schema drift removes that confound — nothing is withdrawn, nothing is
invalidated — and there verification's advantage reappears: +0.0052 to
+0.0101 satisfaction, paired and statistically significant. Both are reported,
because which one an operator sees depends entirely on whether a given churn
event happens to touch the routing plane.

*Gossip loses when there is nobody to share with.* On a single edge router it is
a net cost; the benefit appears from about four edge routers upward.

## Related work

BibTeX for everything below is in [`references.bib`](references.bib); the
longer annotated list, with a verification status per entry, is in
[`references/README.md`](references/README.md). Entries there marked
*unverified* were assembled from search results rather than read.

**Semantic forwarding in ICN — the line this extends.**

- Amadeo et al., *Enhancing IoT Service Discovery Through Semantic Name-Based
  Forwarding*, IEEE Internet of Things Magazine, 2026 — the semantic scheme,
  the Embedding Store, Th = 0.7, and the single-router evaluation extended
  here. It is a magazine article, so short and tutorial-framed.
- Chan et al., *Fuzzy Interest Forwarding*, AINTEC 2017 — Word2Vec component
  matching at both Content Store and FIB. The original statement of this
  problem, and implemented in ndnSIM with released code.
- Raza et al., *INF-NDN IoT*, IEEE Access, 2024 — LDA semantic tags routed
  through central supernodes. A *competing* answer to the problem §2 attacks:
  concentrate semantic resolution rather than replicate or share it.
- Hlaing and Asaeda, *NeuName*, IEEE NetSoft 2026 `[hlaing2026neuname]` —
  neural semantic naming for ICN. **We could not obtain the full text** (IEEE
  paywall) and place it from title and venue only: it addresses how names are
  *generated*, where this work takes names as given and addresses how a
  forwarding decision over them is *calibrated*. That distinction should be
  re-checked against the paper before it is relied on.
- Askar et al., *SEF*, CMC, 2024 — energy model and 20-seed protocol. Cited for
  those; it is reinforcement-learning next-hop selection over geography and
  battery, with no semantic matching, and `gsndn/strategies/sef.py` says so.

**Synchronisation NDN already has — why the gossip layer is not novel
mechanism.** Anti-entropy dataset synchronisation is a decade old in NDN and
the layer here is an instance of it carrying different payload.

- Zhu and Afanasyev, *Let's ChronoSync*, 2013 `[zhu2013chronosync]` — condensed
  digests exchanged to reconcile dataset differences; PSync and State Vector
  Sync succeed it.
- Hoque et al., *NLSR*, ACM ICN workshop at SIGCOMM 2013 `[hoque2013nlsr]` —
  already floods name-prefix reachability network-wide over such a layer. §2
  measures an approximation of an unbounded sync layer (`gs-ndn-full-sync`)
  precisely because "why not just announce the prefix" is the first question
  this work has to answer.

**Learned boundaries under an error bound — why the budget is not novel
mechanism either.**

- Schroeder et al., *vCache*, 2025 `[schroeder2025vcache]` — per-item learned
  thresholds under a user-set error bound, for LLM prompt caching. The direct
  precedent for §6.
- Finamore et al., *Error-controlled Approximate-key Caching*, INFOCOM 2022
  `[finamore2022approximatekey]` — similarity caching of model outputs with
  explicit error control, in a networking setting. Closest published ancestor
  of both the premise and §6.
- Lu et al., *Federated Conformal Predictors*, ICML 2023 — pooling calibration
  across parties, which is what the evidence channel does.

**The bias in §16 is an instance of a known pattern.**

- Saito et al., *Unbiased Recommender Learning from Missing-Not-At-Random
  Implicit Feedback*, WSDM 2020 `[saito2020unbiased]` — learning from feedback
  observed only where the system chose to act, when the choice and the outcome
  share a cause. §16 is that pattern in an NDN forwarding plane: the label a
  route receives depends on a producer's declared vocabulary, which is also
  what determines whether the request could be served. The contribution is
  locating it here and identifying the protocol gap that causes it — a Nack
  with no equivalent of HTTP's 406 or DNS's NODATA — not the pattern itself.

**Name collision.**

- Posch, Rainer and Hellwagner, *SAF: Stochastic Adaptive Forwarding in NDN*,
  IEEE/ACM Transactions on Networking 25(2), 2017 `[posch2017saf]` — the
  established meaning of "SAF" in this literature. See the note at the top.
