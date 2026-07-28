# Risk-Controlled Semantic Forwarding in NDN

When a client asks for `/hospital/temp-sensor/room-101` and the network only
knows `/smart-hospital/building-a/floor-1/temperature/room-101`, exact-match
forwarding drops the request. Embedding the name and matching it by cosine
similarity fixes that — at the cost of running a transformer in the forwarding
path, and of deciding how similar is similar enough.

That second cost is the one nobody has paid properly. Existing work picks a
similarity threshold by tuning it on one catalog. This work replaces the
threshold with an error budget the network holds itself to.

**Authors:** Mohammad Mahdi Yari, Sajjad Taghizadeh · **Advisor:** Dr. Mohammadreza Shakournia

---

## The argument

**A tuned threshold does not transfer.** SAF selects 0.7 on its own catalog. On
the two catalogs here that setting gives recall of 0.66 and 0.50, while the best
operating points sit at 0.55 and 0.45 — different from SAF's and different from
each other. A threshold is a property of the catalog it was tuned on.

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

So: **the operator sets an error budget ε instead of a threshold**, each route
learns its own decision boundary from observed outcomes, and routers pool their
evidence by gossip because no single router sees enough of it alone.

## What comes out

The budget holds. Measured out of sample, over exactly the decisions it governs:

| ε | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| realised error | 0.009 | 0.009 | 0.009 | 0.010 | 0.014 | 0.025 | 0.035 |
| satisfaction | 0.818 | 0.818 | 0.841 | 0.885 | 0.919 | 0.959 | 0.967 |

It is a trade, not a free win. At ε = 0.2 the risk-controlled plane errs at
**half the rate** of a tuned-threshold GS-NDN — 0.014 against 0.027 — for two
points less satisfaction, 0.919 against 0.940. What ε buys is the ability to
move along that curve without retuning anything, and to know where on it you
are. A single-seed run early in development showed it winning on both axes at
once; twenty seeds did not reproduce that, and the claim was withdrawn.

**It does not beat a threshold tuned on a labelled catalog, even someone else's.**
Put the budget against a fixed threshold tuned on one domain and carried to the
other, and rc-ndn Pareto-dominates in none of fourteen comparisons; the
transferred threshold dominates in four. What it does is hold its budget where
the transferred threshold missed one — at ε = 0.02, tuned on city, realising
0.032 on hospital — and get there without a labelled sample of the domain it
runs on. That is the claim, and it is narrower than efficiency. See
[`RESULTS.md`](RESULTS.md) §4.

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

**Not ours: the shape of the idea.** Learning per-item boundaries against an
error bound instead of a fixed threshold is what
[vCache](https://arxiv.org/abs/2502.03771) does for LLM prompt caching, and
pooling calibration across parties is
[federated conformal prediction](https://arxiv.org/abs/2305.17564).

**Ours:** that a forwarding plane generates its own calibration labels for free,
that per-route budgets can therefore be held online without a coordinator, that
doing so deadlocks without controlled exploration, and that mappings and scores
have different portability — a route that served a name serves it however anyone
asked, but a score means nothing outside the embedding space that produced it.

## Layout

```
gsndn/
  datasets/      two labelled name catalogs, built from an explicit lexicon
  admission.py   what a producer knows about its own services
  risk.py        per-route boundaries from an error budget
  strategies/    Vanilla NDN, SAF, SAF+ES, SEF, GS-NDN, RC-NDN and ablations
  gossip.py      anti-entropy over verified mappings and calibration evidence
  churn.py       producers that depart, return and relocate
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
tests/           36 tests, most guarding a specific mistake made while building this
```

## Running it

```bash
pip install -r requirements.txt

python experiments/fetch_model.py                                # ~80 MB, cached outside the repo
python experiments/export_embeddings.py --all                    # encode every catalog name once
python experiments/export_embeddings.py --all --backend lexical  # the no-transformer control
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
`alias_coverage=0.7` a producer refuses 29% of requests genuinely meant for it,
and satisfaction falls from 0.936 to 0.879 as coverage drops to 0.5. The
feedback channel is informative, not correct.

*Exploration is a real cost.* The 5% of refused decisions spent on evidence are
decisions the budget explicitly did not cover, and they are counted separately
rather than folded into the reported rate.

*Poisoning degrades gracefully; it is not prevented, and risk control does not
help.* Against a persistent attacker re-injecting every gossip round,
satisfaction falls from 0.940 to 0.747 at 50% compromise and the realised error
rises to 0.17 — the budget is a guarantee conditional on honest reporting, and
that condition is exactly what the attack removes. All three strategies degrade
alike; what limits the damage is that a router's own confirmed mappings outrank
anything it is told. Provenance and reputation are left as future work.

*Churn hurts everything equally.* Producer mobility was expected to be where
verification finally earns its cost. Over twenty seeds it is not: at one second
between events every strategy lands within 0.02 of the others (0.368 to 0.385).
Verification does not confer a measurable advantage even here, and this is
reported rather than quietly dropped.

*Gossip loses when there is nobody to share with.* On a single edge router it is
a net cost; the benefit appears from about four edge routers upward.

## Sources

- Amadeo et al., *Enhancing IoT Service Discovery Through Semantic Name-Based
  Forwarding*, IEEE Internet of Things Magazine, 2026 — SAF, the Embedding
  Store, Th = 0.7, and the single-router evaluation this work extends.
- Raza et al., *INF-NDN IoT*, IEEE Access, 2024 — LDA semantic tags,
  distributed through a central Principal Node.
- Askar et al., *SEF*, CMC, 2024 — the energy model and the 20-seed protocol.
- Chan et al., *Fuzzy Interest Forwarding*, 2017 — Word2Vec component matching,
  applied at both Content Store and FIB.
- Zhu et al., *vCache: Verified Semantic Prompt Caching*, 2025.
- Lu et al., *Federated Conformal Predictors*, ICML 2023.
