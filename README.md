# GS-NDN — Gossip-based Semantic Named Data Networking

Semantic routing in NDN: when a client asks for `/hospital/temp-sensor/room-101`
and the network only knows `/smart-hospital/building-a/floor-1/temperature/room-101`,
exact-match forwarding drops the request. Embedding the name and matching it by
cosine similarity fixes that, at the cost of running a transformer in the
forwarding path.

This repository asks what that cost really is, and what to do about it.

**Authors:** Mohammad Mahdi Yari, Sajjad Taghizadeh · **Advisor:** Dr. Mohammadreza Shakournia

---

## The short version

Three things came out of measuring rather than assuming.

**The encoder is the whole cost; the search is nothing.** On this machine one
MiniLM-L6 inference takes 7.05 ms single-threaded, while a cosine search over a
50-entry FIB takes 0.0019 ms — a factor of 3,700. Approximate nearest-neighbour
indexes, hash-based lookup and every other way of speeding up the *search* are
optimising a rounding error. The only quantity worth reducing is how often the
encoder runs at all.

**Locality-sensitive hashing cannot route.** Picking the nearest FIB entry by
Hamming distance over signatures agrees with the true cosine argmax 33% of the
time at 64 bits and 84% at 1024. Service names within one domain differ in a
single component out of five, and LSH does not have the resolution to separate
them. Signatures are kept here for gossip digests — 32 bytes against a 1.5 KB
embedding — and never for choosing a route.

**A per-router cache stops paying as the network grows.** SAF's Embedding Store
is highly effective at one router, where it sees the whole request stream. Split
the same traffic across 16 edge routers and each cache sees a thinner slice of
it, so the network runs 60% more inferences for no more traffic. Sharing what
each router proves restores that: GS-NDN's cost rises 10% over the same range,
and at 16 edge routers it needs 28% fewer inferences than SAF+ES and 54% fewer
than SAF. Both effects replicate on two independent domains.

Under load the two cached strategies are indistinguishable, and that result
belongs to SAF rather than to us: at 300 Interests/s through a single access
router SAF's tail latency reaches 172 ms while both cached strategies sit near
82 ms. Caching fixes latency at one router; sharing fixes cost across many.

Full numbers, including where the approach does *not* help, are in
[`RESULTS.md`](RESULTS.md).

---

## What is actually new here

Two mechanisms, and an explicit note on what is inherited.

**Verification.** A semantic match is a guess. A producer that receives a
resolved Interest can see both the rewritten name and the client's original
wording, and can tell whether the request is for one of its services. When it
refuses, the router drops the mapping *and remembers which prefix failed*, so
the next attempt excludes it — an encoder is deterministic, and without that
exclusion a retry returns the same wrong answer. SAF caches on resolution and
never learns that the route it chose does not work.

**Sharing.** Only mappings a producer has actually served are gossiped, by
anti-entropy with per-peer version watermarks plus immediate rumour pushes.
Because nothing unproven travels, a permissive similarity threshold stops being
dangerous: mistakes are retracted locally before anyone else hears about them.

**Not ours: edge tagging.** Attaching the resolved prefix to the Interest so
that later hops skip the encoder is SAF's mechanism — the paper prepends it with
a `|` delimiter. It is implemented here because without it the producer never
sees a name it serves, but it is not claimed as a contribution.

---

## Layout

```
gsndn/
  datasets/      two labelled name catalogs, built from an explicit lexicon
  embeddings.py  MiniLM via ONNX, precomputed vectors, a lexical control
  des.py         discrete-event kernel with a single-server queue per router
  network.py     routers, links, producers, consumers, packet movement
  tables.py      FIB, PIT, Content Store, LRU Embedding Store
  strategies/    Vanilla NDN, SAF, SAF+ES, SEF, GS-NDN and its ablations
  gossip.py      anti-entropy and rumour spreading over verified mappings
  simhash.py     signatures, with their measured limits documented
  energy.py      SEF's radio model, plus the cost of running an encoder
  metrics.py     correctness-aware scoring
  runner.py      assemble a scenario, run it, score it
experiments/     model fetch, embedding export, microbenchmarks, campaign, figures
ndnsim/          ns-3 cross-validation of the transport layer, and its comparison script
tests/           36 tests, most guarding a specific mistake made while building this
data/            exported embeddings and the measured cost model
results/         campaign output and figures
```

## Running it

```bash
pip install -r requirements.txt

# One-time: fetch the MiniLM ONNX graph (~80 MB, cached outside the repo)
python experiments/fetch_model.py

# Encode every catalog name once; the simulator reads these, not the model
python experiments/export_embeddings.py --all
python experiments/export_embeddings.py --all --backend lexical

# Measure this machine's operation costs; without it, latencies are assumed
python experiments/bench_micro.py

# Run the campaign and draw the figures
python experiments/run_experiments.py --all --seeds 20
python experiments/make_figures.py

python -m pytest tests/ -q
```

To cross-check the transport model against a real NDN stack, see
[`ndnsim/README.md`](ndnsim/README.md). On the same topology and load the two
agree on median round-trip time to within 0.11 ms once the producer service time
this model charges — and ndnSIM does not — is accounted for.

Encoding is separated from simulation deliberately. It mirrors the deployment,
where FIB-entry embeddings are computed when routes are installed rather than
per Interest, and it means every reported number can be reproduced without a
model download or a GPU.

## How to read the results honestly

A few things are worth knowing before trusting any of it.

*Producer-side verification is a modelling assumption.* The simulation decides
whether a producer serves a request using the catalog's ground truth, standing
in for a producer that recognises requests for its own services. Without some
such local check, a resolution that lands on the wrong producer is
indistinguishable from one that lands on the right one, and no feedback
mechanism could separate them.

*Gossip loses when there is nobody to share with.* On a single edge router it is
a net cost, and the scaling figure shows it. The benefit appears from roughly
four edge routers upward.

*Latency differences need load.* Spread across several edge routers, the same
total offered load leaves each far from saturation and every strategy looks
alike. The latency experiment therefore uses SAF's own single-router topology,
which is where its bottleneck is.

*The threshold does not transfer.* SAF's Th = 0.7 gives recall of 0.66 on the
hospital catalog and 0.50 on the city one; best F1 sits nearer 0.55–0.60 on
both. Any comparison at a fixed threshold is a comparison at somebody's
operating point, not at the best one.

## Sources

- Amadeo et al., *Enhancing IoT Service Discovery Through Semantic Name-Based
  Forwarding*, IEEE Internet of Things Magazine, 2026 — SAF, the Embedding
  Store, Th = 0.7, and the single-router evaluation whose topology we extend.
- Raza et al., *INF-NDN IoT*, IEEE Access, 2024 — NLP name optimisation and
  LDA semantic tags, distributed through a central Principal Node.
- Askar et al., *SEF: A Smart and Energy-Aware Forwarding Strategy for
  NDN-Based Internet of Healthcare*, CMC, 2024 — the energy model and the
  20-seed evaluation protocol used here.
