# Results

All figures are 20-seed means with 95% confidence intervals, produced by
`experiments/run_experiments.py --all --seeds 20`. Latency and processing costs
come from `data/costs.json`, measured on the machine that ran the campaign by
`experiments/bench_micro.py`; nothing in the simulation uses an assumed latency.

Two independent domains are reported throughout — a smart hospital and a smart
city, 50 advertised services each, 350 resolvable Interest names and 350
unsatisfiable ones. Where the two disagree, that is said explicitly.

---

## 1. What the forwarding path actually costs

Measured single-threaded on the campaign machine:

| Operation | Cost | Relative |
|---|---:|---:|
| MiniLM-L6 inference, one name | 7.05 ms | 1× |
| Character n-gram encoder, one name | 0.096 ms | 73× cheaper |
| Cosine search over a 50-entry FIB | 0.0019 ms | 3,700× cheaper |
| Cosine search over a 1000-entry FIB | 0.027 ms | 260× cheaper |
| Embedding Store probe | 0.0001 ms | 70,000× cheaper |
| 256-bit signature from an embedding | 0.0076 ms | 930× cheaper |

**This table decides the design.** The similarity search is not the bottleneck
and cannot become one at any FIB size a router would plausibly hold — extending
the fit to 10,000 entries still puts it under 0.1 ms, an order of magnitude
below one inference. Approximate nearest-neighbour indexes, hierarchical
navigable graphs and hash-based lookup all optimise the 0.0019 ms. The only
quantity worth attacking is how often the 7.05 ms is paid at all.

## 2. Signatures cannot select routes

Nearest FIB entry by Hamming distance over random-projection signatures, scored
against the true cosine argmax on the 300 reworded hospital names:

| Width | Wire size | Correct entry ranked 1st | in top 3 | in top 8 |
|---:|---:|---:|---:|---:|
| 64 bits | 8 B | 0.327 | 0.563 | 0.793 |
| 128 bits | 16 B | 0.447 | 0.747 | 0.890 |
| 256 bits | 32 B | 0.597 | 0.823 | 0.953 |
| 512 bits | 64 B | 0.743 | 0.923 | 0.990 |
| 1024 bits | 128 B | 0.840 | 0.983 | 1.000 |

A forwarding plane that routes on the nearest signature would misroute two
Interests in three at 64 bits, and one in six even at 128 bytes per entry.
Service names inside one domain differ in a single component out of five, and
locality-sensitive hashing does not have the resolution to separate them.

This is a negative result about a design that looks attractive on paper, and it
is why the Interest tag here carries the resolved prefix as a string rather than
a signature. Signatures are kept for one job they are good at: anti-entropy
digests, at 32 bytes against the 1.5 KB a float32 embedding would cost.

## 3. Head-to-head

Six edge routers, 150 Interests/s for 60 s, threshold 0.6.

**Hospital**

| Strategy | ISR | Precision | F1 | Mean IRT | p95 IRT | Encoder runs |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla NDN | 0.599 | 1.000 | 0.749 | 65.3 ms | 71.2 ms | 0 |
| SAF | 0.934 | 0.993 | 0.962 | 65.8 ms | 77.9 ms | 4,114 |
| SAF+ES | 0.934 | 0.993 | 0.962 | 63.9 ms | 77.9 ms | 1,843 |
| **GS-NDN** | **0.942** | 0.992 | **0.966** | **63.7 ms** | **74.8 ms** | **1,562** |

**City**

| Strategy | ISR | Precision | F1 | Mean IRT | p95 IRT | Encoder runs |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla NDN | 0.599 | 1.000 | 0.749 | 65.3 ms | 71.2 ms | 0 |
| SAF | 0.940 | 0.997 | 0.968 | 65.3 ms | 77.0 ms | 4,110 |
| SAF+ES | 0.940 | 0.997 | 0.968 | 63.7 ms | 77.0 ms | 1,751 |
| **GS-NDN** | **0.959** | 0.998 | **0.978** | **63.3 ms** | **73.4 ms** | **1,470** |

Semantic forwarding is worth a great deal against exact matching: satisfaction
rises from 0.60 to about 0.94 because 40% of the offered traffic is reworded and
Vanilla NDN resolves none of it. Among the three semantic strategies the
accuracy difference is small — GS-NDN is ahead by 0.8 points on hospital and 1.9
on city — and the real separation is in work done: **GS-NDN runs 62% fewer
inferences than SAF and 15% fewer than SAF+ES at this size.**

## 4. Where the cost goes as the network grows

Encoder inferences per run, 30 s at 150 Interests/s, as edge routers increase:

| Edge routers | 1 | 2 | 4 | 6 | 8 | 12 | 16 | Growth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAF (hospital) | 2,022 | 2,045 | 2,062 | 2,070 | 2,073 | 2,077 | 2,079 | +3% |
| SAF+ES (hospital) | 833 | 901 | 1,006 | 1,087 | 1,155 | 1,259 | 1,330 | **+60%** |
| GS-NDN (hospital) | 868 | 886 | 868 | 885 | 905 | 926 | 952 | **+10%** |

Three distinct behaviours, and each has a clear cause.

SAF caches nothing, so it pays one inference per FIB miss no matter where the
miss happens — flat, and the most expensive.

SAF+ES caches per router. At one edge router that cache sees the whole request
stream and is highly effective; as traffic splits across more routers each cache
sees a thinner slice of it, so the same total traffic produces 60% more
inferences. **The Embedding Store's benefit erodes as the network grows** —
which is exactly the setting SAF's conclusion names as future work, having
evaluated a single router.

GS-NDN shares what each router proves, so a wording resolved anywhere is
available everywhere. Its cost stays near flat. At 16 edge routers it needs 28%
fewer inferences than SAF+ES and 54% fewer than SAF.

## 5. Latency under load

95th-percentile resolution time on SAF's own single-router topology, where the
whole offered load lands on one access-edge router:

| Interests/s | 30 | 60 | 100 | 150 | 200 | 250 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAF (hospital) | 79.1 | 82.1 | 84.6 | 89.6 | 98.3 | 116.9 | **172.2** |
| SAF+ES (hospital) | 77.9 | 77.9 | 77.9 | 78.2 | 79.7 | 80.9 | **82.2** |
| GS-NDN (hospital) | 77.9 | 77.9 | 77.9 | 78.5 | 80.1 | 81.4 | **82.8** |

SAF's processor utilisation reaches 0.90 at 300 Interests/s and its tail latency
more than doubles — the bottleneck the paper reports past roughly 210
Interests/s, reproduced here. Both cached strategies stay flat.

**This result belongs to the Embedding Store, not to us.** On one router there
is nobody to gossip with, and GS-NDN and SAF+ES are indistinguishable — 82.8 ms
against 82.2 ms, well inside the confidence interval. Caching fixes latency at a
single router; sharing fixes cost across many. Reporting the latency win as
ours would be taking credit for SAF's mechanism.

Spread across six edge routers, no strategy saturates at these rates and all
three are flat. The single-router topology is used here precisely because it is
the only one where the bottleneck exists.

## 6. Does the threshold transfer?

Precision, recall and F1 against the cosine threshold, GS-NDN, both domains:

| Threshold | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Recall (city) | 0.985 | 0.984 | 0.976 | 0.957 | 0.877 | **0.809** | 0.725 | 0.681 |
| Precision (city) | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | **0.999** | 1.000 | 1.000 |
| F1 (city) | 0.992 | 0.991 | 0.987 | 0.977 | 0.934 | **0.893** | 0.840 | 0.810 |

SAF selects Th = 0.7 by a tuning campaign on its own catalog. On these harder
name variations it costs 18 points of recall against the best operating point,
while buying almost no precision — the precision curve is flat above 0.99
throughout, because the distractors are far enough away in embedding space that
even a permissive threshold rejects them.

The best F1 sits near 0.45–0.50 here and near 0.55–0.60 on hospital. **A
threshold tuned on one catalog does not transfer to another**, which is an
argument against fixed global thresholds generally, and a reason to prefer
mechanisms — caching, verification, sharing — that make the threshold matter
less once a name has been resolved once.

## 7. Ablation

Eight edge routers, threshold 0.6, hospital.

| Variant | ISR | Precision | Encoder runs | Gossip bytes | Coverage |
|---|---:|---:|---:|---:|---:|
| GS-NDN | 0.941 | 0.992 | 1,594 | 447 KB | 0.80 |
| no gossip | 0.939 | 0.992 | 2,019 (+27%) | — | — |
| no verification | 0.933 | 0.992 | 1,501 (−6%) | 491 KB | 0.81 |
| anti-entropy only | 0.940 | 0.992 | 1,676 (+5%) | 414 KB | 0.64 |
| 5 s gossip period | 0.941 | 0.992 | 1,647 (+3%) | 319 KB (−29%) | 0.86 |

Reading these honestly:

**Gossip is the load-bearing mechanism.** Removing it costs 27% more inferences.

**Verification is a cost, not a saving.** Turning it off makes the system 6%
*cheaper* and 0.8 points less accurate. Retrying a refused resolution means
running the encoder again, so verification buys accuracy with work. Its other
role — keeping unproven guesses out of the gossip layer — is what makes sharing
safe rather than what makes it fast, and precision stays at 0.992 either way
here because the encoder's mistakes on this catalog are consistent rather than
random.

**Rumour pushes matter more for coverage than for cost.** Dropping them leaves
periodic anti-entropy as the only path: 5% more inferences, but coverage falls
from 0.80 to 0.64 — knowledge still spreads, just far more slowly.

**A slower gossip period is a good trade.** Ten times the interval uses 29%
fewer bytes for 3% more inferences, and reaches *higher* coverage, because a
longer period batches more into each exchange and lets the digest comparison
skip more of them.

## 8. Does a transformer earn its cost?

GS-NDN with MiniLM-L6 against the character n-gram control, which is 73× cheaper
per inference:

| Encoder | Threshold | ISR (hospital) | ISR (city) |
|---|---:|---:|---:|
| MiniLM-L6 | 0.6 | 0.939 | 0.957 |
| MiniLM-L6 | 0.7 | 0.866 | 0.803 |
| Character n-gram | 0.6 | 0.695 | 0.730 |
| Character n-gram | 0.7 | 0.652 | 0.643 |

Yes, decisively. The lexical encoder handles abbreviation and morphology —
`temperature-reading` against `temperature`, `bldg-a` against `building-a` — and
fails on genuine synonymy, which is most of the problem: `cardiac-monitor`
against `heart-rate-monitor` shares almost no character n-grams. It recovers
about a third of the gap between exact matching and MiniLM at 1/73 of the cost,
which makes it a reasonable fallback for a router that cannot host a
transformer, and not a replacement for one.

## 9. Energy

Semantic forwarding shifts the energy budget from radio to compute. Eight edge
routers, 30 s, idle draw charged over the same window for every strategy:

**Hospital**

| Strategy | Total | Radio | Compute | Compute share | ISR |
|---|---:|---:|---:|---:|---:|
| Vanilla NDN | 549.1 J | 6.27 J | 3.3 J | 0.6% | 0.598 |
| SAF | 734.8 J | 9.57 J | 213.7 J | 29.1% | 0.933 |
| SAF+ES | 654.4 J | 9.57 J | 120.9 J | 18.5% | 0.933 |
| **GS-NDN** | **632.7 J** | 9.60 J | **95.8 J** | **15.1%** | **0.938** |

Radio energy is essentially identical across the three semantic strategies —
9.6 J — because the same Interests cross the same links. Everything that
separates them is compute: GS-NDN spends 55% less than SAF and 21% less than
SAF+ES, tracking the inference counts directly. City behaves the same way
(78.0 J against 183.1 J and 99.4 J).

The comparison worth drawing from this table is not that semantic forwarding is
cheap. It is not: it costs 15–29% of the whole budget on a task where exact
matching spends 0.6%. The question is what that buys — satisfaction from 0.60 to
0.94 — and whether the bill can be reduced without giving that up.

The comparison against SEF is on a different axis and should not be read as a
ranking. SEF has no semantic layer: it resolves exactly what Vanilla NDN does
(ISR 0.59) and spends almost nothing on compute. Its contribution is choosing
among next hops by remaining battery, which matters in the dense mesh it was
designed for and is orthogonal to name resolution. The two could be combined;
nothing here tests that.

## 10. Is the simulator right?

A hand-written simulator can be perfectly self-consistent and still get NDN
wrong, and every comparison above would inherit the error without looking
suspicious. The transport layer — the part that is not our contribution — is
therefore checked against ndnSIM on the same topology, links and load, with
exact-match forwarding only:

| | n | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| ndnSIM | 4,351 | 53.72 ms | 70.08 ms | 70.18 ms | 70.21 ms |
| This model | 4,545 | 63.49 ms | **71.19 ms** | 71.20 ms | 71.20 ms |

The medians differ by 1.11 ms, of which 1.00 ms is the producer service time
this model charges and ndnSIM does not. **The unexplained residual is 0.11 ms.**

The means differ by more, and that is a difference in request pattern rather
than in transport: ndnSIM's Zipf consumer has all six consumers drawing from one
popularity distribution, so many Interests are satisfied by PIT aggregation
almost instantly, while this model's consumers hold overlapping but distinct
slices. The median is the quantity both setups define identically.

Semantic resolution has no ndnSIM counterpart and is not validated this way —
it is the contribution, not the baseline. See [`ndnsim/`](ndnsim/) for the
scenario, the two build fixes ndnSIM needs on GCC 13, and the comparison script.

## 11. What this does not show

- **Producer-side verification is assumed, not derived.** The simulation decides
  whether a producer serves a request using the catalog's ground truth, standing
  in for a producer that recognises requests for its own services. Without some
  such local check, a resolution that lands on the wrong producer is
  indistinguishable from one that lands on the right one.
- **Gossip is a net loss on a single router**, and the scaling table shows it:
  868 inferences against SAF+ES's 833 at one edge. The benefit appears from
  about four edge routers upward.
- **The catalogs are generated from a lexicon**, not collected from a
  deployment. The rewrite families are hand-authored to mirror how IoT
  namespaces diverge, but they are still our idea of that.
- **No FIB churn.** Routes never disappear mid-run, so the Embedding Store's
  invalidation path is implemented and tested but never exercised under load.
  A deployment with mobile producers would stress exactly the mechanism this
  campaign leaves idle.
- **One encoder.** SAF compares MiniLM-L6, MiniLM-L12 and MPNet and finds
  recognition quality nearly identical; that comparison is not repeated here.
