# Extensions under evaluation

Added 2026-09-14. Nothing here is in `RESULTS.md` yet: the pilot numbers below
say how many seeds produced them, and the twenty-seed campaign has not been run.

**These changes alter what `gs-ndn` means.** Verification now reaches mappings
learned over gossip, which it did not before (§0.2). That was a deliberate
decision, taken because the alternative was shipping a paper whose central
"verify before you trust" claim held for half the system and silently failed for
the half the design is actually about. The consequence is that **every result
file in `results/` predates this change and must be regenerated**, including the
numbers currently in `RESULTS.md` and `Project_Status_Report.pdf`.

`gs-ndn-unverified-import` restores the old behaviour exactly, so the pre-fix
results remain reproducible on demand and the defect's cost stays measurable.

Every arm is **one variable** from its own baseline. `python -m pytest tests/ -q`
is 65 tests, of which 14 are new.

---

## 0. Two defects found while building these

Both were found by instrumenting rather than by reading, and both matter to
claims already in `RESULTS.md`.

### 0.1 A third cross-process determinism bug, in the attack path

Commit `73ad0ae` fixed two places where an interpreter-salted iteration order
leaked into results, and added a test that runs two child processes under
differing `PYTHONHASHSEED`. That test covers the honest path only. A third
instance survived in `adversary.py`:

```python
for router_id in self.compromised:      # a set of router ids
```

Injection order set event order. Across eight interpreter salts at a fixed seed
the run produced **two distinct outcomes**, differing by 0.006 ISR — an order of
magnitude larger than the ~0.03% counter drift the earlier two caused, and on an
outcome metric rather than a diagnostic one. **§9's published numbers were not
reproducible in a fresh process.**

Fixed by sorting; guarded by
`test_adversary_is_reproducible_across_interpreter_hash_seeds`, verified by
reverting.

### 0.2 Verification never ran on anything gossip taught us

This one is larger, because it undercuts a claim rather than a number.

A gossip cache hit in `GsNdn.resolve` returned a `Resolution` with no `learn=`.
So the PIT entry carried no `pending_learn`. So neither `on_confirmed` nor
`on_rejected` ever fired for a mapping learned from a peer — however many times
that mapping was used. Measured directly, 30 s, 8 edges, 25% compromised:

| | injected | gossip hits served | **retracted** |
|---|---:|---:|---:|
| `gs-ndn-unverified-import` | 608 | 897 | **0** |
| `gs-ndn-no-verify` | 608 | 910 | **0** |

`adversary.py` states that a victim "accepts the mapping and forwards on it,
gets refused by the producer, drops it, and does not pass it on... the lie
survives one round trip per victim." That describes the design. In the code the
lie survived the entire run, and every poisoned mapping is an imported one.

This explains a result §9 reports without explaining: `gs-ndn` and
`gs-ndn-no-verify` degrade almost identically under attack. They were not being
told apart, because on the poisoned mappings neither of them verified anything.

Fixed behind `GsNdn.verify_imported`, **on by default since this was found**.
`gs-ndn-unverified-import` restores the old behaviour and changes nothing else,
so the defect's cost stays measurable.

One thing this implies is worth carrying into the paper rather than burying: the
§7 ablation that found verification "costing 6% more work for 0.8 points of
satisfaction" was measuring locally resolved mappings only. That number is not
wrong, but it is answering a narrower question than its wording claims, and the
regenerated ablation will say so.

A second instance of the same defect was found immediately afterwards, and is
the reason `test_every_verifying_arm_checks_its_imported_mappings` is
parameterised over arms rather than testing one: `RiskControlledNdn` overrides
`resolve()` wholesale, so it kept the old cached branch and stayed undefended
while `gs-ndn-robust` worked. Any future strategy that overrides `resolve()` will
trip that test.

---

## 1. Verification-grounded peer reputation (`gsndn/reputation.py`)

**The gap.** §9 names its own missing defences: "Provenance ... and reputation
... are the two obvious defences, and neither is here."

**Why it is cheap here and not in the literature.** Byzantine-robust gossip
normally defends *without ground truth* — a neighbour ships a gradient and no
honest node can check it, so the defences are statistical (clipping, trimming,
bounded influence; e.g. EdgeClippedGossip, arXiv 2405.03449). A gossiped mapping
is different: it is a **falsifiable claim**, and NDN answers it. Forward on it
and the producer returns Data or refuses. That is the same free label §6
calibrates from, read a second time and attributed to whoever supplied the claim.

So the mechanism is detection, not just influence-bounding — and it is available
only because the forwarding plane already verifies.

**Two channels, two defences**, because §9 established they break differently:

| channel | failure mode | defence |
|---|---|---|
| mappings | falsifiable; re-injected every round | attribute refusals to the peer; never reinstall a pair disproved first-hand |
| evidence | not immediately falsifiable | cap any one peer at `peer_share` of a route's calibration window |

Trust is a Wilson **lower** bound on a peer's success rate, mirroring §6's Wilson
**upper** bound on error — both bound in the direction that costs us. A fresh
peer is fully trusted, deliberately: distrusting unaudited peers would break the
bootstrap gossip exists for. The cost is bounded and stated — `MIN_CLAIMS` free
lies per identity, paid once, not per round.

**Pilot (5 seeds, hospital, 60 s, 8 edges, ε = 0.2).** Not yet 20 seeds.
Arm names are the post-decision ones: `gs-ndn` verifies its imports,
`gs-ndn-unverified-import` is the behaviour that shipped before the fix.

| compromised | arm | ISR | realised error | poison live | peers distrusted |
|---|---|---:|---:|---:|---:|
| 0.0 | `gs-ndn-unverified-import` | 0.9405 | 0.0156 | 0 | 0 |
| 0.0 | `gs-ndn` | 0.9404 | 0.0147 | 0 | 0 |
| 0.0 | `gs-ndn-robust` | 0.9404 | 0.0140 | 0 | 0.2 |
| 0.125 | `gs-ndn-unverified-import` | 0.8323 | 0.0813 | 1814 | 0 |
| 0.125 | `gs-ndn` | 0.8956 | 0.0430 | 1598 | 0 |
| 0.125 | `gs-ndn-robust` | **0.9261** | **0.0238** | **502** | 8.0 |
| 0.25 | `gs-ndn-unverified-import` | 0.7864 | 0.1168 | 1885 | 0 |
| 0.25 | `gs-ndn` | 0.8769 | 0.0542 | 1683 | 0 |
| 0.25 | `gs-ndn-robust` | **0.9144** | **0.0362** | **697** | 14.6 |
| 0.5 | `gs-ndn-unverified-import` | 0.7432 | 0.1540 | 1429 | 0 |
| 0.5 | `gs-ndn` | 0.8536 | 0.0690 | 1320 | 0 |
| 0.5 | `gs-ndn-robust` | **0.9002** | **0.0403** | **645** | 29.0 |

Share of the attack's cost in satisfaction that the robust arm gives back:
87% at 12.5% compromise, 83% at 25%, 80% at 50%. Realised error falls by
71%, 69% and 74% respectively.

**City, same configuration.** The effect replicates and is slightly larger:

| compromised | arm | ISR | realised error | poison live | peers distrusted |
|---|---|---:|---:|---:|---:|
| 0.0 | `gs-ndn-unverified-import` | 0.9566 | 0.0120 | 0 | 0 |
| 0.0 | `gs-ndn` | 0.9622 | 0.0074 | 0 | 0 |
| 0.0 | `gs-ndn-robust` | 0.9622 | 0.0074 | 0 | 0.2 |
| 0.125 | `gs-ndn-unverified-import` | 0.8388 | 0.0766 | 1819 | 0 |
| 0.125 | `gs-ndn` | 0.9132 | 0.0391 | 1593 | 0 |
| 0.125 | `gs-ndn-robust` | **0.9477** | **0.0176** | **505** | 7.4 |
| 0.25 | `gs-ndn-unverified-import` | 0.7940 | 0.1101 | 1893 | 0 |
| 0.25 | `gs-ndn` | 0.8951 | 0.0471 | 1694 | 0 |
| 0.25 | `gs-ndn-robust` | **0.9357** | **0.0303** | **719** | 14.2 |
| 0.5 | `gs-ndn-unverified-import` | 0.7458 | 0.1584 | 1428 | 0 |
| 0.5 | `gs-ndn` | 0.8727 | 0.0606 | 1317 | 0 |
| 0.5 | `gs-ndn-robust` | **0.9188** | **0.0381** | **636** | 29.0 |

Recovery 92% / 87% / 82% of the attack's satisfaction cost; error down 77% /
72% / 76%.

**One result here is an argument, not a measurement, and needs deciding.** On
city the import-verification fix *improves the honest network*: 0.9566 → 0.9622
satisfaction and 0.0120 → 0.0074 realised error with no attacker present at all.
That is not a defence paying off, it is a correctness defect being repaired —
imported mappings that were simply wrong, for ordinary reasons, were also never
being retracted. It is evidence for making `verify_imported` the default rather
than an opt-in arm, at the cost of re-running the campaign.

Three things to read off it, in order of how much they matter:

1. **The defence is free on an honest network.** 0.9405 → 0.9404 at zero
   compromise. A defence that costs satisfaction when nobody is attacking cannot
   be left switched on, and every other result would need a caveat saying which
   arm produced it.
2. **It recovers most of the attack.** At 25% compromise the attack costs
   `gs-ndn` 0.154 ISR; the robust arm gives back 0.128 of that — 83% — and cuts
   realised error by 69%.
3. **It works for the stated reason.** `peers distrusted` is non-zero and grows
   with the compromised share. A defence that improved the metric while never
   distrusting anybody would not have been shown to work, only to correlate.

Roughly half the recovery is the §0.2 fix and half is reputation on top: both
rows are reported so the two are not conflated.

**Which sub-mechanism does the work.** At 25% compromise the robust arm blocked
7,916 mappings from peers whose trust had collapsed and only 24 on the
refuted-pair test. Peer distrust is carrying the result; pair memory is a
rounding error on top of it. Evidence-channel blocks are zero here for a
structural reason worth stating rather than hiding — `gs-ndn` has no risk
controller, so the evidence channel is never exercised at all. The peer-share cap
can only matter on the `rc-ndn-robust` arm, and its value has to be read there.

**What it does not do.** It does not authenticate anyone, so it is orthogonal to
signing rather than a substitute — an attacker who can forge identities gets a
fresh reputation each time, which is exactly why §9's threat model excludes that
capability. It does not address a compromised *producer*, still unmeasured.
Provenance — the other named defence — is still not implemented.

---

## 2. The horizon decay is an artifact of a closed vocabulary

**The scope condition under attack.** §2 reports that gossip's saving falls from
26% to 7.5% as the run goes 60 s → 600 s, and treats it as a property of the
protocol amortising. It is a property of the *experiment*: all 300 rewordings
are askable from t = 0, so once every router has met all of them there is nothing
left to learn and sharing must stop paying. The decay measures the catalog
running out.

A deployment's vocabulary does not run out. `WorkloadConfig.vocabulary_arrival_s`
introduces wordings over the run instead, from an **independent RNG stream** so
that switching it on changes *which* wording a request uses and nothing else —
same request count, same arrival times, same consumers, same services. Without
that separation an arrival sweep compares traces of different lengths and has two
variables. (Guarded by
`test_arrivals_change_which_wording_is_asked_and_nothing_else`.)

**Pre-registered prediction.** The saving decays toward zero only when arrivals
are off, and settles at a positive floor when they are on. *Falsified if it
decays to the same place regardless* — in which case §2's scope condition stands
as written and this section is deleted.

**Status: partially falsified already, and the first hypothesis was wrong.**

An earlier attempt used producer churn as the source of continued learning. It
does not work, and the reason is worth keeping: churn invalidates mappings at
*every router at once*, because a route withdrawal is a routing-plane event
everyone sees. It never creates the asymmetry gossip exploits. Measured, 16
edges, 3 seeds: frozen catalog 23.5% → 6.7% across 60 s → 600 s; with
drift+relocate churn every 5 s, 17.9% → 9.6%. The floor lifts slightly; the decay
does not go away.

**And the vocabulary hypothesis is failing too.** Hospital, 3 seeds, 16 edges,
encoder inferences saved by gossip against `saf+es`:

| arrival | 60 s | 240 s | 600 s |
|---|---:|---:|---:|
| closed | 23.5% | 14.3% | 6.7% |
| 0.5 s | 13.3% | 11.0% | 5.6% |
| 2.0 s | 6.3% | 5.6% | 4.2% |
| 5.0 s | 2.1% | 1.8% | 1.7% |

A slower arrival rate does not hold the saving up. It lowers it at **every**
horizon, including the 60 s one where §2's headline lives, and the ordering is
monotone in the arrival interval across the whole grid. The prediction was that
arrivals would sustain a floor; what actually happens is the exact opposite.

The mechanism, in hindsight, is clear and is worth stating in the paper because
it sharpens §2 rather than weakening it. Gossip's saving comes from the
**backlog** of wordings that some router has resolved and others have not. A
closed catalog at t = 0 is the largest backlog obtainable — every wording is new
to everybody at once. Introducing wordings slowly means the network is never far
from equilibrium: each new wording is disseminated long before the next arrives,
so there is nothing queued up to share. The instantaneous arrival rate relative
to gossip's dissemination speed is what sets the saving, and a closed catalog
maximises it.

**Conclusion: §2's scope condition stands as originally written, and the decay
really is a warm-up effect.** The value of this experiment is that it closes the
most obvious reviewer objection — "your advantage is an artifact of a static
catalog" — with a measurement instead of an argument. That is worth a short
subsection and the `vocabulary_arrival_s` knob staying in the codebase. It is not
worth a reframing of the contribution, and the reframing is withdrawn.

---

## 3. Running it

```bash
python -m pytest tests/ -q
python experiments/run_experiments.py --all --seeds 20 --jobs 0
python experiments/make_figures.py
```

`--jobs N` runs N seeded runs in parallel; `--jobs 0` uses one worker per core.
Output is reassembled in submission order and every run is a pure function of
its own seeded config, so results are identical to `--jobs 1` -- verified at
both the `Bench` level and the CLI level before the flag was documented.
Measured speedup on four workers: 3.05x.

`make_figures.py` now draws `fig_robust` and `fig_vocabulary` alongside the
existing eight.
