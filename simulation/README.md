# Phase 2 prototype — superseded by `../gsndn`

The original three-strategy comparison, kept for reference. It established the
scenario and the Embedding Store idea that the current work builds on, and its
name and synonym lists were the starting point for `../gsndn/datasets`.

It is not used for any reported result, for reasons worth recording.

**Latency was configuration, not outcome.** Every delay was a constant in
`config.py` plus 5% jitter — `EMBEDDING_COMPUTE_DELAY = 8.0` among them. There
was no clock, no arrival process and no queue, so the comparison between
strategies was decided by the numbers typed into the config file. The measured
value on the machine that ran the campaign turns out to be 7.05 ms, so the guess
was a good one; but a good guess and a measurement are different things, and
queueing under load — the effect SAF's efficiency analysis is entirely about —
could not appear at all.

**A wrong delivery counted as a success.** `run_pure_semantic` and
`run_semantic_with_es` never compared the matched name against the name the
request should have resolved to. A variant for room 101 that matched the sensor
in room 103 was recorded as `resolved=True` and counted toward the delivery
ratio. There were no precision, recall or false-positive metrics, and no
unsatisfiable requests to measure them against.

**Convergence was true by construction.** The workload drew only from the same
20 fixed variant names the Embedding Store would cache, and the cache was
unbounded with no eviction, so the hit ratio had to approach 100%. That measures
the workload, not the design. SAF specifies an LRU cache with FIB-driven
invalidation; both are implemented in `../gsndn/tables.py`.

**One router, one seed.** No PIT, no Content Store, no multi-hop path, and no
gossip — none of the distributed behaviour the Phase 1 and Phase 2 decks
describe. A single run of a stochastic simulation, with no repetitions and no
confidence intervals.

One smaller thing worth knowing if you run it: `EmbeddingEngine.find_best_match`
memoises query embeddings in `self._embeddings`, so the "no caching" strategy
does cache them. It changes nothing here, because the delays are constants
either way, but it would have as soon as timing became real.

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Needs `sentence-transformers` and a HuggingFace download. The current work uses
the same model through ONNX Runtime and reads precomputed vectors instead — see
`../README.md`.

## Authors

Mohammad Mahdi Yari & Sajjad Taghizadeh · Supervisor: Prof. Mohammadreza Shakournia
