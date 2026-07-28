# Semantic Routing in NDN Networks — Simulation

## Project Overview
This simulation compares three NDN forwarding strategies for IoT service
discovery in a **Smart Hospital** scenario:

| Strategy | Description |
|:---|:---|
| **Vanilla NDN** | Standard exact-string-match FIB lookup. Fails on semantic name mismatches. |
| **Pure Semantic** | Uses MiniLM-L6 embeddings + cosine similarity on every FIB miss. High accuracy, but high latency. |
| **Semantic NDN + Embedding Store (Ours)** | Combines semantic matching with an **Embedding Store cache** that learns resolved mappings. Converges toward exact-match speed after warm-up. |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the simulation
python main.py
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~80 MB).
> Subsequent runs use the cached model.

## Output
- **Console:** Detailed metrics table (PDR, IRT, match breakdown).
- **Plot:** `ndn_semantic_routing_results.png` — 4-panel comparison figure.

## File Structure
```
simulation/
├── config.py        # All tunable parameters & name definitions
├── simulation.py    # Core engine: embeddings, strategies, workload
├── main.py          # Entry point: runs experiments & generates plots
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Tuning Parameters
Edit `config.py` to adjust:
- `NUM_REQUESTS` — Number of Interest packets (default: 1000)
- `SEMANTIC_VARIATION_RATE` — Fraction of requests using synonym names (default: 0.40)
- `SIMILARITY_THRESHOLD` — Cosine similarity cutoff (default: 0.70)
- Network delay values (link latency, embedding inference time, etc.)

## Authors
Mohammad Mahdi Yari & Sajjad Taghizadeh
Supervisor: Prof. Mohammadreza Shakournia
