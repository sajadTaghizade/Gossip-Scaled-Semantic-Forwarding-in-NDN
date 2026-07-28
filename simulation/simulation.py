"""
═══════════════════════════════════════════════════════════════════════
  Core Simulation Engine for Semantic NDN Routing
  ───────────────────────────────────────────────
  Implements:
    • EmbeddingEngine  — MiniLM-L6 wrapper for semantic name matching
    • EmbeddingStore   — Cache mapping variant names → canonical names
    • NDNSimulator     — Event-driven simulation of three strategies
    • generate_workload — Zipf-distributed Smart Hospital request trace
═══════════════════════════════════════════════════════════════════════
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config


# ═══════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RequestResult:
    """Result of processing a single Interest packet."""
    request_name: str        # The name the consumer requested
    resolved: bool           # True if the Interest was satisfied
    latency_ms: float        # End-to-end latency (0.0 if NACK)
    match_type: str          # "exact" | "semantic" | "es_cache" | "nack"
    matched_name: str = ""   # The canonical name that was matched (if any)
    similarity: float = 0.0  # Cosine similarity score (if semantic match)


# ═══════════════════════════════════════════════════════════════════
#  EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Wraps the all-MiniLM-L6-v2 Sentence Transformer for semantic
    name matching in NDN. Converts hierarchical NDN names into
    384-dimensional dense vector embeddings and performs cosine
    similarity search.
    """

    def __init__(self):
        print(f"[*] Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self._embeddings: Dict[str, np.ndarray] = {}
        print("[✓] Model loaded successfully.\n")

    @staticmethod
    def _name_to_text(ndn_name: str) -> str:
        """
        Convert an NDN hierarchical name to readable text for the
        embedding model. Slashes and hyphens become spaces.
        Example: '/smart-hospital/temperature/room-101'
              → 'smart hospital temperature room 101'
        """
        return ndn_name.replace("/", " ").replace("-", " ").strip()

    def precompute_embeddings(self, names: List[str]) -> None:
        """Batch-encode a list of NDN names and cache their embeddings."""
        print(f"[*] Pre-computing embeddings for {len(names)} names...")
        readable = [self._name_to_text(n) for n in names]
        embeddings = self.model.encode(readable, show_progress_bar=False,
                                       convert_to_numpy=True)
        for name, emb in zip(names, embeddings):
            self._embeddings[name] = emb
        print("[✓] All embeddings cached.\n")

    def get_embedding(self, name: str) -> np.ndarray:
        """Retrieve (or compute on-the-fly) the embedding for a name."""
        if name not in self._embeddings:
            readable = self._name_to_text(name)
            self._embeddings[name] = self.model.encode(
                [readable], show_progress_bar=False, convert_to_numpy=True
            )[0]
        return self._embeddings[name]

    def find_best_match(
        self,
        query_name: str,
        candidate_names: List[str],
        threshold: float,
    ) -> Optional[Tuple[str, float]]:
        """
        Search for the best semantic match among candidates.

        Returns:
            (matched_name, similarity_score) if best score ≥ threshold,
            None otherwise.
        """
        query_emb = self.get_embedding(query_name).reshape(1, -1)
        candidate_embs = np.array(
            [self._embeddings[n] for n in candidate_names]
        )

        sims = cosine_similarity(query_emb, candidate_embs)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            return candidate_names[best_idx], best_sim
        return None

    def verify_semantic_pairs(self) -> bool:
        """
        Verify all configured variant → canonical pairs meet the
        similarity threshold. Prints a diagnostic report.
        """
        print("[*] Verifying semantic variant pairs...")
        all_pass = True
        failing = []

        for variant, canonical in config.SEMANTIC_VARIANTS.items():
            emb_v = self.get_embedding(variant).reshape(1, -1)
            emb_c = self.get_embedding(canonical).reshape(1, -1)
            sim = float(cosine_similarity(emb_v, emb_c)[0, 0])

            if sim < config.SIMILARITY_THRESHOLD:
                all_pass = False
                failing.append((variant, canonical, sim))

        if all_pass:
            print(f"[✓] All {len(config.SEMANTIC_VARIANTS)} pairs exceed "
                  f"threshold {config.SIMILARITY_THRESHOLD:.2f}\n")
        else:
            print(f"[!] {len(failing)} pair(s) below threshold "
                  f"{config.SIMILARITY_THRESHOLD:.2f}:")
            for var, can, sim in failing:
                print(f"    ✗ {sim:.4f}  {var}")
                print(f"             → {can}")
            print("    Consider lowering SIMILARITY_THRESHOLD in config.py\n")

        return all_pass


# ═══════════════════════════════════════════════════════════════════
#  EMBEDDING STORE (CACHE)
# ═══════════════════════════════════════════════════════════════════

class EmbeddingStore:
    """
    The Embedding Store (ES) caches resolved semantic mappings:
        variant_name → canonical_name

    This is the core optimization of our approach. After the first
    semantic lookup resolves a variant, all subsequent requests for
    the same variant name are served from this O(1) cache instead
    of re-running the expensive embedding + similarity search.
    """

    def __init__(self):
        self.cache: Dict[str, str] = {}
        self.hits: int = 0
        self.misses: int = 0
        self._hit_ratio_history: List[float] = []

    def lookup(self, name: str) -> Optional[str]:
        """
        Look up a variant name in the cache.
        Returns the canonical name if found, None otherwise.
        Also tracks cumulative hit/miss statistics.
        """
        result = self.cache.get(name)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1

        # Record cumulative hit ratio after every query
        total = self.hits + self.misses
        self._hit_ratio_history.append(self.hits / total if total > 0 else 0.0)
        return result

    def store(self, variant_name: str, canonical_name: str) -> None:
        """Store a resolved semantic mapping in the cache."""
        self.cache[variant_name] = canonical_name

    @property
    def hit_ratio(self) -> float:
        """Current cumulative hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def hit_ratio_history(self) -> List[float]:
        """Full history of cumulative hit ratio after each ES query."""
        return self._hit_ratio_history


# ═══════════════════════════════════════════════════════════════════
#  WORKLOAD GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_workload(rng: np.random.Generator) -> List[str]:
    """
    Generate a Zipf-distributed workload of Interest requests for
    the Smart Hospital scenario.

    The Zipf distribution models the realistic "hot content" pattern:
    a few sensors (e.g., temperature in ICU) are queried very
    frequently, while others are queried rarely.

    With probability SEMANTIC_VARIATION_RATE, a request uses a
    semantic variant (e.g., '/hospital/temp-sensor/room-101')
    instead of the canonical name.
    """
    n_canonical = len(config.PRODUCER_NAMES)

    # Generate Zipf-distributed indices (rejection sampling for bounds)
    indices: List[int] = []
    while len(indices) < config.NUM_REQUESTS:
        samples = rng.zipf(config.ZIPF_ALPHA, size=config.NUM_REQUESTS * 2)
        valid = samples[samples <= n_canonical] - 1  # 0-indexed
        indices.extend(valid.tolist())
    indices = indices[:config.NUM_REQUESTS]

    # Build reverse map: canonical → [variants]
    canonical_to_variants: Dict[str, List[str]] = {}
    for variant, canonical in config.SEMANTIC_VARIANTS.items():
        canonical_to_variants.setdefault(canonical, []).append(variant)

    # Generate the request trace
    workload: List[str] = []
    for idx in indices:
        canonical = config.PRODUCER_NAMES[idx]

        # With probability p, substitute with a semantic variant
        if (rng.random() < config.SEMANTIC_VARIATION_RATE
                and canonical in canonical_to_variants):
            variants = canonical_to_variants[canonical]
            variant = rng.choice(variants)
            workload.append(variant)
        else:
            workload.append(canonical)

    return workload


# ═══════════════════════════════════════════════════════════════════
#  UTILITY: JITTER
# ═══════════════════════════════════════════════════════════════════

def _add_jitter(base_ms: float, rng: np.random.Generator) -> float:
    """Add ±5% Gaussian noise to a delay value (minimum 0.01 ms)."""
    return max(0.01, rng.normal(base_ms, base_ms * config.JITTER_STD_FACTOR))


# ═══════════════════════════════════════════════════════════════════
#  NDN SIMULATOR
# ═══════════════════════════════════════════════════════════════════

class NDNSimulator:
    """
    Discrete-event simulator for NDN Semantic Routing.

    Simulates three forwarding strategies over the same workload
    and records per-packet metrics (latency, match type, PDR).

    Network topology modeled:

        Consumer(s) ──[2ms]──▶ Edge Router ──[5ms]──▶ Producer(s)

    The Edge Router is where forwarding decisions (exact match,
    semantic lookup, or ES cache) happen.
    """

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.engine = embedding_engine
        self.fib = set(config.PRODUCER_NAMES)  # Exact-match FIB entries

    def _is_exact_match(self, name: str) -> bool:
        """Check if the name has an exact entry in the FIB."""
        return name in self.fib

    # ───────────────────────────────────────────────────────────────
    #  Strategy 1: Vanilla NDN (Exact String Matching Only)
    # ───────────────────────────────────────────────────────────────
    def run_vanilla_ndn(self, workload: List[str]) -> List[RequestResult]:
        """
        Standard NDN: only exact string matching in the FIB.
        Any semantic variant results in a NACK (packet drop).
        """
        print("[▶] Running Strategy 1: Vanilla NDN ...")
        rng = np.random.default_rng(config.RANDOM_SEED + 1)
        results: List[RequestResult] = []

        for name in workload:
            if self._is_exact_match(name):
                latency = _add_jitter(config.EXACT_MATCH_DELAY, rng)
                results.append(RequestResult(
                    name, True, latency, "exact", name, 1.0
                ))
            else:
                # No exact match → NACK (Interest unsatisfied)
                results.append(RequestResult(
                    name, False, 0.0, "nack"
                ))

        resolved = sum(1 for r in results if r.resolved)
        print(f"    Resolved: {resolved}/{len(results)} "
              f"({resolved/len(results):.1%})")
        return results

    # ───────────────────────────────────────────────────────────────
    #  Strategy 2: Pure Semantic NDN (No Caching)
    # ───────────────────────────────────────────────────────────────
    def run_pure_semantic(self, workload: List[str]) -> List[RequestResult]:
        """
        Semantic NDN without caching: on every FIB miss, the router
        runs full MiniLM embedding + cosine similarity search.
        High PDR, but consistently high latency for every miss.
        """
        print("[▶] Running Strategy 2: Pure Semantic NDN ...")
        rng = np.random.default_rng(config.RANDOM_SEED + 2)
        results: List[RequestResult] = []

        for name in workload:
            if self._is_exact_match(name):
                latency = _add_jitter(config.EXACT_MATCH_DELAY, rng)
                results.append(RequestResult(
                    name, True, latency, "exact", name, 1.0
                ))
            else:
                # Full semantic search every time (no cache)
                match = self.engine.find_best_match(
                    name, config.PRODUCER_NAMES, config.SIMILARITY_THRESHOLD
                )
                if match:
                    matched_name, sim = match
                    latency = _add_jitter(config.SEMANTIC_MATCH_DELAY, rng)
                    results.append(RequestResult(
                        name, True, latency, "semantic", matched_name, sim
                    ))
                else:
                    results.append(RequestResult(
                        name, False, 0.0, "nack"
                    ))

        resolved = sum(1 for r in results if r.resolved)
        print(f"    Resolved: {resolved}/{len(results)} "
              f"({resolved/len(results):.1%})")
        return results

    # ───────────────────────────────────────────────────────────────
    #  Strategy 3: Semantic NDN + Embedding Store (Our Approach)
    # ───────────────────────────────────────────────────────────────
    def run_semantic_with_es(
        self, workload: List[str]
    ) -> Tuple[List[RequestResult], EmbeddingStore]:
        """
        Our proposed approach: Semantic matching with an Embedding
        Store (ES) cache. The flow is:

            1. Check FIB for exact match        → O(1), ~0.01ms
            2. Check ES cache for variant name   → O(1), ~0.10ms
            3. Full semantic search (last resort) → O(n), ~8.50ms
               └─ On success, store mapping in ES for future hits.

        Over time, the ES cache "warms up" and most variant requests
        are resolved at Step 2, converging toward exact-match speed.
        """
        print("[▶] Running Strategy 3: Semantic NDN + Embedding Store ...")
        rng = np.random.default_rng(config.RANDOM_SEED + 3)
        es = EmbeddingStore()
        results: List[RequestResult] = []

        for name in workload:
            # Step 1: Exact FIB match (fast path)
            if self._is_exact_match(name):
                latency = _add_jitter(config.EXACT_MATCH_DELAY, rng)
                results.append(RequestResult(
                    name, True, latency, "exact", name, 1.0
                ))
                continue

            # Step 2: Embedding Store cache lookup
            cached = es.lookup(name)
            if cached is not None:
                latency = _add_jitter(config.ES_CACHE_MATCH_DELAY, rng)
                results.append(RequestResult(
                    name, True, latency, "es_cache", cached, 1.0
                ))
                continue

            # Step 3: Full semantic search (slow path)
            match = self.engine.find_best_match(
                name, config.PRODUCER_NAMES, config.SIMILARITY_THRESHOLD
            )
            if match:
                matched_name, sim = match
                es.store(name, matched_name)  # Cache for future lookups
                latency = _add_jitter(config.SEMANTIC_MATCH_DELAY, rng)
                results.append(RequestResult(
                    name, True, latency, "semantic", matched_name, sim
                ))
            else:
                results.append(RequestResult(
                    name, False, 0.0, "nack"
                ))

        resolved = sum(1 for r in results if r.resolved)
        print(f"    Resolved: {resolved}/{len(results)} "
              f"({resolved/len(results):.1%})")
        print(f"    ES Cache — Hits: {es.hits}, Misses: {es.misses}, "
              f"Final Hit Ratio: {es.hit_ratio:.1%}")
        print(f"    ES Cache — Unique entries learned: {len(es.cache)}")
        return results, es
