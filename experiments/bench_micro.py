#!/usr/bin/env python3
"""Measure what each forwarding operation actually costs on this machine.

    python experiments/bench_micro.py --out data/costs.json

Writes a cost model the simulator loads instead of assuming latencies.  The
figures that matter most are the encoder inference time and how similarity
search scales with FIB size, because those two decide whether a semantic
forwarding plane is affordable at all -- and the second is the one a constant
cannot express, since it is a matrix-vector product that grows with the number
of routes.

Everything is timed single-threaded.  A router forwards on one core; a benchmark
that lets ONNX Runtime spread across every core would report an encoder cost the
deployment cannot reproduce.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsndn import datasets  # noqa: E402
from gsndn.costs import CostModel  # noqa: E402
from gsndn.embeddings import (  # noqa: E402
    LexicalBackend,
    NameIndex,
    OnnxMiniLMBackend,
    PrecomputedBackend,
)
from gsndn.simhash import SimHasher, hamming_array  # noqa: E402

FIB_SIZES = (10, 25, 50, 100, 250, 500, 1000, 2500)
REPEATS = 200
WARMUP = 20


def timed(fn: Callable[[], object], repeats: int = REPEATS) -> Tuple[float, float]:
    """Median and 95th-percentile duration in milliseconds.

    The median because a single-threaded Python process picks up scheduler noise
    that a mean would let dominate; the 95th because tail cost is what fills a
    router's queue.
    """
    for _ in range(WARMUP):
        fn()
    samples: List[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(samples)), float(np.percentile(samples, 95))


def bench_encoder(names: Sequence[str]) -> Dict[str, float]:
    print("[*] encoder inference (single-threaded ONNX MiniLM-L6)")
    backend = OnnxMiniLMBackend(threads=1)
    cursor = {"i": 0}

    def one() -> None:
        cursor["i"] = (cursor["i"] + 1) % len(names)
        backend.encode([names[cursor["i"]]])

    median, p95 = timed(one, repeats=100)
    print(f"    encode 1 name : {median:7.3f} ms  (p95 {p95:.3f})")

    lexical = LexicalBackend()
    lex_median, _ = timed(lambda: lexical.encode([names[0]]), repeats=200)
    print(f"    lexical control: {lex_median:7.3f} ms  ({median / lex_median:.0f}x cheaper)")
    return {
        "encode_ms": median,
        "encode_p95_ms": p95,
        "encode_lexical_ms": lex_median,
        "model_load_ms": backend.cost.model_load_ms,
    }


def bench_search(index: NameIndex) -> Dict[str, object]:
    """Cosine search against FIBs of growing size.

    Fitted as ``base + per_entry * n`` so the simulator can charge a router for
    the FIB it actually has rather than the one this benchmark happened to use.

    Each size gets its own freshly generated table.  Tiling one small table
    would leave the whole thing resident in cache at every size and measure the
    cache rather than the search.
    """
    print("[*] cosine similarity search vs FIB size")
    rng = np.random.default_rng(0)
    dim = index.backend.dim
    query = index.vectors[0]
    sizes: List[int] = []
    medians: List[float] = []

    for size in FIB_SIZES:
        table = rng.standard_normal((size, dim)).astype(np.float32)
        table /= np.linalg.norm(table, axis=1, keepdims=True)
        median, _ = timed(lambda t=table: t @ query, repeats=400)
        sizes.append(size)
        medians.append(median)
        print(f"    n={size:>5}: {median:8.4f} ms")

    slope, intercept = np.polyfit(sizes, medians, 1)
    print(f"    fit: {intercept * 1000:.2f} us + {slope * 1000:.4f} us/entry")
    print(f"    at n=50 the search is {medians[FIB_SIZES.index(50)]:.4f} ms -- "
          f"negligible beside the encoder, which is the point")
    return {
        "search_base_ms": max(0.0, float(intercept)),
        "search_per_entry_ms": max(0.0, float(slope)),
        "search_samples": {str(s): m for s, m in zip(sizes, medians)},
    }


def bench_tables(names: Sequence[str]) -> Dict[str, float]:
    print("[*] table operations")
    table = {name: i for i, name in enumerate(names)}
    probe = names[len(names) // 2]
    es_median, _ = timed(lambda: table.get(probe), repeats=2000)
    print(f"    ES hash probe : {es_median:8.5f} ms")

    prefix = "/" + "/".join(probe.strip("/").split("/")[:3])
    tag_median, _ = timed(lambda: table.get(prefix), repeats=2000)
    print(f"    tag FIB match : {tag_median:8.5f} ms")
    return {"es_lookup_ms": es_median, "tag_match_ms": tag_median}


def bench_simhash(index: NameIndex, queries: np.ndarray) -> Dict[str, object]:
    """Signature cost, and how well signatures actually rank routes.

    The accuracy is measured with *variant* names queried against the FIB, which
    is the job a forwarding plane would be asking of them.  Scoring signatures
    against the same vectors they were built from would return each entry's own
    row and report a perfect result that means nothing.

    Reported next to the timing on purpose.  Signature comparison is genuinely
    fast, and that speed would read as a reason to route on it if the accuracy
    were not in the same table.
    """
    print("[*] signatures: cost, and route-selection accuracy on variant queries")
    gold = (queries @ index.vectors.T).argmax(1)

    accuracy: Dict[str, Dict[str, float]] = {}
    for bits in (64, 128, 256, 512, 1024):
        hasher = SimHasher(dim=index.backend.dim, bits=bits)
        table = hasher.signatures(index.vectors)
        query_sigs = hasher.signatures(queries)

        sig_ms, _ = timed(lambda h=hasher: h.signature(index.vectors[0]), repeats=200)
        cmp_ms, _ = timed(
            lambda t=table, q=query_sigs[0]: hamming_array(q, t), repeats=500
        )

        order = np.stack([hamming_array(query_sigs[i], table) for i in range(len(queries))]).argsort(1)
        recall = {
            f"recall_at_{k}": float(np.mean([gold[i] in order[i, :k] for i in range(len(queries))]))
            for k in (1, 3, 8)
        }
        accuracy[str(bits)] = {
            "signature_ms": sig_ms, "compare_ms": cmp_ms,
            "bytes": bits // 8, **recall,
        }
        print(
            f"    {bits:>5} bits ({bits // 8:>3}B): sign {sig_ms:.4f} ms, "
            f"R@1 {recall['recall_at_1']:.3f}  R@3 {recall['recall_at_3']:.3f}  "
            f"R@8 {recall['recall_at_8']:.3f}"
        )
    print("    routing on nearest-signature alone would misroute most variants;")
    print("    signatures are kept for gossip digests, not for route selection")

    return {"simhash": accuracy, "signature_ms": accuracy["256"]["signature_ms"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="hospital", choices=datasets.DOMAINS)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "costs.json",
    )
    parser.add_argument(
        "--bundle", type=Path, default=None, help="embedding bundle to search over"
    )
    parser.add_argument("--skip-encoder", action="store_true", help="skip the ONNX benchmark")
    args = parser.parse_args()

    catalog = datasets.load(args.domain)
    bundle = args.bundle or (
        Path(__file__).resolve().parents[1]
        / "data" / "embeddings" / f"{args.domain}.all-MiniLM-L6-v2-onnx.npz"
    )
    backend = PrecomputedBackend(bundle)
    index = NameIndex(backend, catalog.canonical_names)
    names = [i.name for i in catalog.interests]

    print(f"machine: {platform.processor() or platform.machine()}, "
          f"python {platform.python_version()}\n")

    measurements: Dict[str, object] = {}
    if not args.skip_encoder:
        measurements.update(bench_encoder(names))
    measurements.update(bench_search(index))
    measurements.update(bench_tables(catalog.canonical_names))

    from gsndn.datasets import VARIANT

    variant_names = [i.name for i in catalog.by_kind(VARIANT)]
    measurements.update(bench_simhash(index, backend.encode(variant_names)))

    model = CostModel()
    for field_name in ("encode_ms", "search_base_ms", "search_per_entry_ms",
                       "es_lookup_ms", "tag_match_ms", "signature_ms"):
        if field_name in measurements:
            setattr(model, field_name, float(measurements[field_name]))  # type: ignore[arg-type]
    model.measured = {
        "encode_ms": not args.skip_encoder,
        "search_base_ms": True,
        "search_per_entry_ms": True,
        "es_lookup_ms": True,
        "tag_match_ms": True,
        "signature_ms": True,
        "qtable_update_ms": False,
        "gossip_apply_ms_per_entry": False,
    }
    model.machine = f"{platform.processor() or platform.machine()} / python {platform.python_version()}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    args.out.with_name(args.out.stem + ".detail.json").write_text(
        json.dumps(measurements, indent=2, default=float)
    )
    print(f"\n[+] cost model -> {args.out}")
    print(f"[+] full measurements -> {args.out.with_name(args.out.stem + '.detail.json')}")

    print(f"\n    slow path at FIB=50 : {model.semantic_resolve_ms(50):.3f} ms")
    print(f"    slow path at FIB=1000: {model.semantic_resolve_ms(1000):.3f} ms")
    print(f"    ES hit               : {model.es_lookup_ms:.5f} ms "
          f"({model.semantic_resolve_ms(50) / model.es_lookup_ms:.0f}x cheaper)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
