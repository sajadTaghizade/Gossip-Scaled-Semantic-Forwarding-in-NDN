#!/usr/bin/env python3
"""Encode every catalog name once, offline, and write a reusable bundle.

Writes ``data/embeddings/<domain>.<model>.npz`` plus a ``.json`` sidecar holding
the *measured* encoder timings, which the simulator then uses instead of
invented latency constants.

    python experiments/export_embeddings.py --all                     # MiniLM-L6
    python experiments/export_embeddings.py --all --backend lexical   # the control

Separating encoding from simulation mirrors the deployment, where FIB-entry
embeddings are computed when routes are installed rather than per Interest, and
it means every reported number reproduces without a model download or a GPU.
Requires the ONNX graph from ``fetch_model.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsndn import datasets  # noqa: E402
from gsndn.datasets.schema import name_to_text  # noqa: E402
from gsndn.embeddings import (  # noqa: E402
    EmbeddingBackend,
    EncoderCost,
    LexicalBackend,
    OnnxMiniLMBackend,
    l2_normalize,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "embeddings"
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
TIMING_REPEATS = 30


def make_backend(kind: str) -> EmbeddingBackend:
    if kind == "onnx":
        return OnnxMiniLMBackend()
    if kind == "lexical":
        return LexicalBackend()
    raise ValueError(f"unknown backend {kind!r}")


def measure_cost(backend: EmbeddingBackend, samples: list[str]) -> EncoderCost:
    """Time single-name and batched encoding, reporting the median per batch size.

    The median rather than the mean, because a stray scheduler hiccup on a
    single-threaded Python process otherwise dominates a per-Interest cost that
    the simulator will apply hundreds of thousands of times.

    These are the numbers the simulator charges a router for running the
    encoder, so they must come from this machine rather than from a constant
    somebody once wrote down.
    """
    cost = EncoderCost(encode_ms=0.0, model_load_ms=backend.cost.model_load_ms, measured=True)

    singles: list[float] = []
    for i in range(TIMING_REPEATS):
        started = time.perf_counter()
        backend.encode([samples[i % len(samples)]])
        singles.append((time.perf_counter() - started) * 1000.0)
    cost.encode_ms = float(np.median(singles))
    cost.batch_ms[1] = cost.encode_ms

    for batch in BATCH_SIZES[1:]:
        if batch > len(samples):
            break
        runs: list[float] = []
        for _ in range(max(3, TIMING_REPEATS // batch)):
            started = time.perf_counter()
            backend.encode(samples[:batch])
            runs.append((time.perf_counter() - started) * 1000.0)
        cost.batch_ms[batch] = float(np.median(runs))

    try:
        import resource

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cost.resident_mb = peak_kb / 1024.0
    except Exception:  # pragma: no cover - platform dependent
        pass
    return cost


def export(domain: str, kind: str, out_dir: Path) -> Path:
    catalog = datasets.load(domain)
    names = sorted(set(catalog.all_names))

    backend = make_backend(kind)
    print(
        f"[{domain}] {len(names)} distinct names, backend={backend.id}, "
        f"dim={backend.dim}, load={backend.cost.model_load_ms:.0f} ms"
    )

    vectors = np.zeros((len(names), backend.dim), dtype=np.float32)
    for start in range(0, len(names), 64):
        chunk = names[start:start + 64]
        vectors[start:start + len(chunk)] = backend.encode(chunk)
    vectors = l2_normalize(vectors)

    texts = [name_to_text(n) for n in names]
    cost = measure_cost(backend, texts[:64])
    print(
        f"[{domain}] encode cost: {cost.encode_ms:.2f} ms/name single, "
        f"{cost.per_name_ms(32):.3f} ms/name at batch 32"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = backend.id.replace("/", "_")
    npz_path = out_dir / f"{domain}.{slug}.npz"
    np.savez_compressed(
        npz_path,
        names=np.array(names, dtype=object).astype("U"),
        vectors=vectors,
        model_id=np.array(backend.id),
    )
    npz_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "domain": domain,
                "backend": kind,
                "model": backend.id,
                "dim": backend.dim,
                "n_names": len(names),
                "bytes_per_entry": backend.bytes_per_entry,
                "catalog_summary": catalog.summary(),
                "cost": cost.to_dict(),
            },
            indent=2,
        )
    )
    size_mb = npz_path.stat().st_size / 1e6
    print(f"[{domain}] wrote {npz_path.name} ({size_mb:.2f} MB) + sidecar\n")
    return npz_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=datasets.DOMAINS, help="single domain to export")
    parser.add_argument("--all", action="store_true", help="export every domain")
    parser.add_argument(
        "--backend",
        default="onnx",
        choices=("onnx", "lexical"),
        help="encoder to export with (default: onnx MiniLM-L6)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()

    if not args.all and not args.domain:
        parser.error("pass --domain <name> or --all")

    targets = datasets.DOMAINS if args.all else (args.domain,)
    for domain in targets:
        export(domain, args.backend, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
