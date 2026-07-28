#!/usr/bin/env python3
"""Fetch the all-MiniLM-L6-v2 ONNX graph into the local model cache.

    python experiments/fetch_model.py

Downloads roughly 80 MB into ``~/.cache/gsndn/all-MiniLM-L6-v2-onnx`` (override
with ``GSNDN_MODEL_DIR``).  The graph is deliberately *not* committed: the
repository carries the exported vectors, which are two orders of magnitude
smaller and are what the simulator actually reads.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsndn.embeddings import OnnxMiniLMBackend, default_model_dir  # noqa: E402

REQUIRED = ("model.onnx", "tokenizer.json")


def fetch(target: Path, force: bool = False) -> Path:
    if not force and all((target / f).exists() for f in REQUIRED):
        print(f"[=] model already present in {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    url = OnnxMiniLMBackend.MODEL_URL
    print(f"[*] downloading {url}")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "onnx.tar.gz"
        with urllib.request.urlopen(url, timeout=300) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        size_mb = archive.stat().st_size / 1e6
        print(f"[*] downloaded {size_mb:.1f} MB, extracting")

        with tarfile.open(archive) as tar:
            # Flatten whatever directory layout the archive happens to use, and
            # refuse absolute or parent-relative members rather than trusting
            # the tarball's own paths.
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name or name.startswith("."):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                (target / name).write_bytes(extracted.read())

    missing = [f for f in REQUIRED if not (target / f).exists()]
    if missing:
        raise SystemExit(f"[!] archive did not contain {missing}")
    print(f"[+] model ready in {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=None, help="destination directory")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    target = args.dir or default_model_dir()
    fetch(target, force=args.force)

    print("[*] verifying the graph loads and produces sane similarities")
    backend = OnnxMiniLMBackend(model_dir=target)
    vectors = backend.encode(
        [
            "/smart-hospital/building-a/floor-1/temperature/room-101",
            "/hospital/temp-sensor/room-101",
            "/video-streaming/movies/catalog",
        ]
    )
    print(f"    variant similarity   : {float(vectors[0] @ vectors[1]):+.4f}  (expect ~0.80)")
    print(f"    unrelated similarity : {float(vectors[0] @ vectors[2]):+.4f}  (expect ~0.00)")
    print(f"    single encode        : {backend.cost.encode_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
