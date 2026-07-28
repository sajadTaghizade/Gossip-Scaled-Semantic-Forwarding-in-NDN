"""Name encoders, and the cost of running them.

The simulator needs two things from an encoder: the vectors themselves, and how
long producing one takes.  Both are provided together here so that no timing
constant is invented -- the same object that produces a vector reports the
measured cost of having produced it.

Three backends:

``OnnxMiniLMBackend``
    ``all-MiniLM-L6-v2`` -- the model SAF settles on -- executed through ONNX
    Runtime on CPU.  Same weights, no PyTorch, no GPU, and a dependency
    footprint an edge router could plausibly carry.  This is the default.

``SentenceTransformerBackend``
    The same family of models through the original PyTorch stack, for
    cross-checking the ONNX path and for trying MiniLM-L12 or MPNet.

``PrecomputedBackend``
    Reads vectors from an ``.npz`` produced earlier by
    ``experiments/export_embeddings.py``.  This is the path that makes results
    reproducible on a machine with no model access at all, and it is how the
    simulator is normally run: encoding is a fixed, offline preprocessing step,
    exactly as SAF precomputes FIB-entry embeddings.

``LexicalBackend``
    A stateless character n-gram hashing encoder.  It is *not* a semantic model
    and is not presented as one -- it is the "do you actually need a transformer
    at the edge?" control.  It should resolve abbreviations and morphological
    variants (``temp`` against ``temperature``) while failing on true synonymy
    (``heartrate`` against ``cardiac-monitor``), and that gap is itself a
    result worth reporting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Sequence

import numpy as np

from .datasets.schema import name_to_text

DEFAULT_MODEL = "all-MiniLM-L6-v2"

#: Where the ONNX MiniLM graph and tokenizer are cached on disk.  Kept outside
#: the repository: the exported ``.npz`` vectors are the committed artefact, not
#: the 80 MB graph that produced them.
MODEL_CACHE_ENV = "GSNDN_MODEL_DIR"


def default_model_dir() -> Path:
    """Resolve the model cache directory, honouring ``GSNDN_MODEL_DIR``."""
    import os

    override = os.environ.get(MODEL_CACHE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "gsndn" / "all-MiniLM-L6-v2-onnx"

#: Per-entry memory of a 384-d float32 embedding, as reported by SAF (~1.5 KB).
BYTES_PER_FLOAT32 = 4


@dataclass
class EncoderCost:
    """Measured cost of encoding, in milliseconds per name.

    ``batch_ms`` maps a batch size to the wall-clock time to encode a batch of
    that size, so the simulator can model both the per-Interest path (batch of
    one) and bulk FIB precomputation.
    """

    encode_ms: float
    batch_ms: Dict[int, float] = field(default_factory=dict)
    model_load_ms: float = 0.0
    resident_mb: float = 0.0
    measured: bool = False

    def per_name_ms(self, batch: int = 1) -> float:
        if batch <= 1:
            return self.encode_ms
        known = sorted(self.batch_ms)
        if not known:
            return self.encode_ms
        nearest = min(known, key=lambda b: abs(b - batch))
        return self.batch_ms[nearest] / nearest

    def to_dict(self) -> Dict[str, object]:
        return {
            "encode_ms": self.encode_ms,
            "batch_ms": {str(k): v for k, v in self.batch_ms.items()},
            "model_load_ms": self.model_load_ms,
            "resident_mb": self.resident_mb,
            "measured": self.measured,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "EncoderCost":
        return cls(
            encode_ms=float(raw["encode_ms"]),  # type: ignore[arg-type]
            batch_ms={int(k): float(v) for k, v in (raw.get("batch_ms") or {}).items()},  # type: ignore[union-attr]
            model_load_ms=float(raw.get("model_load_ms", 0.0)),  # type: ignore[arg-type]
            resident_mb=float(raw.get("resident_mb", 0.0)),  # type: ignore[arg-type]
            measured=bool(raw.get("measured", False)),
        )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so that cosine similarity is a plain dot product."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class EmbeddingBackend:
    """Base class: turns NDN names into unit-length row vectors."""

    id: str = "abstract"
    dim: int = 0

    def __init__(self) -> None:
        self.cost = EncoderCost(encode_ms=0.0)

    def encode(self, names: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, name: str) -> np.ndarray:
        return self.encode([name])[0]

    @property
    def bytes_per_entry(self) -> int:
        return self.dim * BYTES_PER_FLOAT32

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} id={self.id!r} dim={self.dim}>"


class OnnxMiniLMBackend(EmbeddingBackend):
    """``all-MiniLM-L6-v2`` on CPU via ONNX Runtime.

    Mean-pools the token states under the attention mask and L2-normalises,
    which is what the Sentence-Transformers pooling head for this model does;
    the two paths agree to within float tolerance.

    Running the released ONNX export rather than PyTorch keeps the encoder
    honest as an *edge* component: no CUDA, no framework, a single 80 MB graph.
    """

    MODEL_URL = (
        "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
    )
    MAX_TOKENS = 128  # NDN names are short; capping keeps inference cost stable

    def __init__(self, model_dir: Optional[str | Path] = None, threads: int = 1) -> None:
        super().__init__()
        import onnxruntime as ort  # lazy: heavy import
        from tokenizers import Tokenizer

        directory = Path(model_dir) if model_dir else default_model_dir()
        graph = directory / "model.onnx"
        tokenizer_file = directory / "tokenizer.json"
        if not graph.exists() or not tokenizer_file.exists():
            raise FileNotFoundError(
                f"MiniLM ONNX assets missing from {directory}. "
                f"Run: python experiments/fetch_model.py"
            )

        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=self.MAX_TOKENS)

        options = ort.SessionOptions()
        # A router forwards on one core; letting ORT fan out across every core
        # would report an encoder cost the deployment could not reproduce.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads

        started = time.perf_counter()
        self._session = ort.InferenceSession(
            str(graph), options, providers=["CPUExecutionProvider"]
        )
        load_ms = (time.perf_counter() - started) * 1000.0

        self.id = "all-MiniLM-L6-v2-onnx"
        self.dim = 384
        self._input_names = {i.name for i in self._session.get_inputs()}
        self.cost = EncoderCost(encode_ms=0.0, model_load_ms=load_ms, measured=True)
        self.encode(["warm up"])

    def encode(self, names: Sequence[str]) -> np.ndarray:
        texts = [name_to_text(n) for n in names]
        started = time.perf_counter()
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        states = self._session.run(None, feed)[0]
        weights = mask[..., None].astype(np.float32)
        pooled = (states * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        vectors = l2_normalize(pooled)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        batch = max(1, len(texts))
        self.cost.batch_ms[batch] = elapsed_ms
        if batch == 1:
            self.cost.encode_ms = elapsed_ms
        return vectors


class SentenceTransformerBackend(EmbeddingBackend):
    """Live Sentence-Transformers model, as used by SAF."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        started = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        load_ms = (time.perf_counter() - started) * 1000.0

        self.id = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.cost = EncoderCost(encode_ms=0.0, model_load_ms=load_ms, measured=True)
        self._warm_up()

    def _warm_up(self) -> None:
        """First inference includes lazy allocation; exclude it from the cost."""
        self.model.encode(["warm up"], show_progress_bar=False, convert_to_numpy=True)

    def encode(self, names: Sequence[str]) -> np.ndarray:
        texts = [name_to_text(n) for n in names]
        started = time.perf_counter()
        vectors = self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        batch = max(1, len(texts))
        self.cost.batch_ms[batch] = elapsed_ms
        if batch == 1:
            self.cost.encode_ms = elapsed_ms
        return l2_normalize(vectors)


class PrecomputedBackend(EmbeddingBackend):
    """Vectors loaded from an ``.npz`` bundle written by the exporter.

    Unknown names are a hard error rather than a silent zero vector: a missing
    name means the export ran against a different catalog, and silently
    returning garbage would corrupt every downstream metric.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"no embedding bundle at {self.path}; run "
                f"experiments/export_embeddings.py on a machine with model access"
            )
        with np.load(self.path, allow_pickle=False) as bundle:
            names = [str(n) for n in bundle["names"]]
            vectors = np.asarray(bundle["vectors"], dtype=np.float32)
            self.id = str(bundle["model_id"].item()) if "model_id" in bundle else self.path.stem
        if len(names) != len(vectors):
            raise ValueError(
                f"bundle {self.path} has {len(names)} names but {len(vectors)} vectors"
            )
        self.dim = int(vectors.shape[1])
        self._table: Dict[str, np.ndarray] = dict(zip(names, l2_normalize(vectors)))

        meta_path = self.path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.cost = EncoderCost.from_dict(meta.get("cost", {"encode_ms": 0.0}))

    def encode(self, names: Sequence[str]) -> np.ndarray:
        missing = [n for n in names if n not in self._table]
        if missing:
            raise KeyError(
                f"{len(missing)} name(s) absent from {self.path.name}, "
                f"first: {missing[0]!r}. Re-run the exporter for this catalog."
            )
        return np.stack([self._table[n] for n in names]).astype(np.float32)

    def covers(self, names: Sequence[str]) -> bool:
        return all(n in self._table for n in names)


class LexicalBackend(EmbeddingBackend):
    """Stateless character n-gram hashing encoder -- the no-transformer control.

    Uses ``HashingVectorizer`` so there is nothing to fit and unseen names are
    handled natively, which matters because a router meets names it has never
    seen.  Deliberately compact (512 buckets, 2 KB per entry) so its memory
    footprint is comparable to a 384-d float32 embedding rather than
    flattering itself with a huge sparse space.
    """

    def __init__(self, n_features: int = 512, ngram_range: tuple[int, int] = (3, 5)) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import HashingVectorizer  # lazy

        self.id = f"lexical-char{ngram_range[0]}{ngram_range[1]}-{n_features}"
        self.dim = n_features
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            n_features=n_features,
            norm=None,
            alternate_sign=False,
        )
        self.cost = EncoderCost(encode_ms=0.0, measured=True)
        self.encode(["warm up"])

    def encode(self, names: Sequence[str]) -> np.ndarray:
        texts = [name_to_text(n) for n in names]
        started = time.perf_counter()
        sparse = self._vectorizer.transform(texts)
        dense = np.asarray(sparse.todense(), dtype=np.float32)
        vectors = l2_normalize(dense)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        batch = max(1, len(texts))
        self.cost.batch_ms[batch] = elapsed_ms
        if batch == 1:
            self.cost.encode_ms = elapsed_ms
        return vectors


class NameIndex:
    """A frozen table of name vectors with a cosine-similarity search.

    This is the object routers hold: FIB-entry embeddings on one side, an
    Interest embedding on the other.  Because every vector is unit length the
    similarity is a single matrix-vector product, which is also why brute force
    is the honest choice at the FIB sizes in play -- a 50x384 product is a few
    microseconds, so approximate search would optimise something that is not
    the bottleneck.
    """

    def __init__(self, backend: EmbeddingBackend, names: Sequence[str]) -> None:
        self.backend = backend
        self.names: List[str] = list(names)
        self._position = {name: i for i, name in enumerate(self.names)}
        self.vectors = backend.encode(self.names) if self.names else np.zeros((0, backend.dim), np.float32)

    def __len__(self) -> int:
        return len(self.names)

    def vector_of(self, name: str) -> np.ndarray:
        return self.vectors[self._position[name]]

    def contains(self, name: str) -> bool:
        return name in self._position

    def similarities(self, query: np.ndarray) -> np.ndarray:
        if not self.names:
            return np.zeros(0, dtype=np.float32)
        return self.vectors @ np.asarray(query, dtype=np.float32).reshape(-1)

    def best_match(
        self,
        query: np.ndarray,
        threshold: float,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Optional[tuple[str, float]]:
        """Highest-scoring entry at or above ``threshold``, else ``None``.

        Ties are broken by catalog order, which is deterministic; SAF breaks
        them by lowest round-trip time, a refinement that only matters when two
        entries score bit-identically.

        ``exclude`` skips entries already known not to serve this name.  An
        encoder is deterministic, so without it a second attempt at a name would
        return the same wrong answer as the first; the exclusion set is what
        turns a Nack into information rather than just a failure.
        """
        scores = self.similarities(query)
        if scores.size == 0:
            return None
        if exclude:
            scores = scores.copy()
            for name in exclude:
                position = self._position.get(name)
                if position is not None:
                    scores[position] = -np.inf
        best = int(np.argmax(scores))
        score = float(scores[best])
        return (self.names[best], score) if score >= threshold else None

    @property
    def memory_bytes(self) -> int:
        return int(self.vectors.nbytes)


def build_backend(
    kind: str = "precomputed",
    *,
    path: Optional[str | Path] = None,
    model_name: str = DEFAULT_MODEL,
    n_features: int = 512,
) -> EmbeddingBackend:
    """Construct a backend by name, for use from config and CLI flags."""
    if kind == "precomputed":
        if path is None:
            raise ValueError("the precomputed backend needs a bundle path")
        return PrecomputedBackend(path)
    if kind in ("onnx", "minilm", "onnx-minilm"):
        return OnnxMiniLMBackend(model_dir=path)
    if kind in ("sentence-transformer", "st"):
        return SentenceTransformerBackend(model_name)
    if kind == "lexical":
        return LexicalBackend(n_features=n_features)
    raise ValueError(
        f"unknown embedding backend {kind!r}; expected one of "
        f"'precomputed', 'onnx', 'sentence-transformer', 'lexical'"
    )
