"""Scoring a run.

The metric this replaces is the one the Phase 2 prototype reported as packet
delivery ratio.  That figure counted an Interest as delivered whenever something
came back, without ever checking that what came back was what the client asked
for -- so a semantic match that reached the wrong room's sensor scored as a
success.  Nothing in that prototype ever compared the matched name against the
ground truth.

Here every Interest ends in exactly one of four states:

``correct``      the Data returned came from the service the name should resolve to
``misdelivered`` Data or a refusal came back from some other service
``rejected``     a Nack, correct for a distractor and a miss for anything else
``timeout``      nothing came back inside the Interest lifetime

Because distractors are supposed to be rejected, a single accuracy number would
reward a strategy that answers nothing.  Precision, recall and false-positive
rate are reported separately, which is also the form SAF's own threshold tuning
takes.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .datasets import DISTRACTOR
from .des import Stopwatch
from .packets import OUTCOME_MISDELIVERED, OUTCOME_NACK, OUTCOME_TIMEOUT

CORRECT = "correct"
MISDELIVERED = "misdelivered"
REJECTED = "rejected"
TIMEOUT = "timeout"


@dataclass
class RequestRecord:
    """How one Interest ended."""

    name: str
    kind: str
    expected: Optional[str]
    verdict: str
    latency_ms: float
    consumer: str
    at_ms: float


@dataclass
class Collector:
    """Accumulates per-request outcomes during a run."""

    records: List[RequestRecord] = field(default_factory=list)
    latency: Stopwatch = field(default_factory=Stopwatch)

    def record(self, interest, outcome: str, latency_ms: float, at_ms: float) -> None:
        resolvable = interest.expected is not None
        if outcome == OUTCOME_TIMEOUT:
            verdict = TIMEOUT
        elif outcome == OUTCOME_NACK:
            verdict = REJECTED
        elif outcome == OUTCOME_MISDELIVERED:
            verdict = MISDELIVERED
        else:
            verdict = CORRECT

        if verdict == CORRECT and not resolvable:
            # A distractor cannot be answered correctly; anything returned for
            # one is a false positive by definition.
            verdict = MISDELIVERED

        self.records.append(
            RequestRecord(
                name=interest.name, kind=interest.kind, expected=interest.expected,
                verdict=verdict, latency_ms=latency_ms,
                consumer=interest.origin, at_ms=at_ms,
            )
        )
        if verdict == CORRECT:
            self.latency.record(latency_ms)

    def __len__(self) -> int:
        return len(self.records)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarise(collector: Collector) -> Dict[str, float]:
    """Turn raw records into the reported metrics."""
    records = collector.records
    total = len(records)
    if total == 0:
        return {"requests": 0}

    verdicts = Counter(r.verdict for r in records)
    resolvable = [r for r in records if r.expected is not None]
    unresolvable = [r for r in records if r.expected is None]

    correct = verdicts[CORRECT]
    misdelivered = verdicts[MISDELIVERED]
    forwarded_distractors = sum(1 for r in unresolvable if r.verdict == MISDELIVERED)

    # Recall over resolvable traffic; precision over everything that was
    # answered at all, so a confident wrong answer costs precision rather than
    # quietly counting as delivery.
    recall = _safe_div(correct, len(resolvable))
    precision = _safe_div(correct, correct + misdelivered)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    latency = collector.latency
    metrics: Dict[str, float] = {
        "requests": total,
        "resolvable": len(resolvable),
        "distractors": len(unresolvable),
        "correct": correct,
        "misdelivered": misdelivered,
        "rejected": verdicts[REJECTED],
        "timeout": verdicts[TIMEOUT],

        "isr": recall,                       # Interest Satisfaction Rate
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _safe_div(forwarded_distractors, len(unresolvable)),
        "accuracy": _safe_div(
            correct + sum(1 for r in unresolvable if r.verdict == REJECTED), total
        ),

        "irt_mean_ms": latency.mean,
        "irt_p50_ms": latency.percentile(50),
        "irt_p95_ms": latency.percentile(95),
        "irt_p99_ms": latency.percentile(99),
        "irt_max_ms": latency.percentile(100),
    }

    by_kind: Dict[str, List[RequestRecord]] = defaultdict(list)
    for record in records:
        by_kind[record.kind].append(record)
    for kind, group in by_kind.items():
        metrics[f"isr_{kind}"] = _safe_div(
            sum(1 for r in group if r.verdict == CORRECT), len(group)
        )
    return metrics


def convergence_curve(
    collector: Collector, window: int = 200
) -> List[Tuple[float, float]]:
    """Rolling share of resolvable requests served without an encoder run.

    Latency is the visible symptom; this is the underlying quantity. It rises as
    routers learn, and how fast it rises is the difference between learning
    alone and being taught.
    """
    points: List[Tuple[float, float]] = []
    fast = 0
    seen: List[bool] = []
    for record in collector.records:
        if record.expected is None:
            continue
        is_fast = record.latency_ms < _FAST_PATH_CEILING_MS
        seen.append(is_fast)
        fast += int(is_fast)
        if len(seen) > window:
            fast -= int(seen.pop(0))
        points.append((record.at_ms, fast / len(seen)))
    return points


#: An Interest resolved without running an encoder completes within the round
#: trip plus table lookups. Anything materially above that paid for inference.
_FAST_PATH_CEILING_MS = 75.0


def aggregate(runs: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean and 95% confidence half-width across repeated runs.

    SEF averages 20 seeded runs; a single run of a stochastic simulation is an
    anecdote, and reporting one number without a spread invites a reader to
    treat noise as a result.
    """
    if not runs:
        return {}
    keys = set().union(*(set(r) for r in runs))
    out: Dict[str, Dict[str, float]] = {}
    for key in sorted(keys):
        values = [float(r[key]) for r in runs if key in r]
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            stderr = math.sqrt(variance / n)
            # Normal approximation; with 20 seeds the t correction is under 5%.
            half_width = 1.96 * stderr
        else:
            variance = 0.0
            half_width = 0.0
        out[key] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "ci95": half_width,
            "n": n,
            "min": min(values),
            "max": max(values),
        }
    return out


def format_table(
    rows: Dict[str, Dict[str, Dict[str, float]]],
    columns: Sequence[str],
    label: str = "strategy",
) -> str:
    """Render aggregated results as a fixed-width table with confidence bounds."""
    width = max(len(label), *(len(name) for name in rows)) if rows else len(label)
    header = f"{label:<{width}}  " + "  ".join(f"{c:>18}" for c in columns)
    lines = [header, "-" * len(header)]
    for name, stats in rows.items():
        cells = []
        for column in columns:
            entry = stats.get(column)
            if entry is None:
                cells.append(f"{'--':>18}")
            elif entry["n"] > 1:
                cells.append(f"{entry['mean']:>11.3f}±{entry['ci95']:<6.3f}")
            else:
                cells.append(f"{entry['mean']:>18.3f}")
        lines.append(f"{name:<{width}}  " + "  ".join(cells))
    return "\n".join(lines)
