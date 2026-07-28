#!/usr/bin/env python3
"""Compare the Python model's transport behaviour against ndnSIM's.

    python ndnsim/compare.py --delays <ns-3>/results-ndnsim-delays.txt

Runs the Python simulator on the configuration the ndnSIM scenario encodes and
prints both round-trip time distributions side by side.

What this checks is deliberately narrow. Semantic resolution is the thing under
study and has no ndnSIM counterpart, so it is switched off here: only exact-match
forwarding runs, and the question is whether a hand-written NDN model reproduces
what a real NDN stack does with the same links, the same topology and the same
request pattern. If it does not, every strategy comparison built on it inherits
the error.

One difference is expected and is not a disagreement: the Python model charges a
producer 1 ms to answer, and ndnSIM's ``ns3::ndn::Producer`` answers instantly.
The Python figure should therefore sit about 1 ms high.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsndn.metrics import CORRECT  # noqa: E402
from gsndn.runner import ScenarioConfig, run_once  # noqa: E402
from gsndn.workload import WorkloadConfig  # noqa: E402

#: The producer service time the Python model charges and ndnSIM does not.
PRODUCER_SERVICE_MS = 1.0


def read_ndnsim(path: Path) -> List[float]:
    """Full round-trip delays, in milliseconds, from an AppDelayTracer file."""
    delays: List[float] = []
    with path.open() as handle:
        header = handle.readline().split()
        column = {name: i for i, name in enumerate(header)}
        for line in handle:
            fields = line.split()
            if fields[column["Type"]] == "FullDelay":
                delays.append(float(fields[column["DelayS"]]) * 1000.0)
    return delays


def percentiles(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q / 100.0))]

    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "p50": at(50),
        "p95": at(95),
        "p99": at(99),
    }


def run_python(edges: int, rate: float, duration_s: float, seed: int) -> List[float]:
    """The same scenario, exact-match forwarding only."""
    config = ScenarioConfig(
        domain="hospital",
        strategy="vanilla-ndn",
        topology="multi_edge",
        n_edges=edges,
        cs_capacity=0,
        workload=WorkloadConfig(
            rate_per_s=rate,
            duration_ms=duration_s * 1000.0,
            # Exact names only: ndnSIM's consumer asks for names the producer
            # publishes, so a reworded name has no counterpart to compare with.
            variation_rate=0.0,
            distractor_rate=0.0,
            overlap=1.0,
            seed=seed,
        ),
    )
    result = run_once(config)
    return [r.latency_ms for r in result.collector.records if r.verdict == CORRECT]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delays", type=Path, required=True,
                       help="AppDelayTracer output from the ndnSIM scenario")
    parser.add_argument("--edges", type=int, default=6)
    parser.add_argument("--rate", type=float, default=150.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2.0,
                       help="allowed median gap in ms, after the producer offset")
    args = parser.parse_args()

    if not args.delays.exists():
        parser.error(
            f"{args.delays} not found. Run the scenario first:\n"
            f'  ./waf --run "gsndn-validation --edges={args.edges} --rate={args.rate}"'
        )

    ndnsim = percentiles(read_ndnsim(args.delays))
    python = percentiles(run_python(args.edges, args.rate, args.duration, args.seed))

    print(f"\n{args.edges} edge routers, {args.rate:.0f} Interests/s, "
          f"{args.duration:.0f} s, exact-match forwarding\n")
    print(f"{'':<10}{'n':>8}{'mean':>10}{'p50':>10}{'p95':>10}{'p99':>10}")
    for label, stats in (("ndnSIM", ndnsim), ("gsndn", python)):
        print(f"{label:<10}{stats['n']:>8}{stats['mean']:>10.2f}"
              f"{stats['p50']:>10.2f}{stats['p95']:>10.2f}{stats['p99']:>10.2f}")

    gap = python["p50"] - ndnsim["p50"]
    residual = abs(gap - PRODUCER_SERVICE_MS)
    print(f"\nmedian gap            : {gap:+.2f} ms")
    print(f"expected offset       : {PRODUCER_SERVICE_MS:+.2f} ms "
          f"(producer service time, modelled here and not in ndnSIM)")
    print(f"unexplained residual  : {residual:.2f} ms")

    if residual <= args.tolerance:
        print(f"\nAGREE — within {args.tolerance:.1f} ms once the producer offset is accounted for.")
        return 0
    print(f"\nDISAGREE — {residual:.2f} ms unaccounted for, above the {args.tolerance:.1f} ms tolerance.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
