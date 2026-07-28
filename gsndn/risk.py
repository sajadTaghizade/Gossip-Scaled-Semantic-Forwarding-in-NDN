"""Choosing where to draw the line, per route, from evidence rather than by hand.

SAF picks a single cosine threshold -- 0.7 -- by tuning it on its own catalog.
Section 6 of the results shows what that costs elsewhere: on the two catalogs
here, 0.7 gives recall of 0.66 and 0.50 while the best operating point sits
nearer 0.55 and 0.45. A threshold is not a property of the method; it is a
property of the catalog it was tuned on, and it does not travel.

The deeper problem is that one number has to serve every route. Some services
sit alone in embedding space and a loose score is safe; others have five
near-identical siblings and even a high score is a coin flip. A global threshold
is simultaneously too strict for the first and too permissive for the second.

What replaces it here is a per-route decision boundary, learned online, from
evidence the network produces for free. Every semantic forwarding decision is an
experiment: forward, and the producer's answer says whether it was right. Those
(score, verdict) pairs are exactly a calibration set, and the operator sets a
target error rate instead of a similarity cutoff:

    "forward only when at most epsilon of such decisions have historically
     been wrong for this route"

Concretely, for route *r* with observations ``(s_i, y_i)``, the boundary is the
smallest score ``tau_r`` such that among past observations scoring at or above
``tau_r`` the empirical error rate is at most ``epsilon``, with a finite-sample
correction so that a route seen three times does not get to claim a 5%
guarantee. Until a route has enough evidence it falls back to the global prior,
so the system is never worse than a fixed threshold and improves as it learns.

Relation to prior work, stated plainly. Learning per-item boundaries with an
error bound instead of a fixed similarity threshold is what vCache does for LLM
prompt caching (arXiv 2502.03771), and pooling calibration across parties is
federated conformal prediction (Lu et al., ICML 2023). Neither is in a network,
and the reason this is practical here is specific to NDN: an LLM cache needs a
judge model to label a hit, which is expensive enough to be the thing you were
avoiding, whereas a forwarding plane gets its labels from Data and Nack packets
it was going to exchange anyway. The labels are a by-product of forwarding.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

#: Below this many observations a route keeps using the global prior. Chosen so
#: the finite-sample correction is not doing all the work.
MIN_OBSERVATIONS = 8

#: How many observations a route remembers. Bounded because the right boundary
#: moves when routes churn, and because a router cannot store history forever.
WINDOW = 256

#: Range the adaptive budget is allowed to move in. The floor stops a run of bad
#: luck from driving the effective budget to zero, which would be the deadlock of
#: section 5 arrived at by a different route; the ceiling stops a long clean
#: stretch from inflating it past anything an operator would have asked for.
ACI_EPSILON_MIN = 0.001
ACI_EPSILON_MAX = 0.6


@dataclass(frozen=True)
class Observation:
    """One resolved-and-answered forwarding decision."""

    score: float
    correct: bool
    at_ms: float = 0.0
    source: str = "local"


@dataclass
class RouteCalibrator:
    """The evidence for one route, and the boundary it implies."""

    prefix: str
    window: int = WINDOW
    observations: Deque[Observation] = field(default_factory=deque)

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)
        while len(self.observations) > self.window:
            self.observations.popleft()

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def errors(self) -> int:
        return sum(1 for o in self.observations if not o.correct)

    def raw_boundary(self, epsilon: float, confidence: float) -> Optional[float]:
        """Lowest cutoff whose upper-bounded error rate stays within budget.

        Sweeps candidate cutoffs downward through the observed scores and keeps
        the lowest one that still satisfies the budget. The *upper* confidence
        bound rather than the point estimate is what stops a route with two
        lucky observations from declaring itself safe.

        ``None`` means no cutoff on this evidence meets the budget -- the route
        is not safe to use at any score it has been seen at.
        """
        if not self.observations:
            return None
        ordered = sorted(self.observations, key=lambda o: o.score, reverse=True)
        errors = 0
        best: Optional[float] = None
        for seen, observation in enumerate(ordered, start=1):
            if not observation.correct:
                errors += 1
            if _upper_error_bound(errors, seen, confidence) <= epsilon:
                best = observation.score
        return best

    def empirical_error(self, boundary: float) -> Tuple[int, int]:
        """Errors and total among observations at or above ``boundary``."""
        above = [o for o in self.observations if o.score >= boundary]
        return sum(1 for o in above if not o.correct), len(above)


def _upper_error_bound(errors: int, trials: int, confidence: float) -> float:
    """One-sided upper confidence bound on an error rate.

    The Wilson interval rather than the normal approximation, because the counts
    here are small and skewed -- a route with 10 observations and 0 errors has a
    normal-approximation bound of exactly 0, which is the kind of claim that
    makes a guarantee meaningless.
    """
    if trials <= 0:
        return 1.0
    z = _z_for(confidence)
    phat = errors / trials
    denominator = 1.0 + z * z / trials
    centre = phat + z * z / (2 * trials)
    spread = z * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4 * trials * trials))
    return min(1.0, (centre + spread) / denominator)


def _stable_unit(key: str, seed: int) -> float:
    """Deterministic value in [0, 1) -- exploration without a shared RNG."""
    import hashlib

    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _z_for(confidence: float) -> float:
    """Normal quantile for the common confidence levels, without scipy."""
    table = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}
    if confidence in table:
        return table[confidence]
    nearest = min(table, key=lambda c: abs(c - confidence))
    return table[nearest]


@dataclass
class RiskController:
    """Per-route boundaries for one router, plus the evidence behind them.

    ``epsilon`` is the operator-facing knob and replaces the similarity
    threshold entirely: the share of semantic forwarding decisions that may turn
    out wrong.

    Boundaries are estimated hierarchically, which is not a refinement but a
    necessity. A single route is seen a few dozen times at most, and a
    conservative bound on a few dozen observations is nearly vacuous -- an early
    version of this returned "refuse everything" for every calibrated route,
    because at epsilon = 0.05 the Wilson bound needs upwards of thirty clean
    observations before it will concede anything. Pooling every route's evidence
    gives one well-supported global boundary; each route then shrinks from that
    toward its own estimate at a rate set by how much evidence it has:

        tau_r = w_r * tau_route + (1 - w_r) * tau_global,   w_r = n_r / (n_r + k)

    So a route with no history behaves exactly like the pooled boundary, a route
    with plenty of history follows its own, and nothing in between claims more
    than its evidence supports. ``prior`` -- the fixed similarity threshold --
    survives only for the first moments of a run, before the pool itself has
    enough observations to say anything.
    """

    epsilon: float = 0.05
    prior: float = 0.6
    confidence: float = 0.9
    window: int = WINDOW

    #: Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021). Off by
    #: default, in which case ``epsilon`` is exactly the operator's number and
    #: never moves.
    #:
    #: The fixed-epsilon controller assumes exchangeability: that the decisions
    #: it calibrated on and the decisions it now faces come from the same
    #: distribution. Producer churn and schema drift break that assumption
    #: directly, and section 14 has been listing "a weighted or adaptive variant
    #: would be the principled fix" as unimplemented. This is that variant.
    #:
    #: Instead of holding epsilon fixed and hoping the calibration transfers, the
    #: controller tracks an effective budget that reacts to realised coverage:
    #:
    #:     eps_{t+1} = eps_t + gamma * (eps_target - err_t)
    #:
    #: with ``err_t`` the miscoverage indicator of the decision just judged --
    #: one if the producer refused, zero if it answered. A wrong decision pushes
    #: the effective budget down, which raises the boundary and makes the router
    #: stricter; a clean stretch lets it drift back up towards the target. The
    #: guarantee it buys is different in kind from the fixed version's: not "at
    #: most eps of decisions were wrong" on any finite run, but that the
    #: long-run realised rate tracks the target even when the distribution moves
    #: under it.
    adaptive: bool = False
    #: Step size. Larger reacts faster and oscillates more.
    adapt_rate: float = 0.05
    #: The operator's number, which ``epsilon`` drifts around under adaptation.
    epsilon_target: Optional[float] = None
    adapt_steps: int = 0
    _epsilon_sum: float = 0.0
    #: Shrinkage constant: how much route-local evidence is worth half the pool.
    shrinkage: float = float(MIN_OBSERVATIONS)

    #: Share of blocked decisions forwarded anyway, to keep learning.
    #:
    #: Without this the controller deadlocks, and the deadlock is not a bug in
    #: the estimator but a property of learning from your own choices: a
    #: boundary set too high blocks the decisions that would have produced the
    #: evidence to lower it, so the system stops forwarding, stops learning, and
    #: stays there. Measured directly -- at epsilon <= 0.10 an unexploring
    #: controller settles at "refuse everything" and never recovers, while at
    #: epsilon = 0.20 it forwards enough to gather nine times the evidence and
    #: settles at 0.69. Spending a small share of decisions on evidence is what
    #: makes a tight budget reachable rather than self-defeating.
    explore_rate: float = 0.05
    explore_seed: int = 0

    routes: Dict[str, RouteCalibrator] = field(default_factory=dict)
    pooled: RouteCalibrator = field(
        default_factory=lambda: RouteCalibrator(prefix="*", window=4096)
    )
    _cache: Dict[str, float] = field(default_factory=dict)

    observations_local: int = 0
    observations_remote: int = 0
    decisions_allowed: int = 0
    decisions_blocked: int = 0
    decisions_explored: int = 0
    _explore_counter: int = 0

    def __post_init__(self) -> None:
        if self.epsilon_target is None:
            self.epsilon_target = self.epsilon

    # -- adaptive budget --------------------------------------------------

    def adapt(self, err: float) -> float:
        """One Adaptive Conformal Inference step. Returns the new budget.

        ``err`` is the miscoverage signal for the decision just judged: 1.0 if it
        turned out wrong, 0.0 if right. Passing a rate rather than an indicator
        works identically and is what the unit test uses to check the direction
        of movement without simulating a stream.

        Kept public and free of any router state so the update rule can be tested
        as arithmetic, which is the only part of this that is a claim about a
        published method rather than about our system.
        """
        target = self.epsilon_target if self.epsilon_target is not None else self.epsilon
        moved = self.epsilon + self.adapt_rate * (target - err)
        self.epsilon = min(ACI_EPSILON_MAX, max(ACI_EPSILON_MIN, moved))
        self.adapt_steps += 1
        self._epsilon_sum += self.epsilon
        # Every boundary is a function of epsilon, so all of them just moved.
        self._cache.clear()
        return self.epsilon

    def global_boundary(self) -> float:
        """The pooled boundary every route starts from."""
        if len(self.pooled) < MIN_OBSERVATIONS:
            return self.prior
        boundary = self.pooled.raw_boundary(self.epsilon, self.confidence)
        # Even pooled, the budget may be unreachable -- an encoder that cannot
        # separate this catalog at the requested error rate should forward
        # nothing rather than pretend.
        return boundary if boundary is not None else 1.0

    def boundary_for(self, prefix: str) -> float:
        cached = self._cache.get(prefix)
        if cached is not None:
            return cached

        pooled = self.global_boundary()
        calibrator = self.routes.get(prefix)
        if calibrator is None or len(calibrator) == 0:
            boundary = pooled
        else:
            local = calibrator.raw_boundary(self.epsilon, self.confidence)
            local = 1.0 if local is None else local
            weight = len(calibrator) / (len(calibrator) + self.shrinkage)
            boundary = weight * local + (1.0 - weight) * pooled

        self._cache[prefix] = boundary
        return boundary

    def allows(self, prefix: str, score: float) -> Tuple[bool, bool]:
        """May this route be used at this score, and was it an exploration?

        Returns ``(allowed, exploring)``. An exploratory forward is one the
        boundary refused and the controller took anyway; the harness counts them
        separately, because they are decisions the operator's budget explicitly
        did not cover and it would be dishonest to fold them into the same rate.
        """
        if score >= self.boundary_for(prefix):
            self.decisions_allowed += 1
            return True, False

        self.decisions_blocked += 1
        if self.explore_rate > 0.0:
            self._explore_counter += 1
            if _stable_unit(f"{prefix}|{self._explore_counter}", self.explore_seed) < self.explore_rate:
                self.decisions_explored += 1
                return True, True
        return False, False

    def observe(self, prefix: str, observation: Observation) -> None:
        if self.adaptive:
            # Adapt only on decisions the boundary in force actually covered.
            # An exploratory forward below the boundary is a decision the budget
            # explicitly did not cover -- section 5 counts those separately, and
            # feeding them into the coverage signal would make the controller
            # punish itself for the evidence-gathering it needs.
            if observation.score >= self.boundary_for(prefix):
                self.adapt(0.0 if observation.correct else 1.0)

        calibrator = self.routes.get(prefix)
        if calibrator is None:
            calibrator = RouteCalibrator(prefix=prefix, window=self.window)
            self.routes[prefix] = calibrator
        calibrator.add(observation)
        self.pooled.add(observation)
        # A new observation moves this route's boundary directly and every other
        # route's through the pool, so the whole cache goes.
        self._cache.clear()
        if observation.source == "local":
            self.observations_local += 1
        else:
            self.observations_remote += 1

    def absorb(self, prefix: str, observations: Iterable[Observation]) -> int:
        """Take in evidence gathered elsewhere. Returns how much was used."""
        taken = 0
        for observation in observations:
            self.observe(prefix, observation)
            taken += 1
        return taken

    # -- reporting -------------------------------------------------------

    def calibrated_routes(self) -> int:
        return sum(1 for c in self.routes.values() if len(c) >= MIN_OBSERVATIONS)

    @property
    def pooled_observations(self) -> int:
        return len(self.pooled)

    def realised_error(self) -> Tuple[int, int]:
        """Errors and decisions among everything the boundaries let through.

        This is the number that says whether the guarantee held. It is computed
        from the same evidence the boundaries were fitted on, so it is optimistic
        by construction; the harness measures the honest version out-of-sample,
        from what actually happened to forwarded Interests.
        """
        errors = total = 0
        for prefix, calibrator in self.routes.items():
            route_errors, route_total = calibrator.empirical_error(self.boundary_for(prefix))
            errors += route_errors
            total += route_total
        return errors, total

    def boundary_spread(self) -> Dict[str, float]:
        """How far the learned boundaries have moved apart.

        A tight spread means one global threshold would have done; a wide one is
        the evidence that per-route calibration was necessary.
        """
        boundaries = [
            self.boundary_for(p) for p, c in self.routes.items() if len(c) > 0
        ]
        if not boundaries:
            return {"n": 0}
        boundaries.sort()
        mean = sum(boundaries) / len(boundaries)
        variance = sum((b - mean) ** 2 for b in boundaries) / len(boundaries)
        return {
            "n": len(boundaries),
            "min": boundaries[0],
            "max": boundaries[-1],
            "mean": mean,
            "std": math.sqrt(variance),
            "range": boundaries[-1] - boundaries[0],
        }

    def stats(self) -> Dict[str, float]:
        errors, total = self.realised_error()
        spread = self.boundary_spread()
        return {
            "risk_epsilon": self.epsilon,
            "risk_epsilon_target": (
                self.epsilon_target if self.epsilon_target is not None else self.epsilon
            ),
            "risk_epsilon_mean": (
                self._epsilon_sum / self.adapt_steps if self.adapt_steps else self.epsilon
            ),
            "risk_adapt_steps": self.adapt_steps,
            "risk_routes_tracked": len(self.routes),
            "risk_routes_calibrated": spread.get("n", 0),
            "risk_observations_local": self.observations_local,
            "risk_observations_remote": self.observations_remote,
            "risk_decisions_allowed": self.decisions_allowed,
            "risk_decisions_blocked": self.decisions_blocked,
            "risk_decisions_explored": self.decisions_explored,
            "risk_insample_error": errors / total if total else 0.0,
            "risk_boundary_mean": spread.get("mean", self.prior),
            "risk_boundary_std": spread.get("std", 0.0),
            "risk_boundary_range": spread.get("range", 0.0),
            "risk_boundary_global": self.global_boundary(),
            "risk_pooled_observations": len(self.pooled),
        }
