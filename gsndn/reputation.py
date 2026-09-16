"""Which neighbours have earned the right to be believed.

Section 9 measures an attack and then names its own missing defence: "Provenance
(signing an observation with the router that made it, so a lie can be
attributed) and reputation (down-weighting a peer whose contributions are
repeatedly refuted) are the two obvious defences, and neither is here." This
module is the second one, and it is cheaper here than the literature would
predict, for a reason that is specific to this system rather than clever.

**Why reputation is nearly free in a verifying forwarding plane.** Byzantine-
robust gossip normally has to defend without ground truth: in gossip learning a
neighbour ships a gradient, and no honest node can check it, so the defences are
statistical -- clip the update, trim the tails, bound the contribution
(EdgeClippedGossip, arXiv 2405.03449, and the robust-aggregation line behind
it). Those methods bound an adversary's *influence* because they cannot detect
the adversary.

Here a router can detect one, for free, using a signal it was already
collecting. A gossiped mapping is a falsifiable claim -- "this wording is served
by that route" -- and NDN answers it: forward on the mapping and the producer
either returns Data or refuses. That is the same free label §6 calibrates
boundaries from, read a second time and attributed to whoever supplied the
claim. So the defence is not "bound what a peer can do to me" but "find out
which peers are lying, from evidence the protocol already produces."

**Two channels, two failure modes, two defences.** §9 established that the
mapping channel and the evidence channel break differently, and they need
different answers:

*Poisoned mappings* are self-limiting per round and unlimited across rounds. A
victim installs one, forwards on it, gets refused, drops it -- and the attacker
re-injects it next round. The existing design retracts but does not *remember*:
:class:`gsndn.strategies.semantic.GsNdn` records the refuted prefix in its own
``refuted`` table and excludes it from the next encoder run, but
:meth:`gsndn.gossip.GossipAgent.apply` never consults that table, so a pair the
router disproved first-hand is reinstalled from gossip minutes later. That is
the concrete gap :attr:`ReputationTable.refuted_pairs` closes, and it is why
``gs-ndn`` and ``gs-ndn-no-verify`` degrade almost identically under a
persistent attacker in §9: against re-injection, one-shot retraction is barely
verification at all.

*Poisoned evidence* is quieter, as §9 found: fabricated "this score worked"
observations pull a boundary down and nothing ever looks wrong. Detection is
weaker here -- an observation about a route is not immediately falsifiable the
way a mapping is -- so this channel keeps the statistical defence the
literature prescribes: no single peer may occupy more than
:attr:`peer_share` of a route's calibration window. That is a breakdown-point
argument, not a detection argument, and it is stated as such.

**The bound direction.** Trust is a Wilson *lower* confidence bound on a peer's
success rate, mirroring :func:`gsndn.risk._upper_error_bound`, which bounds
error from above. Both bound in the direction that costs us: a peer with two
lucky confirmations has not proved anything, exactly as a route with two lucky
observations has not.

**What this does not do.** It does not authenticate anyone, so it is orthogonal
to signing rather than a substitute for it -- an attacker who can forge a peer
identity gets a fresh reputation each time, and §9's threat model excludes that
capability precisely because this defence does not survive it. It does not
detect a compromised *producer*, which §9 also leaves unmeasured. And it is
per-router local state: no consensus, no shared reputation, no coordinator,
which keeps it inside the same "no coordinator" claim the rest of the design
makes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from .risk import _z_for

#: Below this many judged claims a peer keeps the benefit of the doubt.
#:
#: It has to be a real number rather than zero. A router that distrusts every
#: peer it has not yet audited cannot bootstrap: gossip's whole value is the
#: first mapping a neighbour teaches you, and that one arrives before any
#: evidence about that neighbour exists. So a fresh peer is believed, and the
#: cost of that choice is bounded and worth stating -- an attacker gets this
#: many free lies per victim per identity, after which its trust collapses and
#: stays collapsed. Against the persistent re-injecting attacker of §9 that is
#: a fixed cost paid once, not a per-round cost.
#:
#: Two rather than four, taken from the sweep in ``exp_breakdown`` rather than
#: from intuition. At a compromised share of 0.25 a fee of 2 realises 0.0374
#: error against 4's 0.0437, and on an honest network it falsely distrusts 1.1
#: peers against 0.4, costing 0.06% more encoder work -- inside the noise.
#:
#: A fee of 1 is better still under attack (0.0340) and is not worth taking: a
#: single good-faith misroute then collapses a peer's trust outright, so 10.9
#: honest peers end up distrusted on a clean network and sharing degrades by
#: 3%. The honest-network control is what makes that visible; the attack
#: numbers alone would have argued for 1.
MIN_CLAIMS = 2

#: Trust below which a peer's mappings stop being installed.
DEFAULT_FLOOR = 0.5

#: Most of one route's calibration window any single peer may occupy.
#:
#: The evidence channel cannot be audited the way the mapping channel can, so
#: what bounds it is arithmetic: with a cap of s, a peer contributes at most s
#: of the observations behind a boundary regardless of how much it sends, and
#: an attacker controlling k peers is bounded by k*s rather than by its message
#: volume. This is the "damage is bounded by the fraction of evidence an
#: attacker supplies" claim in :mod:`gsndn.adversary`, made true by enforcement
#: instead of left as an observation.
DEFAULT_PEER_SHARE = 0.25


@dataclass
class PeerRecord:
    """What one neighbour's claims turned out to be worth."""

    confirmed: int = 0
    refuted: int = 0
    mappings_offered: int = 0
    mappings_installed: int = 0
    evidence_offered: int = 0
    evidence_installed: int = 0

    @property
    def judged(self) -> int:
        return self.confirmed + self.refuted


def _wilson_lower(successes: int, trials: int, confidence: float) -> float:
    """One-sided lower confidence bound on a success rate.

    The mirror of :func:`gsndn.risk._upper_error_bound`. Same reason for the
    Wilson form rather than the normal approximation: the counts are small, and
    a peer with three clean claims should not be handed a trust of 1.0.
    """
    if trials <= 0:
        return 1.0
    z = _z_for(confidence)
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = phat + z * z / (2 * trials)
    spread = z * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4 * trials * trials))
    return max(0.0, (centre - spread) / denominator)


@dataclass
class ReputationTable:
    """One router's view of the neighbours that teach it.

    Local state only: nothing here is exchanged, so two routers may hold
    different opinions of the same peer and neither has to reconcile with the
    other. That is deliberate -- a shared reputation is itself a gossiped
    quantity and would need its own defence.
    """

    floor: float = DEFAULT_FLOOR
    confidence: float = 0.9
    peer_share: float = DEFAULT_PEER_SHARE
    min_claims: int = MIN_CLAIMS

    peers: Dict[str, PeerRecord] = field(default_factory=dict)

    #: (variant, canonical) pairs this router disproved first-hand. Membership
    #: is the only operation performed on it, never iteration, so it cannot
    #: leak set ordering into the run -- see the determinism note in
    #: :meth:`gsndn.gossip.GossipAgent.local_digest`.
    refuted_pairs: Set[Tuple[str, str]] = field(default_factory=set)

    mappings_blocked_untrusted: int = 0
    mappings_blocked_refuted: int = 0
    evidence_blocked_untrusted: int = 0
    evidence_blocked_capped: int = 0

    def record(self, peer_id: str) -> PeerRecord:
        record = self.peers.get(peer_id)
        if record is None:
            record = PeerRecord()
            self.peers[peer_id] = record
        return record

    # -- auditing --------------------------------------------------------

    def credit(self, peer_id: str) -> None:
        """A mapping this peer taught us was proved by a producer."""
        self.record(peer_id).confirmed += 1

    def debit(self, peer_id: str, variant: str, canonical: str) -> None:
        """A mapping this peer taught us was refused by the producer."""
        record = self.record(peer_id)
        record.refuted += 1
        self.refuted_pairs.add((variant, canonical))

    def note_refuted(self, variant: str, canonical: str) -> None:
        """A pair this router disproved first-hand, whoever it came from.

        Recorded even for locally resolved mappings, because the attacker's
        target is a wording clients actually request: once this router has
        established that the pair does not work, a neighbour asserting it is
        either compromised or about to be corrected, and installing it again
        costs a round trip either way.
        """
        self.refuted_pairs.add((variant, canonical))

    def trust(self, peer_id: str) -> float:
        record = self.peers.get(peer_id)
        if record is None or record.judged < self.min_claims:
            return 1.0
        return _wilson_lower(record.confirmed, record.judged, self.confidence)

    # -- admission -------------------------------------------------------

    def admits_mapping(self, peer_id: str, variant: str, canonical: str) -> bool:
        record = self.record(peer_id)
        record.mappings_offered += 1
        if (variant, canonical) in self.refuted_pairs:
            self.mappings_blocked_refuted += 1
            return False
        if self.trust(peer_id) < self.floor:
            self.mappings_blocked_untrusted += 1
            return False
        record.mappings_installed += 1
        return True

    def admits_evidence(self, peer_id: str, held_from_peer: int, window: int) -> bool:
        """May this peer add one more observation to this route's window?

        ``held_from_peer`` is how many of the route's current observations came
        from this peer, and ``window`` the calibrator's capacity. The cap is on
        the share of the window rather than on a rate, because what corrupts a
        boundary is the composition of the evidence behind it, not how fast it
        arrived.
        """
        record = self.record(peer_id)
        record.evidence_offered += 1
        if self.trust(peer_id) < self.floor:
            self.evidence_blocked_untrusted += 1
            return False
        if window > 0 and held_from_peer >= self.peer_share * window:
            self.evidence_blocked_capped += 1
            return False
        record.evidence_installed += 1
        return True

    # -- reporting -------------------------------------------------------

    def stats(self) -> Dict[str, float]:
        judged = [r for r in self.peers.values() if r.judged >= self.min_claims]
        distrusted = sum(
            1 for peer_id in self.peers if self.trust(peer_id) < self.floor
        )
        trusts = [self.trust(p) for p in self.peers]
        return {
            "rep_peers_known": len(self.peers),
            "rep_peers_judged": len(judged),
            "rep_peers_distrusted": distrusted,
            "rep_trust_mean": sum(trusts) / len(trusts) if trusts else 1.0,
            "rep_trust_min": min(trusts) if trusts else 1.0,
            "rep_refuted_pairs": len(self.refuted_pairs),
            "rep_mappings_blocked_untrusted": self.mappings_blocked_untrusted,
            "rep_mappings_blocked_refuted": self.mappings_blocked_refuted,
            "rep_evidence_blocked_untrusted": self.evidence_blocked_untrusted,
            "rep_evidence_blocked_capped": self.evidence_blocked_capped,
        }
