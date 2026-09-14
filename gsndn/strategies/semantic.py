"""Vanilla NDN, SAF, SAF+ES and GS-NDN.

The three published behaviours and ours, sharing one code path so the
differences are visible rather than buried in four separate implementations.

``VanillaNdn``
    Exact match or nothing.  A name the FIB does not contain is dropped with a
    Nack, which is standard NDN and the reason the whole problem exists.

``Saf``
    Amadeo et al., IEEE IoT Magazine 2026.  On a FIB miss, encode the name and
    take the highest-scoring FIB entry above a threshold.  Every miss pays for
    an encoder run.  The resolved prefix is attached to the Interest -- the
    paper prepends it with a ``|`` delimiter -- so routers further along
    recognise it without recomputing the score, and so the producer receives a
    name it actually serves.

``SafWithEs``
    SAF plus its Embedding Store: the resolved route is cached under the
    Interest name, so a repeat of the same wording skips the encoder.  Caching
    happens on resolution, before anything has confirmed the route was right --
    faithful to the paper, and the behaviour our variant departs from.

``GsNdn``
    Ours.  Edge tagging is *not* one of our contributions -- it is SAF's, and it
    is inherited here rather than claimed.  Two things are new:

    1. *Mappings are verified before they are trusted.*  A semantic match is a
       guess, and NDN checks it for free: Data comes back only if the chosen
       producer really serves the name.  A guess that fails is retracted rather
       than cached, so a permissive threshold stops meaning permanent
       misrouting.  SAF caches on resolution and therefore keeps serving a wrong
       match once it has made one.
    2. *Verified mappings are shared.*  Anti-entropy gossip in
       :mod:`gsndn.gossip` spreads what one router proved to its neighbours, so
       the network pays the encoder cost roughly once per distinct name rather
       than once per distinct name per router.  SAF resolves at a single router
       and shares nothing; INF-NDN shares, but through a central Principal Node.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Set

from ..costs import CostModel
from ..packets import (
    REFUSAL_UNKNOWN_WORDING,
    OUTCOME_ES_HIT,
    OUTCOME_GOSSIP_HIT,
    OUTCOME_NACK,
    OUTCOME_SEMANTIC,
    OUTCOME_TAGGED,
    SemanticTag,
)
from ..reputation import DEFAULT_FLOOR, DEFAULT_PEER_SHARE, ReputationTable
from ..tables import EsEntry, PendingMapping
from .base import ForwardingStrategy

if TYPE_CHECKING:  # pragma: no cover
    from ..network import Resolution, Router
    from ..packets import Interest


class VanillaNdn(ForwardingStrategy):
    """Exact longest-prefix match only."""

    name = "vanilla-ndn"
    semantic = False

    def resolve(self, router: "Router", interest: "Interest") -> "Resolution":
        from ..network import Resolution

        return Resolution(outcome=OUTCOME_NACK, cpu_ms=0.0)


class Saf(ForwardingStrategy):
    """Semantic-Aware Forwarding: encode on every FIB miss."""

    name = "saf"
    semantic = True

    def __init__(self, threshold: float = 0.7, costs: Optional[CostModel] = None) -> None:
        super().__init__(costs)
        self.threshold = threshold

    # -- the shared slow path --------------------------------------------

    def _semantic_lookup(self, router: "Router", name: str, exclude=None):
        """Encode the name and compare it against every FIB entry.

        Returns ``(canonical, score, cpu_ms)``; ``canonical`` is ``None`` when
        nothing cleared the threshold.
        """
        index = router.name_index
        cpu_ms = self.costs.semantic_resolve_ms(len(index) if index else 0)
        self.encoder_runs += 1
        router.encoder_runs += 1
        if index is None or len(index) == 0:
            return None, 0.0, cpu_ms
        query = index.backend.encode([name])[0]
        match = index.best_match(query, self.threshold, exclude=exclude)
        if match is None:
            return None, 0.0, cpu_ms
        canonical, score = match
        return canonical, score, cpu_ms

    def _face_for(self, router: "Router", canonical: str) -> Optional[str]:
        entry = router.fib.get(canonical)
        return entry.face if entry else None

    def _tagged(self, router: "Router", interest: "Interest"):
        """Forward on a prefix an upstream router already resolved.

        Introduced by SAF so that the semantic step runs once per Interest
        rather than once per hop. Inherited by every semantic strategy here.
        """
        from ..network import Resolution

        if interest.tag is None:
            return None
        entry = router.fib.get(interest.tag.resolved_prefix)
        if entry is None:
            return None
        self.tag_hits += 1
        return Resolution(
            outcome=OUTCOME_TAGGED, canonical=entry.prefix, face=entry.face,
            score=interest.tag.score, cpu_ms=self.costs.tag_match_ms,
        )

    def resolve(self, router: "Router", interest: "Interest") -> "Resolution":
        from ..network import Resolution

        tagged = self._tagged(router, interest)
        if tagged is not None:
            return tagged

        canonical, score, cpu_ms = self._semantic_lookup(router, interest.name)
        if canonical is None:
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=cpu_ms)
        face = self._face_for(router, canonical)
        if face is None:
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=cpu_ms)
        return Resolution(
            outcome=OUTCOME_SEMANTIC, canonical=canonical, face=face,
            score=score, cpu_ms=cpu_ms,
            tag=SemanticTag(canonical, score, router.id),
        )


class SafWithEs(Saf):
    """SAF with its Embedding Store, cached on resolution as published."""

    name = "saf+es"

    def resolve(self, router: "Router", interest: "Interest") -> "Resolution":
        from ..network import Resolution

        tagged = self._tagged(router, interest)
        if tagged is not None:
            return tagged

        cached = router.es.lookup(interest.name)
        if cached is not None:
            self.es_hits += 1
            return Resolution(
                outcome=OUTCOME_ES_HIT, canonical=cached.canonical, face=cached.face,
                score=cached.score, cpu_ms=self.costs.es_lookup_ms,
                tag=SemanticTag(cached.canonical, cached.score, router.id),
            )

        canonical, score, cpu_ms = self._semantic_lookup(router, interest.name)
        cpu_ms += self.costs.es_lookup_ms
        if canonical is None:
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=cpu_ms)
        face = self._face_for(router, canonical)
        if face is None:
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=cpu_ms)

        # Cached immediately, without waiting to see whether the route works.
        # This is what the paper specifies, and it is why SAF+ES keeps serving a
        # wrong match once it has made one.
        router.es.store(
            EsEntry(
                variant=interest.name, canonical=canonical, face=face,
                score=score, confirmed=True, learned_at=router.sim.now,
            )
        )
        return Resolution(
            outcome=OUTCOME_SEMANTIC, canonical=canonical, face=face,
            score=score, cpu_ms=cpu_ms,
            tag=SemanticTag(canonical, score, router.id),
        )


class GsNdn(Saf):
    """Gossip-based Semantic NDN: resolve at the edge, verify, then share."""

    name = "gs-ndn"

    def __init__(
        self,
        threshold: float = 0.7,
        costs: Optional[CostModel] = None,
        *,
        verify: bool = True,
        gossip: bool = True,
        reason_aware: bool = False,
        robust: bool = False,
        verify_imported: bool = True,
        trust_floor: float = DEFAULT_FLOOR,
        peer_share: float = DEFAULT_PEER_SHARE,
    ) -> None:
        super().__init__(threshold, costs)
        #: Retract mappings a Nack disproves, and only gossip proven ones.
        self.verify = verify
        #: Attribute every retraction to whoever supplied the claim, and stop
        #: believing peers that keep being wrong. Off by default: every result
        #: published before this existed was measured with it off.
        self.robust = robust
        #: Check a peer's mapping against a producer the first time this router
        #: uses it. See :meth:`_pending_for_imported` for what its absence cost.
        self.verify_imported = verify_imported
        self.imported_confirmed = 0
        self.imported_retracted = 0
        self.reputations: Dict[str, ReputationTable] = {}
        self._reputation_defaults = dict(
            floor=trust_floor, confidence=0.9, peer_share=peer_share
        )
        #: Read the producer's refusal reason instead of treating every refusal
        #: as a refutation. Off by default, because every result published
        #: before this existed was measured with it off.
        self.reason_aware = reason_aware
        self.wording_refusals = 0
        #: Accept mappings other routers have proven.  Off for the ablation.
        self.gossip = gossip
        self.tag_misses = 0

        #: Per router, per wording, the canonical prefixes that have already
        #: been tried and refused.  An encoder is deterministic, so re-running
        #: it after a failure returns the same wrong answer; excluding what has
        #: been disproved is what lets a second attempt find a different -- and
        #: usually the right -- producer.
        self.refuted: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.reresolutions = 0
        self.recoveries = 0

    def resolve(self, router: "Router", interest: "Interest") -> "Resolution":
        from ..network import Resolution

        # 1. An upstream edge router already resolved this (SAF's mechanism).
        tagged = self._tagged(router, interest)
        if tagged is not None:
            return tagged
        if interest.tag is not None:
            self.tag_misses += 1

        # 2. A mapping this router already holds, whether it learned it itself
        #    or was taught it by a neighbour.
        cached = router.es.lookup(interest.name)
        if cached is not None and (self.gossip or cached.source == "local"):
            outcome = OUTCOME_ES_HIT if cached.source == "local" else OUTCOME_GOSSIP_HIT
            if cached.source == "local":
                self.es_hits += 1
            else:
                self.gossip_hits += 1
            return Resolution(
                outcome=outcome, canonical=cached.canonical, face=cached.face,
                score=cached.score, cpu_ms=self.costs.es_lookup_ms,
                tag=SemanticTag(cached.canonical, cached.score, router.id),
                learn=self._pending_for_imported(cached),
            )

        # 3. Nothing known, so this router pays to find out. In practice that is
        #    always the ingress edge router: anything further along matched the
        #    tag in step 1 and never reached here.
        refuted = self.refuted[router.id].get(interest.name) if self.verify else None
        if refuted:
            self.reresolutions += 1
        canonical, score, cpu_ms = self._semantic_lookup(
            router, interest.name, exclude=refuted
        )
        cpu_ms += self.costs.es_lookup_ms
        if canonical is None:
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=cpu_ms)
        face = self._face_for(router, canonical)
        if face is None:
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=cpu_ms)
        if refuted:
            self.recoveries += 1

        mapping = PendingMapping(
            variant=interest.name, canonical=canonical, face=face, score=score
        )
        # Cached straight away so repeats of this wording skip the encoder, but
        # flagged unconfirmed: it is not gossiped, and a Nack will remove it.
        entry = EsEntry(
            variant=interest.name, canonical=canonical, face=face, score=score,
            confirmed=not self.verify, learned_at=router.sim.now, source="local",
        )
        router.es.store(entry)
        if not self.verify and self.gossip and router.gossip_protocol is not None:
            # Verification off: share the guess straight away, unproven. This is
            # the ablation arm, and keeping it on the gossip path is the whole
            # point of it -- otherwise switching verification off would switch
            # sharing off too, and the two effects could not be told apart.
            router.gossip_protocol.on_confirmed(router, entry)
        return Resolution(
            outcome=OUTCOME_SEMANTIC, canonical=canonical, face=face, score=score,
            cpu_ms=cpu_ms, tag=SemanticTag(canonical, score, router.id),
            learn=mapping if self.verify else None,
        )

    def reputation_for(self, router: "Router") -> Optional[ReputationTable]:
        """This router's peer table. ``None`` disables every robust path."""
        if not self.robust:
            return None
        table = self.reputations.get(router.id)
        if table is None:
            table = ReputationTable(**self._reputation_defaults)
            self.reputations[router.id] = table
        return table

    def on_confirmed(self, router: "Router", mapping: PendingMapping, now: float) -> None:
        super().on_confirmed(router, mapping, now)
        # Who told us this, before confirming clears the question. A mapping
        # that came from a neighbour and then worked is that neighbour's credit:
        # the producer just audited it for us, at no cost we were not already
        # paying.
        taught_by = self._taught_by(router, mapping.variant)
        held = router.es.peek(mapping.variant)
        entry = router.es.confirm(mapping.variant, now)
        reputation = self.reputation_for(router)
        if taught_by is not None:
            # An imported mapping this router has now seen work. It is not
            # re-published: it is already travelling from whoever proved it
            # first, and echoing it back would turn one confirmation into a
            # broadcast storm proportional to how popular the wording is.
            if held is not None and not held.locally_verified:
                held.locally_verified = True
                self.imported_confirmed += 1
            if reputation is not None:
                reputation.credit(taught_by)
        # Proven locally, so now it is worth telling the neighbours. This is the
        # only path by which a mapping enters the gossip layer.
        if entry is not None and self.gossip and router.gossip_protocol is not None:
            router.gossip_protocol.on_confirmed(router, entry)

    def _pending_for_imported(self, cached: EsEntry) -> Optional[PendingMapping]:
        """Make a peer's mapping falsifiable the first time this router uses it.

        Without this the verification GS-NDN is built on never runs on anything
        gossip taught us. A cache hit returned no pending mapping, so the PIT
        entry carried nothing to judge, so neither ``on_confirmed`` nor
        ``on_rejected`` fired -- and a mapping a neighbour asserted was never
        checked against a producer no matter how many times it was used.

        On an honest network that is invisible, because the only mappings in
        circulation are ones somebody did verify. Under §9's attacker it is the
        whole game: every poisoned mapping is imported, so none of them was ever
        retracted. Measured directly before this existed -- 608 fabricated
        mappings injected, 897 gossip hits served from them, and
        ``adv_poison_retracted`` exactly zero for the entire run. The claim in
        :mod:`gsndn.adversary` that "the lie survives one round trip per victim"
        described the design and not the code; the lie survived the run.

        **On by default since the defect was found.** GS-NDN's first claim is
        that a mapping is verified before it is trusted; leaving this off would
        mean shipping a system where that claim holds for locally resolved
        mappings and silently fails for every mapping the network shared, which
        is the half the design is actually about.

        ``gs-ndn-unverified-import`` restores the old behaviour and changes
        nothing else, so the cost of the defect stays measurable and every
        number published before the fix can still be reproduced on demand.
        """
        if not (self.verify and self.verify_imported):
            return None
        if cached.source == "local" or cached.locally_verified:
            return None
        return PendingMapping(
            variant=cached.variant, canonical=cached.canonical,
            face=cached.face, score=cached.score,
        )

    @staticmethod
    def _taught_by(router: "Router", variant: str) -> Optional[str]:
        """The neighbour a held mapping came from, or ``None`` if first-hand."""
        entry = router.es.peek(variant)
        if entry is None or entry.source == "local":
            return None
        return entry.source

    def on_rejected(
        self, router: "Router", mapping: PendingMapping, reason: str = "",
    ) -> None:
        super().on_rejected(router, mapping, reason)
        # Attribute the refusal before dropping the entry destroys the evidence
        # of where the claim came from.
        taught_by = self._taught_by(router, mapping.variant)
        reputation = self.reputation_for(router)
        # A refusal means this wording did not get served, so the mapping goes
        # either way: keeping it would send the next copy of this wording to the
        # same refusal. What the reason decides is whether the *route* is also
        # written off.
        router.es.drop(mapping.variant)
        if not self.verify:
            return
        if reputation is not None and not (
            self.reason_aware and reason == REFUSAL_UNKNOWN_WORDING
        ):
            # A wording the producer never declared is not evidence against the
            # peer that taught us the route -- the route was right. Debiting a
            # neighbour for it would make an incomplete declaration look like
            # a compromised peer, which is §16's bias arriving in the defence.
            if taught_by is not None:
                reputation.debit(taught_by, mapping.variant, mapping.canonical)
            else:
                reputation.note_refuted(mapping.variant, mapping.canonical)
        if taught_by is not None:
            self.imported_retracted += 1
        if self.reason_aware and reason == REFUSAL_UNKNOWN_WORDING:
            # The producer publishes this name -- the route was right, and only
            # the phrasing was outside what it declared. Blacklisting the prefix
            # here is what the undifferentiated version gets wrong: the next
            # attempt is then forced to exclude the correct producer and pick a
            # worse one, turning a gap in a vocabulary into a misdelivery.
            self.wording_refusals += 1
            return
        # Either an undifferentiated refusal or a genuine routing error: remember
        # which prefix failed, so the next attempt picks a different one instead
        # of repeating the mistake. This is the signal SAF discards -- it caches
        # on resolution and never learns that the route it chose does not work.
        self.refuted[router.id][mapping.variant].add(mapping.canonical)

    def stats(self) -> dict:
        data = super().stats()
        data.update({
            "tag_misses": self.tag_misses,
            "reresolutions": self.reresolutions,
            "recoveries": self.recoveries,
            "wording_refusals": self.wording_refusals,
            "imported_confirmed": self.imported_confirmed,
            "imported_retracted": self.imported_retracted,
            "refuted_pairs": sum(
                len(v) for per_router in self.refuted.values() for v in per_router.values()
            ),
        })
        if self.reputations:
            # Summed across routers, then averaged for the quantities that are
            # rates rather than counts -- the same shape RiskControlledNdn uses
            # so a sweep reports them comparably.
            merged: Dict[str, float] = {}
            for table in self.reputations.values():
                for key, value in table.stats().items():
                    merged[key] = merged.get(key, 0.0) + float(value)
            count = len(self.reputations)
            for key in ("rep_trust_mean", "rep_trust_min"):
                merged[key] /= count
            data.update(merged)
        return data
