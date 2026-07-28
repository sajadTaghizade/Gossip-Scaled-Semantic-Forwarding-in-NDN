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
    OUTCOME_ES_HIT,
    OUTCOME_GOSSIP_HIT,
    OUTCOME_NACK,
    OUTCOME_SEMANTIC,
    OUTCOME_TAGGED,
    SemanticTag,
)
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
        edge_only: bool = True,
        verify: bool = True,
        gossip: bool = True,
    ) -> None:
        super().__init__(threshold, costs)
        #: Restrict encoder runs to ingress edge routers; core routers forward
        #: on the tag the edge attached.
        self.edge_only = edge_only
        #: Retract mappings a Nack disproves, and only gossip proven ones.
        self.verify = verify
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
            )

        # 3. Nothing known. Only an edge router pays to find out.
        if self.edge_only and router.role != "edge":
            return Resolution(outcome=OUTCOME_NACK, cpu_ms=self.costs.es_lookup_ms)

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
        router.es.store(
            EsEntry(
                variant=interest.name, canonical=canonical, face=face, score=score,
                confirmed=not self.verify, learned_at=router.sim.now, source="local",
            )
        )
        return Resolution(
            outcome=OUTCOME_SEMANTIC, canonical=canonical, face=face, score=score,
            cpu_ms=cpu_ms, tag=SemanticTag(canonical, score, router.id),
            learn=mapping if self.verify else None,
        )

    def on_confirmed(self, router: "Router", mapping: PendingMapping, now: float) -> None:
        super().on_confirmed(router, mapping, now)
        entry = router.es.confirm(mapping.variant, now)
        # Proven locally, so now it is worth telling the neighbours. This is the
        # only path by which a mapping enters the gossip layer.
        if entry is not None and self.gossip and router.gossip_protocol is not None:
            router.gossip_protocol.on_confirmed(router, entry)

    def on_rejected(self, router: "Router", mapping: PendingMapping) -> None:
        super().on_rejected(router, mapping)
        # The producer refused it, so this route does not serve this name. Drop
        # the mapping and remember which prefix failed, so the next attempt
        # picks a different one instead of repeating the mistake. This is the
        # signal SAF discards: it caches on resolution and never learns that the
        # route it chose does not work.
        router.es.drop(mapping.variant)
        if self.verify:
            self.refuted[router.id][mapping.variant].add(mapping.canonical)

    def stats(self) -> dict:
        data = super().stats()
        data.update({
            "tag_misses": self.tag_misses,
            "reresolutions": self.reresolutions,
            "recoveries": self.recoveries,
            "refuted_pairs": sum(
                len(v) for per_router in self.refuted.values() for v in per_router.values()
            ),
        })
        return data
