"""Routers, links, producers, consumers, and how a packet moves between them.

The router here is deliberately ordinary NDN: check the Content Store, check the
PIT, check the FIB, forward or reject.  Everything a forwarding strategy is
allowed to change happens at one point -- the FIB miss -- through the
``ForwardingStrategy`` interface in :mod:`gsndn.strategies`.  Keeping mechanism
and policy apart is what lets Vanilla NDN, SAF, SAF+ES, SEF and GS-NDN run over
literally the same node code and the same workload, so a difference in results
is a difference in policy rather than in bookkeeping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .des import ServiceQueue, Simulator
from .packets import (
    Data,
    Interest,
    Nack,
    OUTCOME_MISDELIVERED,
    OUTCOME_NACK,
    OUTCOME_TAGGED,
    OUTCOME_TIMEOUT,
    SemanticTag,
)
from .tables import ContentStore, EmbeddingStore, Fib, PendingMapping, Pit

if TYPE_CHECKING:  # pragma: no cover
    from .admission import AdmissionPolicy

#: Cost of the table lookups every Interest pays, regardless of strategy.
#: A trie or TCAM lookup is tens of nanoseconds; at millisecond resolution this
#: is deliberately small but not zero, so that a "free" fast path still consumes
#: server time and can still queue under extreme load.
FIB_LOOKUP_MS = 0.01
CS_LOOKUP_MS = 0.005
PIT_LOOKUP_MS = 0.005
BASE_LOOKUP_MS = FIB_LOOKUP_MS + CS_LOOKUP_MS + PIT_LOOKUP_MS


@dataclass
class Face:
    """A point-to-point link to one neighbour."""

    peer: str
    delay_ms: float
    bandwidth_mbps: float = 100.0

    def transmit_ms(self, wire_bytes: int) -> float:
        serialization = (wire_bytes * 8.0) / (self.bandwidth_mbps * 1000.0)
        return self.delay_ms + serialization


@dataclass
class Resolution:
    """What a strategy decided to do with an Interest that missed the FIB."""

    outcome: str
    canonical: Optional[str] = None
    face: Optional[str] = None
    score: float = 0.0
    cpu_ms: float = 0.0
    tag: Optional[SemanticTag] = None
    learn: Optional[PendingMapping] = None

    @property
    def forwards(self) -> bool:
        return self.face is not None


class Node:
    """Anything attached to the topology."""

    def __init__(self, node_id: str, sim: Simulator) -> None:
        self.id = node_id
        self.sim = sim
        self.faces: Dict[str, Face] = {}
        self.network: Optional["Network"] = None

    def add_face(self, peer: str, delay_ms: float, bandwidth_mbps: float = 100.0) -> Face:
        face = Face(peer=peer, delay_ms=delay_ms, bandwidth_mbps=bandwidth_mbps)
        self.faces[peer] = face
        return face

    # Subclasses override the handlers they care about.
    def on_interest(self, interest: Interest, in_face: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def on_data(self, data: Data, in_face: str) -> None:  # pragma: no cover
        pass

    def on_nack(self, nack: Nack, in_face: str) -> None:  # pragma: no cover
        pass


class Router(Node):
    """An NDN forwarder with a single packet-processing server."""

    def __init__(
        self,
        node_id: str,
        sim: Simulator,
        *,
        role: str = "core",
        cs_capacity: int = 0,
        es_capacity: int = 1024,
        queue_capacity: Optional[int] = None,
    ) -> None:
        super().__init__(node_id, sim)
        self.role = role
        self.fib = Fib()
        self.pit = Pit()
        self.cs = ContentStore(cs_capacity)
        self.es = EmbeddingStore(es_capacity)
        self.cpu = ServiceQueue(sim, name=f"{node_id}:cpu", capacity=queue_capacity)

        self.strategy = None            # gsndn.strategies.ForwardingStrategy
        self.name_index = None          # gsndn.embeddings.NameIndex, set at build time
        self.peers: List[str] = []      # neighbouring routers, for gossip
        self.gossip = None              # gsndn.gossip.GossipAgent
        self.gossip_protocol = None     # gsndn.gossip.GossipProtocol

        #: Remaining battery as a fraction, read by the energy-aware baseline.
        self.energy_fraction = 1.0

        self.interests_seen = 0
        self.interests_forwarded = 0
        self.interests_rejected = 0
        self.encoder_runs = 0
        self.bytes_tx = 0
        self.bytes_rx = 0

    # -- ingress ---------------------------------------------------------

    def on_interest(self, interest: Interest, in_face: str) -> None:
        self.interests_seen += 1
        self.bytes_rx += interest.wire_bytes
        self.cpu.submit(
            lambda: self._serve_interest(interest, in_face),
            on_drop=lambda: self._reject(interest, in_face, "queue-overflow"),
        )

    def _serve_interest(self, interest: Interest, in_face: str) -> Tuple[float, Callable[[], None]]:
        """Run the forwarding pipeline; return its cost and its effect."""
        interest.hop_count += 1

        # Content Store: exact match only, by architectural rule.
        cached = self.cs.get(interest.lookup_name)
        if cached is not None:
            return BASE_LOOKUP_MS, lambda: self._return_data(cached, in_face, interest)

        # PIT: aggregate duplicate outstanding requests.
        entry, aggregated = self.pit.insert(
            interest.lookup_name, in_face, interest.id, self.sim.now
        )
        if aggregated:
            return BASE_LOOKUP_MS, lambda: None

        # FIB: exact, then longest-prefix, then hand over to the strategy.
        target = interest.lookup_name
        hit = self.fib.exact(target) or self.fib.longest_prefix(target)
        if hit is not None:
            entry.resolved_prefix = hit.prefix
            face = (
                self.strategy.select_face(self, interest, hit)
                if self.strategy is not None
                else hit.face
            )
            resolution = Resolution(
                outcome="exact", canonical=hit.prefix, face=face,
                score=1.0, cpu_ms=BASE_LOOKUP_MS,
            )
        else:
            assert self.strategy is not None, f"router {self.id} has no strategy"
            resolution = self.strategy.resolve(self, interest)
            resolution.cpu_ms += BASE_LOOKUP_MS

        return resolution.cpu_ms, lambda: self._apply(resolution, interest, entry, in_face)

    def _apply(self, resolution: Resolution, interest: Interest, pit_entry, in_face: str) -> None:
        if not resolution.forwards:
            self.pit.satisfy(pit_entry.name)
            self._reject(interest, in_face, resolution.outcome)
            return

        if interest.resolution is None or resolution.outcome != OUTCOME_TAGGED:
            interest.resolution = resolution.outcome
        pit_entry.out_face = resolution.face
        pit_entry.resolved_prefix = resolution.canonical
        pit_entry.pending_learn = resolution.learn
        if resolution.tag is not None:
            interest.tag = resolution.tag
        self.interests_forwarded += 1
        self._send(interest, resolution.face, resolution.outcome)

    def _reject(self, interest: Interest, in_face: str, reason: str) -> None:
        self.interests_rejected += 1
        nack = Nack(
            name=interest.name, interest_id=interest.id,
            reason=reason, origin=interest.origin,
        )
        self._send(nack, in_face, OUTCOME_NACK)

    # -- egress ----------------------------------------------------------

    def _send(self, packet, face_id: str, label: str = "") -> None:
        face = self.faces.get(face_id)
        if face is None or self.network is None:
            return
        self.bytes_tx += packet.wire_bytes
        self.network.deliver(self, face, packet, label)

    def _return_data(self, data: Data, face_id: str, interest: Interest) -> None:
        self._send(data, face_id)

    # -- data path -------------------------------------------------------

    def on_data(self, data: Data, in_face: str) -> None:
        self.bytes_rx += data.wire_bytes
        self.cpu.submit(lambda: (BASE_LOOKUP_MS, lambda: self._deliver_data(data, in_face)))

    def _deliver_data(self, data: Data, in_face: str) -> None:
        entries = []
        direct = self.pit.satisfy(data.name)
        if direct is not None:
            entries.append(direct)
        # Also every entry keyed on a client's own wording that this router
        # resolved to the same prefix. Several wordings routinely resolve to one
        # service, and upstream aggregates them into a single Interest, so one
        # Data legitimately answers all of them. Satisfying only the first would
        # leave the rest to time out -- a failure the model would have invented.
        entries.extend(self._entries_resolving_to(data.name))
        if not entries:
            return

        self.cs.put(data.name, data)
        for entry in entries:
            if entry.pending_learn is not None and self.strategy is not None:
                # The guess worked: Data came back. Only now is the mapping
                # worth keeping, and only now worth telling anyone else about.
                self.strategy.on_confirmed(self, entry.pending_learn, self.sim.now)
            # Stamp each copy with exactly the Interests its own entry folded.
            outgoing = replace(data, satisfied_ids=tuple(entry.interest_ids))
            for face_id in entry.in_faces:
                self._send(outgoing, face_id)

    def _entries_resolving_to(self, canonical: str) -> List:
        """Every outstanding entry this router pointed at ``canonical``."""
        matched = [
            name for name, entry in self.pit._entries.items()  # noqa: SLF001 - internal by design
            if entry.resolved_prefix == canonical
        ]
        return [e for e in (self.pit.satisfy(name) for name in matched) if e is not None]

    def on_nack(self, nack: Nack, in_face: str) -> None:
        self.bytes_rx += nack.wire_bytes
        entries = []
        direct = self.pit.satisfy(nack.name)
        if direct is not None:
            entries.append(direct)
        entries.extend(self._entries_resolving_to(nack.name))
        for entry in entries:
            if entry.pending_learn is not None and self.strategy is not None:
                # The guess was wrong. Forget it rather than serving a route
                # that demonstrably does not carry this name.
                self.strategy.on_rejected(self, entry.pending_learn)
            outgoing = replace(nack, satisfied_ids=tuple(entry.interest_ids))
            for face_id in entry.in_faces:
                self._send(outgoing, face_id)


class Producer(Node):
    """Serves the names it owns, and only those.

    Admission is decided by :class:`gsndn.admission.AdmissionPolicy`, built from
    the schemas this producer itself declares. It never consults the catalog's
    ground truth, so the feedback the network learns from is local knowledge --
    incomplete and occasionally wrong -- rather than an oracle.
    """

    def __init__(
        self,
        node_id: str,
        sim: Simulator,
        names: Sequence[str],
        service_ms: float = 1.0,
        policy: Optional["AdmissionPolicy"] = None,
    ) -> None:
        super().__init__(node_id, sim)
        self.names: Set[str] = set(names)
        self.service_ms = service_ms
        self.policy = policy
        self.served = 0
        self.refused = 0
        self.available = True

    def on_interest(self, interest: Interest, in_face: str) -> None:
        """Serve the name, or refuse it.

        Two checks, not one.  The first is ordinary NDN: is the name being asked
        for one this producer publishes?  The second exists because a
        semantically resolved Interest arrives under a *rewritten* name -- SAF
        prepends the resolved prefix and keeps the client's original wording --
        so the producer can also look at what the client actually asked for and
        decide whether that request is one of its services.

        That second check runs against the schema this producer declared for
        itself: the terms it answers to and the instance it is attached to. It
        is deliberately fallible. A producer that declared only some of the ways
        clients word its service will refuse requests that were meant for it,
        and the network has to learn from that anyway.
        """
        target = interest.lookup_name
        if not self.available:
            self._refuse(interest, in_face, "producer-unavailable")
            return

        if target not in self.names:
            self._refuse(interest, in_face, OUTCOME_MISDELIVERED)
            return

        admitted = (
            self.policy.admits(target, interest.name) if self.policy is not None else True
        )
        if admitted:
            self.served += 1
            data = Data(
                name=target, interest_id=interest.id,
                produced_at=self.sim.now + self.service_ms, producer=self.id,
            )
            self.sim.schedule(self.service_ms, self._reply, data, in_face)
        else:
            # A resolution this producer does not recognise ends here. In a real
            # deployment this is how a confident wrong answer is caught -- and
            # also, sometimes, how a correct one is wrongly rejected.
            self._refuse(interest, in_face, OUTCOME_MISDELIVERED)

    def _refuse(self, interest: Interest, in_face: str, reason: str) -> None:
        self.refused += 1
        nack = Nack(
            name=interest.lookup_name, interest_id=interest.id,
            reason=reason, origin=interest.origin,
        )
        self.sim.schedule(self.service_ms, self._reply, nack, in_face)

    def _reply(self, packet, face_id: str) -> None:
        face = self.faces.get(face_id)
        if face is not None and self.network is not None:
            self.network.deliver(self, face, packet, "")


class Consumer(Node):
    """Emits Interests and records how each one ended."""

    def __init__(self, node_id: str, sim: Simulator, upstream: str) -> None:
        super().__init__(node_id, sim)
        self.upstream = upstream
        self.pending: Dict[int, Interest] = {}
        self.on_complete: Optional[Callable[[Interest, str, float], None]] = None

    def send(self, interest: Interest) -> None:
        self.pending[interest.id] = interest
        face = self.faces[self.upstream]
        if self.network is not None:
            self.network.deliver(self, face, interest, "")

    def on_data(self, data: Data, in_face: str) -> None:
        for interest in self._take(data.interest_id, data.name, data.satisfied_ids):
            correct = interest.expected is not None and data.name == interest.expected
            outcome = "resolved" if correct else OUTCOME_MISDELIVERED
            if self.on_complete is not None:
                self.on_complete(interest, outcome, self.sim.now - interest.created_at)

    def on_nack(self, nack: Nack, in_face: str) -> None:
        for interest in self._take(nack.interest_id, nack.name, nack.satisfied_ids):
            if self.on_complete is not None:
                self.on_complete(interest, OUTCOME_NACK, self.sim.now - interest.created_at)

    def _take(self, interest_id: int, name: str, ids: Sequence[int] = ()) -> List[Interest]:
        """Every outstanding Interest this reply answers.

        A router aggregates duplicate Interests into one PIT entry and returns a
        single Data, so one reply can settle several of this consumer's
        requests -- but only the ones that entry folded together, which the
        reply names explicitly. Falling back to matching on the name would let
        an early Data settle a request issued later, reporting a latency that
        request never experienced.
        """
        settled: List[Interest] = []
        for candidate_id in (interest_id, *ids):
            interest = self.pending.pop(candidate_id, None)
            if interest is not None:
                settled.append(interest)
        return settled

    def expire(self, deadline_ms: float) -> List[Interest]:
        """Fail every Interest older than the timeout, so nothing hangs."""
        stale = [
            i for i in self.pending.values()
            if self.sim.now - i.created_at > deadline_ms
        ]
        for interest in stale:
            self.pending.pop(interest.id, None)
            if self.on_complete is not None:
                self.on_complete(interest, OUTCOME_TIMEOUT, self.sim.now - interest.created_at)
        return stale


class Network:
    """Holds the nodes and moves packets across links."""

    def __init__(self, sim: Simulator) -> None:
        self.sim = sim
        self.nodes: Dict[str, Node] = {}
        self.link_events = 0
        self.link_bytes = 0

    def add(self, node: Node) -> Node:
        node.network = self
        self.nodes[node.id] = node
        return node

    def connect(self, a: str, b: str, delay_ms: float, bandwidth_mbps: float = 100.0) -> None:
        self.nodes[a].add_face(b, delay_ms, bandwidth_mbps)
        self.nodes[b].add_face(a, delay_ms, bandwidth_mbps)
        for node_id, peer_id in ((a, b), (b, a)):
            node = self.nodes[node_id]
            if isinstance(node, Router) and isinstance(self.nodes[peer_id], Router):
                node.peers.append(peer_id)

    def deliver(self, sender: Node, face: Face, packet, label: str = "") -> None:
        target = self.nodes.get(face.peer)
        if target is None:
            return
        self.link_events += 1
        self.link_bytes += packet.wire_bytes
        delay = face.transmit_ms(packet.wire_bytes)
        self.sim.schedule(delay, self._arrive, target, packet, sender.id)

    def _arrive(self, target: Node, packet, from_id: str) -> None:
        if isinstance(packet, Interest):
            target.on_interest(packet, from_id)
        elif isinstance(packet, Data):
            target.on_data(packet, from_id)
        elif isinstance(packet, Nack):
            target.on_nack(packet, from_id)

    # -- routing ---------------------------------------------------------

    def routers(self) -> List[Router]:
        return [n for n in self.nodes.values() if isinstance(n, Router)]

    def install_routes(self, producers: Dict[str, Sequence[str]]) -> None:
        """Populate every router's FIB by shortest path to each producer.

        Stands in for a name-based routing protocol like NLSR: the point of the
        study is what happens when a name is *absent* from the FIB, so the
        routes that do exist are simply assumed correct, exactly as SAF assumes.
        """
        for producer_id, names in producers.items():
            distances = self._bfs(producer_id)
            for router in self.routers():
                hop = self._next_hop(router.id, producer_id, distances)
                if hop is None:
                    continue
                for name in names:
                    router.fib.add(name, hop, cost=float(distances.get(router.id, 1)))

    def _bfs(self, source: str) -> Dict[str, int]:
        distances = {source: 0}
        frontier = deque([source])
        while frontier:
            current = frontier.popleft()
            for peer in self.nodes[current].faces:
                if peer not in distances:
                    distances[peer] = distances[current] + 1
                    frontier.append(peer)
        return distances

    def _next_hop(self, origin: str, target: str, distances: Dict[str, int]) -> Optional[str]:
        if origin == target or origin not in distances:
            return None
        best, best_distance = None, distances[origin]
        for peer in self.nodes[origin].faces:
            peer_distance = distances.get(peer)
            if peer_distance is not None and peer_distance < best_distance:
                best, best_distance = peer, peer_distance
        return best
