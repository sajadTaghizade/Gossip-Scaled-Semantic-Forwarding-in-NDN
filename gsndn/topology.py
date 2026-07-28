"""Topology builders.

SAF evaluates a single access-edge router, and says so in its conclusion: a full
edge-network topology is left to future work.  That single-router setting is
reproduced here as ``single_edge`` so our numbers can be compared against theirs
directly, but it cannot show anything about sharing knowledge between routers,
because there is only one.

``multi_edge`` is the topology the main argument needs: several edge routers,
each serving its own consumer population, meeting at a core.  Every edge router
sees a different slice of the traffic, so without sharing, each pays its own
encoder cost for every wording it meets -- the cost the network pays scales with
the number of edges.  With sharing it scales with the number of distinct names.
That gap is the experiment.

``grid`` reproduces the dense mesh SEF uses for its energy study, where multiple
next hops exist and choosing between them matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .costs import LinkProfile
from .des import Simulator
from .network import Consumer, Network, Producer, Router


@dataclass
class Topology:
    """A built network, with the handles an experiment needs."""

    network: Network
    edges: List[str] = field(default_factory=list)
    cores: List[str] = field(default_factory=list)
    consumers: List[str] = field(default_factory=list)
    producers: List[str] = field(default_factory=list)
    kind: str = ""

    @property
    def routers(self) -> List[Router]:
        return self.network.routers()

    def consumer_nodes(self) -> List[Consumer]:
        return [self.network.nodes[c] for c in self.consumers]  # type: ignore[misc]

    def edge_of(self, consumer_id: str) -> str:
        consumer = self.network.nodes[consumer_id]
        return consumer.upstream  # type: ignore[attr-defined]

    def summary(self) -> Dict[str, int]:
        return {
            "routers": len(self.routers),
            "edges": len(self.edges),
            "cores": len(self.cores),
            "consumers": len(self.consumers),
            "producers": len(self.producers),
        }


def _partition(names: Sequence[str], parts: int) -> List[List[str]]:
    """Split names round-robin, so each producer serves a spread of categories."""
    buckets: List[List[str]] = [[] for _ in range(max(1, parts))]
    for i, name in enumerate(names):
        buckets[i % len(buckets)].append(name)
    return buckets


def _build_producers(
    network: Network,
    sim: Simulator,
    attach_to: Sequence[str],
    names: Sequence[str],
    profile: LinkProfile,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Attach one producer per attachment point and hand it a slice of names."""
    producer_ids: List[str] = []
    ownership: Dict[str, List[str]] = {}
    for i, (router_id, slice_) in enumerate(zip(attach_to, _partition(names, len(attach_to)))):
        producer_id = f"producer-{i}"
        network.add(Producer(producer_id, sim, slice_, service_ms=profile.producer_service_ms))
        network.connect(
            router_id, producer_id,
            profile.router_to_producer_ms, profile.bandwidth_mbps,
        )
        producer_ids.append(producer_id)
        ownership[producer_id] = list(slice_)
    return producer_ids, ownership


def single_edge(
    sim: Simulator,
    names: Sequence[str],
    *,
    profile: Optional[LinkProfile] = None,
    n_producers: int = 4,
    consumers_per_edge: int = 1,
    cs_capacity: int = 0,
    es_capacity: int = 1024,
    queue_capacity: Optional[int] = None,
) -> Topology:
    """SAF's setting: one access-edge router between clients and providers."""
    profile = profile or LinkProfile()
    network = Network(sim)

    edge = "edge-0"
    network.add(Router(
        edge, sim, role="edge", cs_capacity=cs_capacity,
        es_capacity=es_capacity, queue_capacity=queue_capacity,
    ))

    gateway = "core-0"
    network.add(Router(
        gateway, sim, role="core", cs_capacity=cs_capacity,
        es_capacity=es_capacity, queue_capacity=queue_capacity,
    ))
    network.connect(edge, gateway, profile.router_to_router_ms, profile.bandwidth_mbps)

    consumers = []
    for i in range(consumers_per_edge):
        consumer_id = f"consumer-{i}"
        network.add(Consumer(consumer_id, sim, upstream=edge))
        network.connect(consumer_id, edge, profile.consumer_to_edge_ms, profile.bandwidth_mbps)
        consumers.append(consumer_id)

    producer_ids, ownership = _build_producers(
        network, sim, [gateway] * n_producers, names, profile
    )
    network.install_routes(ownership)

    return Topology(
        network=network, edges=[edge], cores=[gateway],
        consumers=consumers, producers=producer_ids, kind="single_edge",
    )


def multi_edge(
    sim: Simulator,
    names: Sequence[str],
    *,
    n_edges: int = 6,
    profile: Optional[LinkProfile] = None,
    n_producers: int = 4,
    consumers_per_edge: int = 1,
    cs_capacity: int = 0,
    es_capacity: int = 1024,
    queue_capacity: Optional[int] = None,
    core_ring: bool = True,
) -> Topology:
    """Several edge routers around a core: the topology the sharing claim needs.

    The core routers are optionally wired into a ring as well as a star, so
    gossip has more than one path between edges and the protocol is not
    trivially a broadcast through a single hub.
    """
    profile = profile or LinkProfile()
    network = Network(sim)

    core = "core-0"
    network.add(Router(
        core, sim, role="core", cs_capacity=cs_capacity,
        es_capacity=es_capacity, queue_capacity=queue_capacity,
    ))

    edges: List[str] = []
    consumers: List[str] = []
    for e in range(n_edges):
        edge_id = f"edge-{e}"
        network.add(Router(
            edge_id, sim, role="edge", cs_capacity=cs_capacity,
            es_capacity=es_capacity, queue_capacity=queue_capacity,
        ))
        network.connect(edge_id, core, profile.router_to_router_ms, profile.bandwidth_mbps)
        edges.append(edge_id)

        for c in range(consumers_per_edge):
            consumer_id = f"consumer-{e}-{c}"
            network.add(Consumer(consumer_id, sim, upstream=edge_id))
            network.connect(
                consumer_id, edge_id, profile.consumer_to_edge_ms, profile.bandwidth_mbps
            )
            consumers.append(consumer_id)

    if core_ring and n_edges > 2:
        for e in range(n_edges):
            network.connect(
                edges[e], edges[(e + 1) % n_edges],
                profile.router_to_router_ms, profile.bandwidth_mbps,
            )

    producer_ids, ownership = _build_producers(
        network, sim, [core] * n_producers, names, profile
    )
    network.install_routes(ownership)

    return Topology(
        network=network, edges=edges, cores=[core],
        consumers=consumers, producers=producer_ids, kind="multi_edge",
    )


def grid(
    sim: Simulator,
    names: Sequence[str],
    *,
    side: int = 4,
    profile: Optional[LinkProfile] = None,
    n_producers: int = 4,
    cs_capacity: int = 0,
    es_capacity: int = 1024,
    queue_capacity: Optional[int] = None,
) -> Topology:
    """A ``side`` x ``side`` mesh, as in SEF's dense deployments.

    Every interior node has four neighbours, so several next hops make progress
    toward a producer and an energy-aware chooser has something to choose.
    """
    profile = profile or LinkProfile()
    network = Network(sim)

    def node_id(row: int, col: int) -> str:
        return f"r-{row}-{col}"

    for row in range(side):
        for col in range(side):
            role = "edge" if (row == 0 or col == 0) else "core"
            network.add(Router(
                node_id(row, col), sim, role=role, cs_capacity=cs_capacity,
                es_capacity=es_capacity, queue_capacity=queue_capacity,
            ))

    for row in range(side):
        for col in range(side):
            if col + 1 < side:
                network.connect(
                    node_id(row, col), node_id(row, col + 1),
                    profile.router_to_router_ms, profile.bandwidth_mbps,
                )
            if row + 1 < side:
                network.connect(
                    node_id(row, col), node_id(row + 1, col),
                    profile.router_to_router_ms, profile.bandwidth_mbps,
                )

    edges = [node_id(0, c) for c in range(side)]
    consumers = []
    for c, edge_id in enumerate(edges):
        consumer_id = f"consumer-{c}"
        network.add(Consumer(consumer_id, sim, upstream=edge_id))
        network.connect(consumer_id, edge_id, profile.consumer_to_edge_ms, profile.bandwidth_mbps)
        consumers.append(consumer_id)

    corner = node_id(side - 1, side - 1)
    producer_ids, ownership = _build_producers(
        network, sim, [corner] * n_producers, names, profile
    )
    network.install_routes(ownership)

    return Topology(
        network=network, edges=edges,
        cores=[n.id for n in network.routers() if n.role == "core"],
        consumers=consumers, producers=producer_ids, kind="grid",
    )


BUILDERS = {
    "single_edge": single_edge,
    "multi_edge": multi_edge,
    "grid": grid,
}


def build(kind: str, sim: Simulator, names: Sequence[str], **kwargs) -> Topology:
    try:
        builder = BUILDERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown topology {kind!r}; expected one of {sorted(BUILDERS)}"
        ) from None
    return builder(sim, names, **kwargs)
