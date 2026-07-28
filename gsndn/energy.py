"""Energy accounting, on the radio model SEF uses.

Askar et al. (CMC 2024) charge each transmission and reception with the standard
first-order radio model:

    TX(i -> j) = e_elec * k + e_amp * d^2 * k
    RX(i)      = e_elec * k

with ``e_elec = 50 nJ/bit``, ``e_amp = 100 pJ/bit/m^2``, ``k`` the packet size in
bits and ``d`` the distance in metres.  Reusing their constants is what makes an
energy comparison against SEF meaningful rather than a separate scale.

One term has to be added that a purely wireless study does not need: running a
sentence encoder costs energy too.  A strategy that saves radio transmissions by
running a transformer on every Interest has not necessarily saved anything, and
the whole argument for resolving once at the edge and sharing the result rests
on being able to show that it has.  Encoder energy is charged per inference at a
measured CPU power draw, so both sides of that trade appear in the same budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

#: Energy to run transmitter or receiver circuitry, joules per bit.
E_ELEC_J_PER_BIT = 50e-9
#: Transmit amplifier energy, joules per bit per square metre.
E_AMP_J_PER_BIT_M2 = 100e-12

#: CPU power while running the encoder, in watts.  A single busy core on the
#: class of edge machine SAF benchmarks on; overridable per experiment because
#: it is a platform figure rather than a measured property of this model.
CPU_ACTIVE_W = 15.0
#: Idle draw attributed to keeping the forwarder resident.
CPU_IDLE_W = 2.0


@dataclass
class EnergyModel:
    """Charges radio and compute energy against a shared budget."""

    e_elec: float = E_ELEC_J_PER_BIT
    e_amp: float = E_AMP_J_PER_BIT_M2
    cpu_active_w: float = CPU_ACTIVE_W
    cpu_idle_w: float = CPU_IDLE_W
    default_distance_m: float = 50.0

    def transmit_j(self, wire_bytes: int, distance_m: Optional[float] = None) -> float:
        bits = wire_bytes * 8
        distance = self.default_distance_m if distance_m is None else distance_m
        return self.e_elec * bits + self.e_amp * (distance ** 2) * bits

    def receive_j(self, wire_bytes: int) -> float:
        return self.e_elec * wire_bytes * 8

    def compute_j(self, busy_ms: float) -> float:
        """Energy for ``busy_ms`` of active processing."""
        return self.cpu_active_w * (busy_ms / 1000.0)

    def idle_j(self, elapsed_ms: float, busy_ms: float) -> float:
        return self.cpu_idle_w * (max(0.0, elapsed_ms - busy_ms) / 1000.0)


@dataclass
class EnergyLedger:
    """Per-node and whole-network energy totals."""

    model: EnergyModel = field(default_factory=EnergyModel)
    tx_j: float = 0.0
    rx_j: float = 0.0
    compute_j: float = 0.0
    idle_j: float = 0.0
    per_node: Dict[str, float] = field(default_factory=dict)

    def charge_node(
        self,
        node_id: str,
        *,
        bytes_tx: int,
        bytes_rx: int,
        busy_ms: float,
        elapsed_ms: float,
        distance_m: Optional[float] = None,
    ) -> float:
        tx = self.model.transmit_j(bytes_tx, distance_m)
        rx = self.model.receive_j(bytes_rx)
        compute = self.model.compute_j(busy_ms)
        idle = self.model.idle_j(elapsed_ms, busy_ms)
        total = tx + rx + compute + idle

        self.tx_j += tx
        self.rx_j += rx
        self.compute_j += compute
        self.idle_j += idle
        self.per_node[node_id] = self.per_node.get(node_id, 0.0) + total
        return total

    @property
    def total_j(self) -> float:
        return self.tx_j + self.rx_j + self.compute_j + self.idle_j

    @property
    def radio_j(self) -> float:
        return self.tx_j + self.rx_j

    def first_node_exhausted_j(self) -> float:
        """Energy drawn by the hardest-worked node.

        SEF defines network lifetime as the time until the first node runs flat,
        so the maximum per-node draw is the quantity that determines it -- a
        strategy with a good total but one overloaded node is worse, not better.
        """
        return max(self.per_node.values(), default=0.0)

    def report(self, interests: int) -> Dict[str, float]:
        return {
            "energy_total_j": self.total_j,
            "energy_radio_j": self.radio_j,
            "energy_compute_j": self.compute_j,
            "energy_idle_j": self.idle_j,
            "energy_per_interest_mj": (
                self.total_j / interests * 1000.0 if interests else 0.0
            ),
            "energy_compute_share": (
                self.compute_j / self.total_j if self.total_j else 0.0
            ),
            "energy_max_node_j": self.first_node_exhausted_j(),
        }


def charge_network(network, ledger: EnergyLedger, elapsed_ms: float) -> EnergyLedger:
    """Settle the energy budget for every router once a run has finished."""
    from .network import Router

    for node in network.nodes.values():
        if not isinstance(node, Router):
            continue
        ledger.charge_node(
            node.id,
            bytes_tx=node.bytes_tx,
            bytes_rx=node.bytes_rx,
            busy_ms=node.cpu.busy_time,
            elapsed_ms=elapsed_ms,
        )
    return ledger
