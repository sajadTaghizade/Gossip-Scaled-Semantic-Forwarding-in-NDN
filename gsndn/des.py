"""A small discrete-event simulation kernel.

Everything downstream depends on one property: simulated time advances only
because an event says so.  Latency is therefore something the model *produces*
-- link delays, queueing behind a busy router, the cost of running an encoder --
rather than a constant read out of a configuration file.  That distinction is
the whole reason this kernel exists: a strategy that adds work to the forwarding
path has to pay for it in queueing delay, which is exactly the effect SAF
measures when it sweeps the Interest rate from 30 to 300 per second.

Time is in milliseconds throughout.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterator, List, Optional, Tuple

from collections import deque


@dataclass(order=True)
class _Event:
    time: float
    seq: int
    callback: Callable[..., None] = field(compare=False)
    args: Tuple[Any, ...] = field(compare=False, default=())


class Simulator:
    """Event-driven clock.

    Events at equal timestamps fire in scheduling order, so a run is fully
    determined by its seed and configuration.
    """

    def __init__(self) -> None:
        self.now: float = 0.0
        self._queue: List[_Event] = []
        self._counter: Iterator[int] = itertools.count()
        self._processed: int = 0
        self._stopped: bool = False

    def schedule(self, delay: float, callback: Callable[..., None], *args: Any) -> None:
        if delay < 0.0:
            raise ValueError(f"cannot schedule {delay} ms into the past")
        heapq.heappush(
            self._queue,
            _Event(self.now + delay, next(self._counter), callback, args),
        )

    def schedule_at(self, when: float, callback: Callable[..., None], *args: Any) -> None:
        self.schedule(when - self.now, callback, *args)

    def stop(self) -> None:
        self._stopped = True

    def run(self, until: Optional[float] = None, max_events: Optional[int] = None) -> None:
        """Drain the queue, optionally bounded by simulated time or event count."""
        while self._queue and not self._stopped:
            if max_events is not None and self._processed >= max_events:
                break
            if until is not None and self._queue[0].time > until:
                self.now = until
                break
            event = heapq.heappop(self._queue)
            self.now = event.time
            self._processed += 1
            event.callback(*event.args)

    @property
    def events_processed(self) -> int:
        return self._processed

    @property
    def pending(self) -> int:
        return len(self._queue)


class ServiceQueue:
    """A single-server FIFO queue: one router's packet-processing CPU.

    Each job holds the server for its own service time, so when the offered load
    approaches the service rate, waiting time grows -- the bottleneck SAF
    observes past roughly 210 Interests/s with MiniLM-L6.  A strategy that
    replaces an 8 ms embedding with a 0.1 ms cache hit does not merely save
    those 8 ms on that packet; it stops every packet behind it from waiting.
    Modelling this is the reason a strategy's benefit is superlinear under load.
    """

    #: A unit of work: run at service start, returns how long it takes and what
    #: to do once that time has elapsed.
    Work = Callable[[], Tuple[float, Callable[[], None]]]

    def __init__(self, sim: Simulator, name: str = "cpu", capacity: Optional[int] = None):
        self.sim = sim
        self.name = name
        self.capacity = capacity
        self._pending: Deque[Tuple[float, "ServiceQueue.Work"]] = deque()
        self._busy = False

        self.jobs_served = 0
        self.jobs_dropped = 0
        self.busy_time = 0.0
        self.total_wait = 0.0
        self.max_queue_len = 0

    def submit(self, work: "ServiceQueue.Work", on_drop: Optional[Callable[[], None]] = None) -> bool:
        """Queue one job. Returns False when a bounded queue is full.

        The job decides its own duration, but only once it reaches the head of
        the queue.  That ordering matters: whether an Interest costs an encoder
        run or a cache hit depends on what the router has learned by the time it
        is actually served, not by the time it arrived.
        """
        if self.capacity is not None and len(self._pending) >= self.capacity:
            self.jobs_dropped += 1
            if on_drop is not None:
                on_drop()
            return False
        self._pending.append((self.sim.now, work))
        self.max_queue_len = max(self.max_queue_len, len(self._pending))
        if not self._busy:
            self._start_next()
        return True

    def _start_next(self) -> None:
        if not self._pending:
            self._busy = False
            return
        enqueued_at, work = self._pending.popleft()
        self._busy = True
        self.total_wait += self.sim.now - enqueued_at
        duration, finish = work()
        duration = max(0.0, float(duration))
        self.busy_time += duration
        self.sim.schedule(duration, self._complete, finish)

    def _complete(self, finish: Callable[[], None]) -> None:
        self.jobs_served += 1
        finish()
        self._start_next()

    @property
    def queue_len(self) -> int:
        return len(self._pending)

    @property
    def utilization(self) -> float:
        return self.busy_time / self.sim.now if self.sim.now > 0 else 0.0

    @property
    def mean_wait(self) -> float:
        return self.total_wait / self.jobs_served if self.jobs_served else 0.0


class Stopwatch:
    """Accumulates a timing series and reports order statistics.

    Tail latency matters more than the mean here: a strategy that resolves 95%
    of Interests instantly and stalls the rest behind an encoder has a fine
    average and a terrible p99, and the average alone would hide that.
    """

    __slots__ = ("samples",)

    def __init__(self) -> None:
        self.samples: List[float] = []

    def record(self, value: float) -> None:
        self.samples.append(value)

    def __len__(self) -> int:
        return len(self.samples)

    def percentile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * (q / 100.0)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0
