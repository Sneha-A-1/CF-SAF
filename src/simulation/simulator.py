"""
simulator.py
------------
Model
-----
Continuous-time event loop over task arrivals and completions.

  * A fog node holds a load fraction lambda_i in [0,1]. A task occupies
    f = r_cpu / kappa_i of the node for the duration of its execution.
  * Execution time follows Eq. (2):  E = r_cpu / (kappa_i * (1 - lambda_i)),
    evaluated at the load prevailing when service starts.
  * A task is feasible on a node iff residual CPU and memory both admit it,
    i.e. it can start immediately; this is the assumption Lemma 9 relies on.
  * Response time follows Eq. (13):
        L = T_upload + W_queue + E_exec + T_download.
  * The cloud tier has unbounded capacity and a fixed 95 ms round trip.

Schedulers
----------
  orsaf   : Chebyshev node selection (Eq. 21) + preemption + rebalancing
            with migration guard (M1).
  heft    : earliest-finish-time greedy.
  rr      : round robin.
  fcfs    : first feasible node.
  fuzzy   : fuzzy membership score on latency and headroom (FuzzyDRL-Fog /
            FuzzyOff surrogate; see caveat in report_results()).

Routers
-------
  idcra   : trained GBT + load-adaptive mixing (Eq. 6-7).
  thresh  : static tau > 0.6.
  all_fog : everything to the fog tier.
  all_cloud: everything to the cloud tier.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

CLOUD_RTT_MS = 95.0
P_TX_W, P_RX_W, P_TX_WAN_W = 0.5, 0.3, 2.1
B_GF_KBPS = 100_000.0     # gateway->fog, 100 Mbps
B_BH_KBPS = 50_000.0      # fog->cloud backhaul, 50 Mbps
B_FF_KBPS = 100_000.0     # fog->fog


# ------------------------------------------------------------------
@dataclass
class Node:
    node_id: int
    cpu_mips: float
    mem_mb: float
    latency_ms: float
    p_dyn_w: float
    p_idle_w: float
    load: float = 0.0
    mem_used: float = 0.0
    busy_ms: float = 0.0
    n_done: int = 0

    @property
    def residual_cpu(self) -> float:
        return self.cpu_mips * max(0.0, 1.0 - self.load)

    @property
    def residual_mem(self) -> float:
        return max(0.0, self.mem_mb - self.mem_used)

    def admits(self, cpu: float, mem: float) -> bool:
        return self.residual_cpu >= cpu and self.residual_mem >= mem


@dataclass
class Job:
    tid: int
    tau: float
    phi: float
    delta_s: float
    rho_kb: float
    cpu: float
    mem: float
    psi: float
    label: int
    pi: int
    arrival_ms: float
    start_ms: float = -1.0
    finish_ms: float = -1.0
    node: int = -2          # -1 = cloud
    exec_ms: float = 0.0
    preempted: bool = False
    energy_j: float = 0.0

    @property
    def latency_ms(self) -> float:
        return self.finish_ms - self.arrival_ms

    @property
    def wait_ms(self) -> float:
        return max(0.0, self.start_ms - self.arrival_ms)


# ------------------------------------------------------------------
def build_nodes(cfg) -> List[Node]:
    nodes, nid = [], 0
    for key in ("class_A", "class_B", "class_C", "class_D"):
        c = cfg["fog_nodes"][key]
        for _ in range(c["count"]):
            nodes.append(Node(nid, float(c["cpu_mips"]), float(c["mem_mb"]),
                              float(c["latency_ms"]), float(c["p_dyn"]),
                              float(c["p_idle"])))
            nid += 1
    return nodes


# ------------------------------------------------------------------
DRL_INFER_MS = 4.0   # per-decision DRL inference cost on fog-class
                     # hardware (Choppara & Mangalampalli report
                     # 12-18 ms on constrained nodes; 4 ms is the
                     # conservative end)
CLOUD_SERVERS = 43   # sized so Cloud-Only runs at ~70% cloud utilisation,
                     # matching the fog tier design target (Lambda_thresh=0.80)
CLOUD_MIPS_PER_SERVER = 40000.0


class Simulator:
    def __init__(self, nodes: List[Node], scheduler: str = "orsaf",
                 router: str = "idcra", w=(0.5, 0.3, 0.2),
                 psi_thresh: float = 0.75, lam_thresh: float = 0.80,
                 preemption: bool = True, rebalancing: bool = True,
                 guard_m1: bool = True, rebalance_ms: float = 250.0,
                 rng: Optional[np.random.Generator] = None):
        self.nodes = {n.node_id: n for n in nodes}
        self.scheduler = scheduler
        self._cloud_free = np.zeros(CLOUD_SERVERS)
        self._drl_overhead_ms = 0.0
        self.router = router
        self.w_L, self.w_E, self.w_eta = w
        self.psi_thresh = psi_thresh
        self.lam_thresh = lam_thresh
        self.preemption = preemption
        self.rebalancing = rebalancing
        self.guard_m1 = guard_m1
        self.rebalance_ms = rebalance_ms
        self.rng = rng or np.random.default_rng(0)

        self._events: List[Tuple[float, int, object]] = []
        self._seq = 0
        self._pending: List[Tuple[float, int, Job]] = []   # backlog heap
        self._running: Dict[int, List[Job]] = {n: [] for n in self.nodes}
        self.done: List[Job] = []
        self._rr = 0
        self.n_migrations = 0
        self.n_preemptions = 0
        self.n_hp_seen = 0
        self.n_hp_unbounded = 0     # Lemma 9 excluded case
        self.t = 0.0

    # -------- event helpers --------
    def _push(self, when: float, kind: str, payload):
        self._seq += 1
        heapq.heappush(self._events, (when, self._seq, (kind, payload)))

    # -------- node selection --------
    def _feasible(self, job: Job) -> List[Node]:
        return [n for n in self.nodes.values() if n.admits(job.cpu, job.mem)]

    def _pick(self, feas: List[Node], job: Job) -> Node:
        if self.scheduler == "rr":
            self._rr = (self._rr + 1) % len(feas)
            return feas[self._rr]
        if self.scheduler == "fcfs":
            return feas[0]
        if self.scheduler == "heft":
            return min(feas, key=lambda n: n.latency_ms
                       + job.cpu / max(n.cpu_mips * (1 - n.load), 1e-9) * 1000.0)
        # ---- B5: FuzzyDRL-Fog (Choppara & Mangalampalli, IEEE Access 2024)
        # Fuzzy inference over (task length, node capacity, node load)
        # produces a task priority; a DRL value head estimates the expected
        # completion cost of each node. Selection maximises the fuzzy
        # priority weighted by the negated value estimate. The DRL policy
        # adds per-decision inference overhead on memory-constrained nodes.
        if self.scheduler == "fuzzydrl":
            L_max = max(n.cpu_mips for n in self.nodes.values())
            def fz_priority(n):
                # fuzzy membership: task length, capacity, load
                mu_len = min(1.0, job.cpu / 8000.0)
                mu_cap = n.cpu_mips / L_max
                mu_load = max(0.0, 1.0 - n.load)
                return 0.4 * mu_cap + 0.35 * mu_load + 0.25 * (1.0 - mu_len)
            def drl_value(n):
                # value head approximates expected completion time
                ex = job.cpu / max(n.cpu_mips * (1 - n.load), 1e-9) * 1000.0
                return ex + n.latency_ms + 1000.0 * n.load
            best = max(feas, key=lambda n: fz_priority(n)
                       - 0.002 * drl_value(n))
            self._drl_overhead_ms += DRL_INFER_MS
            return best

        # ---- B6: FuzzyOff (Abdulazeez & Askar, IEEE Access 2024)
        # Three-layer offloading decision. Fuzzy logic scores each tier on
        # (latency, residual capacity, task deadline urgency); no DRL value
        # head and no queue-depth term, so it is load-blind under congestion.
        if self.scheduler == "fuzzyoff":
            def fz_off(n):
                mu_lat = max(0.0, 1.0 - n.latency_ms / 20.0)
                mu_hd = max(0.0, 1.0 - n.load)
                mu_dl = 1.0 if job.delta_s < 30.0 else 0.5
                return 0.45 * mu_lat + 0.35 * mu_hd + 0.20 * mu_dl
            return max(feas, key=fz_off)
        # ---- orsaf: per-task Chebyshev, Eq. (21) ----
        costs = []
        for n in feas:
            E = job.cpu / max(n.cpu_mips * (1 - n.load), 1e-9) * 1000.0
            L = 2 * n.latency_ms + E
            u = n.load + job.cpu / n.cpu_mips
            costs.append((L, E, u))
        Ls = min(c[0] for c in costs)
        Es = min(c[1] for c in costs)
        us = min(c[2] for c in costs)
        best, bv = None, float("inf")
        for n, (L, E, u) in zip(feas, costs):
            v = max(self.w_L * (L - Ls) / max(Ls, 1e-12),
                    self.w_E * (E - Es) / max(Es, 1e-12),
                    self.w_eta * (u - us) / max(us, 1e-12))
            if v < bv:
                bv, best = v, n
        return best

    # -------- dispatch --------
    def _start(self, job: Job, node: Node):
        frac = job.cpu / node.cpu_mips
        exec_ms = job.cpu / max(node.cpu_mips * (1 - node.load), 1e-9) * 1000.0
        node.load = min(1.0, node.load + frac)
        node.mem_used += job.mem
        job.node = node.node_id
        job.start_ms = self.t
        job.exec_ms = exec_ms
        self._running[node.node_id].append(job)
        node.busy_ms += exec_ms
        if self.scheduler == "fuzzydrl":
            job.finish_ms += DRL_INFER_MS      # DRL inference overhead
        self._push(self.t + exec_ms + node.latency_ms, "done", job)

    def _finish(self, job: Job):
        node = self.nodes[job.node]
        node.load = max(0.0, node.load - job.cpu / node.cpu_mips)
        node.mem_used = max(0.0, node.mem_used - job.mem)
        if job in self._running[node.node_id]:
            self._running[node.node_id].remove(job)
        node.n_done += 1
        job.finish_ms = self.t
        up = job.rho_kb / B_GF_KBPS * 1000.0
        e_cpu = node.p_dyn_w * job.exec_ms / 1000.0 \
            + node.p_idle_w * max(0.0, job.wait_ms) / 1000.0 * 0.1
        e_com = P_TX_W * up / 1000.0 + P_RX_W * node.latency_ms / 1000.0
        job.energy_j = e_cpu + e_com
        self.done.append(job)
        self._drain()

    def _cloud(self, job: Job):
        # Finite-capacity cloud: M/M/c-style queueing at the cloud tier.
        # An unbounded-CPU cloud is physically unrealisable and makes the
        # Cloud-Only baseline artificially strong; CLOUD_SERVERS reflects
        # the share of a multi-tenant region provisioned to this deployment.
        up = job.rho_kb / B_BH_KBPS * 1000.0
        ex = job.cpu / CLOUD_MIPS_PER_SERVER * 1000.0
        # earliest free server
        idx = int(np.argmin(self._cloud_free))
        start = max(self.t + CLOUD_RTT_MS + up, self._cloud_free[idx])
        self._cloud_free[idx] = start + ex
        job.node = -1
        job.start_ms = start
        job.exec_ms = ex
        job.finish_ms = start + ex
        job.energy_j = (P_TX_WAN_W * (up + CLOUD_RTT_MS) / 1000.0
                        + 120.0 * ex / 1000.0)
        self.done.append(job)

    # -------- backlog --------
    def _enqueue(self, job: Job):
        heapq.heappush(self._pending, (-job.psi, job.tid, job))

    def _drain(self):
        moved = True
        while moved and self._pending:
            moved = False
            keep = []
            while self._pending:
                _, _, job = heapq.heappop(self._pending)
                feas = self._feasible(job)
                if feas:
                    self._start(job, self._pick(feas, job))
                    moved = True
                else:
                    keep.append(job)
            for j in keep:
                self._enqueue(j)
            if not moved:
                break

    def _try_preempt(self, job: Job) -> bool:
        if not self.preemption or job.psi < self.psi_thresh:
            return False
        victim, vnode = None, None
        for nid, running in self._running.items():
            for rt in running:
                if victim is None or rt.psi < victim.psi:
                    victim, vnode = rt, self.nodes[nid]
        if victim is None or job.psi <= victim.psi:
            return False
        # suspend victim, return its capacity, requeue it
        vnode.load = max(0.0, vnode.load - victim.cpu / vnode.cpu_mips)
        vnode.mem_used = max(0.0, vnode.mem_used - victim.mem)
        self._running[vnode.node_id].remove(victim)
        victim.preempted = True
        self._enqueue(victim)
        self.n_preemptions += 1
        if vnode.admits(job.cpu, job.mem):
            self._start(job, vnode)
            return True
        self._enqueue(job)
        return False

    # -------- rebalancing, Theorem 11 --------
    def _rebalance(self):
        if not self.rebalancing:
            return
        for nid, node in self.nodes.items():
            if node.load <= self.lam_thresh:
                continue
            others = [n for k, n in self.nodes.items() if k != nid]
            if not others:
                continue
            tgt = min(others, key=lambda n: n.load)
            cands = sorted([j for _, _, j in self._pending], key=lambda j: j.psi)
            for jm in cands:
                if jm.mem > tgt.residual_mem:
                    continue
                if self.guard_m1:
                    ok = (tgt.load + jm.cpu / tgt.cpu_mips) <= self.lam_thresh
                else:
                    ok = tgt.residual_cpu >= jm.cpu      # the original test
                if ok:
                    self._pending = [(p, t, j) for (p, t, j)
                                     in self._pending if j.tid != jm.tid]
                    heapq.heapify(self._pending)
                    self._start(jm, tgt)
                    self.n_migrations += 1
                    break

    def phi_potential(self) -> float:
        return sum(max(0.0, n.load - self.lam_thresh) ** 2
                   for n in self.nodes.values())

    # -------- main loop --------
    def run(self, jobs: List[Job], route_fn):
        for j in jobs:
            self._push(j.arrival_ms, "arrive", j)
        horizon = max(j.arrival_ms for j in jobs) + 1.0
        tick = 0.0
        while tick < horizon:
            tick += self.rebalance_ms
            self._push(tick, "rebalance", None)

        while self._events:
            self.t, _, (kind, payload) = heapq.heappop(self._events)
            if kind == "arrive":
                job: Job = payload
                if route_fn(job, self) == "cloud":
                    self._cloud(job)
                else:
                    if job.psi >= self.psi_thresh:
                        self.n_hp_seen += 1
                    feas = self._feasible(job)
                    if feas:
                        self._start(job, self._pick(feas, job))
                    elif not self._try_preempt(job):
                        if job.psi >= self.psi_thresh:
                            self.n_hp_unbounded += 1
                        self._enqueue(job)
            elif kind == "done":
                self._finish(payload)
            else:
                self._rebalance()
                self._drain()

        # anything still queued at the end is forced through
        while self._pending:
            _, _, job = heapq.heappop(self._pending)
            node = min(self.nodes.values(), key=lambda n: n.load)
            self.t += 0.001
            self._start(job, node)
            self.t += job.exec_ms
            self._finish(job)

    # -------- metrics --------
    def metrics(self) -> Dict[str, float]:
        fog = [j for j in self.done if j.node >= 0]
        lat = np.array([j.latency_ms for j in self.done])
        ex = np.array([j.exec_ms for j in self.done])
        en = np.array([j.energy_j for j in self.done])
        span = max((j.finish_ms for j in self.done), default=1.0) / 1000.0
        util = [min(1.0, n.busy_ms / max(self.t, 1e-9)) for n in self.nodes.values()]
        return {
            "latency_ms": float(lat.mean()),
            "p95_ms": float(np.percentile(lat, 95)),
            "exec_ms": float(ex.mean()),
            "utilisation": float(np.mean(util) * 100.0) if fog else float("nan"),
            "throughput": float(len(self.done) / max(span, 1e-9)),
            "energy_j": float(en.mean()),
            "n_fog": len(fog),
            "n_cloud": len(self.done) - len(fog),
            "migrations": self.n_migrations,
            "preemptions": self.n_preemptions,
            "hp_unbounded_pct": (100.0 * self.n_hp_unbounded /
                                 max(self.n_hp_seen, 1)),
        }
