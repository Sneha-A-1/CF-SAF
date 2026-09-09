"""
orsa_f.py
---------
Optimised Resource Scheduling Algorithm for Fog Nodes (ORSA-F).

Key design elements:
  1. Chebyshev scalarisation of multi-objective problem P1
  2. Urgency-aware preemption
  3. Convergence-guaranteed load rebalancing (Lyapunov proof)
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------
@dataclass
class FogNode:
    node_id:    int
    cpu_mips:   float
    mem_mb:     float
    latency_ms: float
    p_dyn_w:    float
    p_idle_w:   float
    load:       float = 0.0   # lambda_i

    @property
    def residual_cpu(self):
        return self.cpu_mips * (1.0 - self.load)

    @property
    def residual_mem(self):
        return self.mem_mb * (1.0 - self.load)


@dataclass(order=True)
class Task:
    """Priority-queue item: higher psi = higher priority (negate for min-heap)."""
    neg_psi:     float                    # negated priority for min-heap
    task_id:     int = field(compare=False)
    tau:         float = field(compare=False)
    phi:         float = field(compare=False)
    delta:       float = field(compare=False)
    cpu_demand:  float = field(compare=False)
    mem_demand:  float = field(compare=False)
    task_type:   int   = field(compare=False)  # 0=twin update, 1=analytical

    @property
    def psi(self):
        return -self.neg_psi


# ----------------------------------------------------------------
# ORSA-F Scheduler
# ----------------------------------------------------------------
class ORSAF:
    """
    ORSA-F: Optimised Resource Scheduling Algorithm for Fog Nodes.

    Parameters
    ----------
    fog_nodes        : list of FogNode objects
    w_L, w_E, w_eta  : Chebyshev objective weights (sum = 1.0)
    psi_thresh       : Preemption threshold
    lambda_thresh    : Load rebalancing ceiling (Lambda_thresh)
    ref_L, ref_E, ref_eta : Reference (ideal) values for Chebyshev
    """

    def __init__(self,
                 fog_nodes: List[FogNode],
                 w_L: float = 0.50,
                 w_E: float = 0.30,
                 w_eta: float = 0.20,
                 psi_thresh: float = 0.75,
                 lambda_thresh: float = 0.80,
                 ref_L: float = 5.0,
                 ref_E: float = 3.0,
                 ref_eta: float = 0.85):
        self.nodes = {n.node_id: n for n in fog_nodes}
        self.w_L = w_L
        self.w_E = w_E
        self.w_eta = w_eta
        self.psi_thresh = psi_thresh
        self.lambda_thresh = lambda_thresh
        self.ref_L = ref_L
        self.ref_E = ref_E
        self.ref_eta = ref_eta

        # Priority queue (min-heap on neg_psi, so highest psi served first)
        self._queue: List[Task] = []
        # Running tasks per node: {node_id: [Task, ...]}
        self._running: Dict[int, List[Task]] = {n: [] for n in self.nodes}
        # Assignment record
        self.assignments: Dict[int, int] = {}
        # instrumentation for Lemma 9's excluded case and Theorem 11
        self.n_dispatched_feasible = 0
        self.n_dispatched_preempt  = 0
        self.n_reenqueued_highprio = 0
        self.n_highprio_seen       = 0
        self.n_migrations          = 0

    def _per_task_costs(self, node: "FogNode", task: "Task"):
        """
        Per-task costs (L_ij, E_ij, u_ij) of assigning `task` to `node`.
        Eqs. (17)-(19) in the revised manuscript.
        """
        residual = node.cpu_mips * max(1.0 - node.load, 1e-9)
        E_ij = task.cpu_demand / residual                 # Eq. (18)
        L_ij = node.latency_ms + E_ij + node.latency_ms   # Eq. (17), W=0 when feasible
        u_ij = node.load + task.cpu_demand / node.cpu_mips  # Eq. (19)
        return L_ij, E_ij, u_ij

    def _select_node_chebyshev(self, feasible, task):
        """
        Per-task Chebyshev node selection, Eq. (21).

        Two passes over the feasible set:
          1. componentwise utopia point (Eq. 20)
          2. weighted max normalised deviation; return the argmin

        Replaces the previous weighted SUM of raw quantities, which mixed
        units (ms^-1, a dimensionless ratio, a load fraction) and was not
        the Chebyshev scalarisation Eq. (16) declares.
        """
        if not feasible:
            return None
        costs = [self._per_task_costs(n, task) for n in feasible]
        L_star = min(c[0] for c in costs)
        E_star = min(c[1] for c in costs)
        u_star = min(c[2] for c in costs)

        best_node, best_val = None, float("inf")
        for node, (L, E, u) in zip(feasible, costs):
            dL = (L - L_star) / max(L_star, 1e-12)
            dE = (E - E_star) / max(E_star, 1e-12)
            du = (u - u_star) / max(u_star, 1e-12)
            val = max(self.w_L * dL, self.w_E * dE, self.w_eta * du)
            if val < best_val:
                best_val, best_node = val, node
        return best_node

    def _feasible_nodes(self, task: Task) -> List[FogNode]:
        return [n for n in self.nodes.values()
                if n.residual_cpu >= task.cpu_demand
                and n.residual_mem >= task.mem_demand]

    def enqueue(self, task: Task):
        heapq.heappush(self._queue, task)

    def _assign(self, task: Task, node: FogNode):
        node.load = min(
            1.0, node.load + task.cpu_demand / node.cpu_mips)
        self._running[node.node_id].append(task)
        self.assignments[task.task_id] = node.node_id

    def _execution_time(self, task: Task, node: FogNode) -> float:
        """Equation 2 in the paper."""
        residual = node.cpu_mips * (1.0 - node.load)
        return task.cpu_demand / max(residual, 1.0)

    def _release(self, task: Task, node_id: int):
        node = self.nodes[node_id]
        node.load = max(0.0, node.load - task.cpu_demand / node.cpu_mips)
        if task in self._running[node_id]:
            self._running[node_id].remove(task)

    def _rebalance(self):
        """
        Load rebalancing with migration guard (M1), Theorem 11.

            lambda_k + r_m / kappa_k  <=  Lambda_thresh

        The previous test was kappa_k*(1-lambda_k) >= r_m, which only
        requires the destination to stay below 100% and therefore admits
        migrations that push it far past Lambda_thresh. Under that test the
        destination term of Delta_Phi can be positive and the Lyapunov
        argument fails (Corollary 12 gives an explicit counterexample).

        Queued tasks are scanned in increasing psi until one satisfies the
        guard, so a migration is not abandoned merely because the least
        urgent task is too large for the destination.
        """
        for node_id, node in self.nodes.items():
            if node.load <= self.lambda_thresh:
                continue
            other_nodes = [n for nid, n in self.nodes.items() if nid != node_id]
            if not other_nodes:
                continue
            target = min(other_nodes, key=lambda n: n.load)

            candidates = [t for t in self._queue
                          if self.assignments.get(t.task_id) == node_id]
            if not candidates:
                continue

            for task_m in sorted(candidates, key=lambda t: t.psi):
                if (task_m.mem_demand > target.residual_mem):
                    continue
                # ---- guard (M1) ----
                if (target.load + task_m.cpu_demand / target.cpu_mips
                        <= self.lambda_thresh):
                    self._release(task_m, node_id)
                    self._assign(task_m, target)
                    self.n_migrations += 1
                    break

    def schedule(self) -> Dict[int, int]:
        """
        Main scheduling loop. Processes all tasks in the priority queue.
        Returns assignment dict {task_id: node_id}.
        """
        while self._queue:
            task = heapq.heappop(self._queue)
            feasible = self._feasible_nodes(task)

            if task.psi >= self.psi_thresh:
                self.n_highprio_seen += 1

            if feasible:
                self.n_dispatched_feasible += 1
                # Assign to best-scoring feasible node
                best_node = self._select_node_chebyshev(feasible, task)
                self._assign(task, best_node)
            else:
                # Preemption check
                preempted = False
                if task.psi >= self.psi_thresh:
                    # Find node with lowest-priority running task
                    candidate_node, candidate_task = None, None
                    min_running_psi = float('inf')
                    for nid, running in self._running.items():
                        for rt in running:
                            if rt.psi < min_running_psi:
                                min_running_psi = rt.psi
                                candidate_node = self.nodes[nid]
                                candidate_task = rt

                    if (candidate_task is not None
                            and task.psi > candidate_task.psi):
                        # Preempt
                        self._release(candidate_task, candidate_node.node_id)
                        heapq.heappush(self._queue, candidate_task)
                        self._assign(task, candidate_node)
                        preempted = True
                        self.n_dispatched_preempt += 1

                if not preempted:
                    if task.psi >= self.psi_thresh:
                        self.n_reenqueued_highprio += 1
                    # Re-enqueue with back-off (re-add to queue)
                    heapq.heappush(self._queue,
                                   Task(task.neg_psi * 1.01,
                                        task.task_id, task.tau, task.phi,
                                        task.delta, task.cpu_demand,
                                        task.mem_demand, task.task_type))

            # Load rebalancing
            self._rebalance()

        return self.assignments
