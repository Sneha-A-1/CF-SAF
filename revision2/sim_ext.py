"""
sim_ext.py
----------
Extension of the CF-SAF discrete-event simulator used for the second
revision.  It adds the node-local queues that the rebalancing analysis
(Theorem 8) assumes, a theorem-faithful rebalancing procedure, and
instrumentation for Lemma 2 and Corollary 3.

Model additions
  * Each fog node f_i has a local queue Q_i.  Its committed load is
        lambda_i^c = lambda_i^run + sum_{t in Q_i} r_t / kappa_i,
    which is the quantity Theorem 8 reasons about.  With empty local
    queues (the main experiments) lambda^c reduces to lambda^run.
  * Locality-bound tasks (gateway affinity) are committed to their home
    node: they start there if admissible, otherwise join Q_home.
  * Rebalancing (every 250 ms) repeats passes until no migration is
    admissible.  In each pass every node with lambda^c > Lambda scans its
    queued tasks in increasing psi and, for each, the other nodes in
    increasing lambda^c, migrating the first task that satisfies the guard
        lambda_k^c + r_m / kappa_k <= Lambda                (M1)
    (memory permitting).  'unguarded' replaces (M1) with the original
    residual-capacity test kappa_k (1 - lambda_k^c) >= r_m.
  * A migrated task pays a fog-to-fog transfer delay rho / B_ff.
  * Node failure: at t_fail the node's capacity is withdrawn; its running
    tasks are restarted into its local queue and the gateway keeps
    committing to it until t_recover.  Only rebalancing can move this work.
"""
from __future__ import annotations
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
import heapq, math
import numpy as np
from src.simulation.simulator import (Simulator, Job, Node, B_FF_KBPS,
                                      B_GF_KBPS, P_TX_W)
from simfix import SimFix

LAM = 0.80


class SimExt(SimFix):
    def __init__(self, nodes, *, rb_mode="guarded", affinity=None,
                 background=None, fail=None, track=False, **kw):
        kw.setdefault("rebalancing", rb_mode != "off")
        kw.setdefault("guard_m1", rb_mode != "unguarded")
        super().__init__(nodes, **kw)
        self.rb_mode = rb_mode
        self.local = {n: [] for n in self.nodes}          # node-local queues
        self.affinity = affinity or {}                    # tid -> home node
        self.bg = background or {}                        # nid -> fixed load
        for nid, b in self.bg.items():
            self.nodes[nid].load += b
        self.fail = fail                                  # (nid, t_fail, t_rec)
        self.failed = set()
        self.track = track
        self.inv = []        # per-invocation records
        self.phi_steps = []  # Delta Phi per migration
        self.trace = []      # (t, Phi, max lambda^c)
        self.over_ticks = 0
        self.ticks = 0

    # ---- committed load ------------------------------------------------
    def lam_c(self, nid):
        n = self.nodes[nid]
        if nid in self.failed:
            # a failed node has threshold 0: any committed work is excess
            return math.inf if self.local[nid] else 0.0
        return n.load + sum(j.cpu for j in self.local[nid]) / n.cpu_mips

    def lam_run(self, nid):
        return 0.0 if nid in self.failed else self.nodes[nid].load

    def phi(self):
        s = 0.0
        for nid in self.nodes:
            l = self.lam_c(nid)
            if nid in self.failed:
                # failed node: excess measured by queued work over a
                # reference capacity (its nominal kappa)
                l = sum(j.cpu for j in self.local[nid]) / self.nodes[nid].cpu_mips
                l += LAM if self.local[nid] else 0.0
            s += max(0.0, l - LAM) ** 2
        return s

    # ---- admission -----------------------------------------------------
    def _feasible(self, job):
        return [n for n in self.nodes.values()
                if n.node_id not in self.failed and n.admits(job.cpu, job.mem)]

    def _start(self, job, node):
        extra = getattr(job, "mig_delay", 0.0)
        super()._start(job, node)
        if extra:
            # shift completion by the migration transfer delay
            for i, (w, s, (k, p)) in enumerate(self._events):
                if p is job and k == "done":
                    self._events[i] = (w + extra, s, (k, p))
                    heapq.heapify(self._events)
                    break

    def _serve_local(self, nid):
        if nid in self.failed:
            return
        q = self.local[nid]
        q.sort(key=lambda j: -j.psi)
        node = self.nodes[nid]
        i = 0
        while i < len(q):
            if node.admits(q[i].cpu, q[i].mem):
                self._start(q.pop(i), node)
            else:
                i += 1

    def _finish(self, job):
        nid = job.node
        node = self.nodes[nid]
        node.load = max(0.0, node.load - job.cpu / node.cpu_mips)
        node.mem_used = max(0.0, node.mem_used - job.mem)
        if job in self._running[nid]:
            self._running[nid].remove(job)
        node.n_done += 1
        job.finish_ms = self.t
        up = job.rho_kb / B_GF_KBPS * 1000.0
        e_cpu = node.p_dyn_w * job.exec_ms / 1000.0 \
            + node.p_idle_w * max(0.0, job.wait_ms) / 1000.0 * 0.1
        e_com = P_TX_W * up / 1000.0 + 0.3 * node.latency_ms / 1000.0
        e_mig = P_TX_W * getattr(job, "mig_delay", 0.0) / 1000.0
        job.energy_j = e_cpu + e_com + e_mig
        self.done.append(job)
        self._serve_local(nid)
        self._drain()

    # ---- failure ---------------------------------------------------------
    def _do_fail(self, nid):
        self.failed.add(nid)
        node = self.nodes[nid]
        for j in list(self._running[nid]):
            # restart: drop pending completion, return work to local queue
            self._events = [e for e in self._events
                            if not (e[2][0] == "done" and e[2][1] is j)]
            heapq.heapify(self._events)
            self._running[nid].remove(j)
            j.rem_frac, j.exec_done = 1.0, 0.0     # restart: progress lost
            node.busy_ms -= getattr(j, "seg_exec", 0.0)
            self.local[nid].append(j)
        node.load = self.bg.get(nid, 0.0)
        node.mem_used = 0.0

    def _do_recover(self, nid):
        self.failed.discard(nid)
        self._serve_local(nid)

    # ---- rebalancing (Theorem 8) -----------------------------------------
    def _guard_ok(self, k, jm):
        nk = self.nodes[k]
        if k in self.failed or jm.mem > nk.residual_mem:
            return False
        lk = self.lam_c(k)
        if self.rb_mode == "unguarded":
            return nk.cpu_mips * (1.0 - lk) >= jm.cpu
        return lk + jm.cpu / nk.cpu_mips <= LAM + 1e-12

    def _condition(self):
        """Hypotheses of Theorem 8(ii) at the start of an invocation."""
        act = [k for k in self.nodes if k not in self.failed]
        h1 = all(self.lam_run(k) <= LAM + 1e-12 for k in act)
        over = [k for k in self.nodes if self.lam_c(k) > LAM]
        rmax = max((j.cpu for k in over for j in self.local[k]), default=0.0)
        slack = sum(self.nodes[k].cpu_mips * (LAM - self.lam_c(k)) for k in act)
        if self.failed:
            # failed node contributes its queued demand as negative slack
            slack -= sum(j.cpu for k in self.failed for j in self.local[k])
        return h1 and slack >= (len(act) - 1) * rmax

    def _rebalance(self):
        self.ticks += 1
        over0 = [k for k in self.nodes if self.lam_c(k) > LAM]
        if over0:
            self.over_ticks += 1
        if self.track:
            self.trace.append((self.t, self.phi(),
                               max(min(self.lam_c(k), 2.0) for k in self.nodes)))
        if self.rb_mode == "off" or not over0:
            return
        if not any(self.local[k] for k in over0):
            # trigger fires but nothing is queued on the overloaded node(s)
            self.inv.append(dict(t=self.t, mig=0, phi0=self.phi(),
                                 phi1=self.phi(), cond=self._condition(),
                                 conv=False, eligible=False))
            return
        phi0, cond = self.phi(), self._condition()
        mig, moved_once = 0, set()
        progress = True
        while progress:
            progress = False
            for i in list(self.nodes):
                if self.lam_c(i) <= LAM:
                    continue
                cand = sorted(self.local[i], key=lambda j: j.psi)
                dests = sorted((k for k in self.nodes if k != i),
                               key=self.lam_c)
                done = False
                for jm in cand:
                    for k in dests:
                        if self._guard_ok(k, jm):
                            before = self.phi()
                            self.local[i].remove(jm)
                            jm.mig_delay = getattr(jm, "mig_delay", 0.0) + \
                                jm.rho_kb / B_FF_KBPS * 1000.0
                            self.local[k].append(jm)
                            self.phi_steps.append(self.phi() - before)
                            if jm.tid in moved_once:
                                self.n_remig = getattr(self, "n_remig", 0) + 1
                            moved_once.add(jm.tid)
                            mig += 1
                            self.n_migrations += 1
                            self._serve_local(k)
                            done = progress = True
                            break
                    if done:
                        break
        phi1 = self.phi()
        conv = all(self.lam_c(k) <= LAM + 1e-12 for k in self.nodes)
        self.inv.append(dict(t=self.t, mig=mig, phi0=phi0, phi1=phi1,
                             cond=cond, conv=conv, eligible=True))

    # ---- main loop -------------------------------------------------------
    def run(self, jobs, route_fn):
        for j in jobs:
            self._push(j.arrival_ms, "arrive", j)
        horizon = max(j.arrival_ms for j in jobs) + 1.0
        tick = 0.0
        while tick < horizon:
            tick += self.rebalance_ms
            self._push(tick, "rebalance", None)
        if self.fail:
            nid, tf, tr = self.fail
            self._push(tf, "fail", nid)
            self._push(tr, "recover", nid)
        while self._events:
            self.t, _, (kind, payload) = heapq.heappop(self._events)
            if kind == "arrive":
                job = payload
                job.mig_delay = 0.0
                if route_fn(job, self) == "cloud":
                    self._cloud(job)
                    continue
                if job.psi >= self.psi_thresh:
                    self.n_hp_seen += 1
                h = self.affinity.get(job.tid)
                if h is not None:
                    hn = self.nodes[h]
                    if h not in self.failed and hn.admits(job.cpu, job.mem) \
                            and not self.local[h]:
                        self._start(job, hn)
                    else:
                        self.local[h].append(job)
                    continue
                feas = self._feasible(job)
                if feas:
                    self._start(job, self._pick(feas, job))
                elif not self._try_preempt(job):
                    if job.psi >= self.psi_thresh:
                        self.n_hp_unbounded += 1
                    self._enqueue(job)
            elif kind == "done":
                if payload.node in self.failed:
                    continue
                self._finish(payload)
            elif kind == "fail":
                self._do_fail(payload)
            elif kind == "recover":
                self._do_recover(payload)
            else:
                self._rebalance()
                for k in self.nodes:
                    self._serve_local(k)
                self._drain()
        # flush anything left (local queues, then global backlog)
        rest = [j for k in self.nodes for j in self.local[k]]
        for k in self.nodes:
            self.local[k] = []
        while self._pending:
            rest.append(heapq.heappop(self._pending)[2])
        for job in rest:
            node = min((n for n in self.nodes.values()
                        if n.node_id not in self.failed),
                       key=lambda n: n.load)
            self.t += 0.001
            SimFix._start(self, job, node)
            self.t += job.exec_ms
            self._events = []
            self._finish(job)

    # ---- extra metrics -----------------------------------------------------
    def ext_metrics(self):
        m = self.metrics()
        lat = np.array([j.latency_ms for j in self.done])
        urg = np.array([j.latency_ms for j in self.done if j.label == 1])
        dl = np.array([j.latency_ms > 1000 * j.delta_s for j in self.done])
        inv = [r for r in self.inv if r["eligible"]]
        m.update(
            p99_ms=float(np.percentile(lat, 99)),
            urg_mean_ms=float(urg.mean()) if len(urg) else float("nan"),
            urg_p95_ms=float(np.percentile(urg, 95)) if len(urg) else float("nan"),
            dl_miss_pct=float(100 * dl.mean()),
            over_tick_pct=100.0 * self.over_ticks / max(self.ticks, 1),
            invocations=len(inv),
            inv_trigger_noeligible=len(self.inv) - len(inv),
            inv_conv_pct=100.0 * np.mean([r["conv"] for r in inv]) if inv else float("nan"),
            inv_cond_pct=100.0 * np.mean([r["cond"] for r in inv]) if inv else float("nan"),
            conv_given_cond=(100.0 * np.mean([r["conv"] for r in inv if r["cond"]])
                             if any(r["cond"] for r in inv) else float("nan")),
            conv_given_notcond=(100.0 * np.mean([r["conv"] for r in inv if not r["cond"]])
                                if any(not r["cond"] for r in inv) else float("nan")),
            phi_nonneg_steps=int(sum(1 for d in self.phi_steps if d >= -1e-15)),
            mig_steps=len(self.phi_steps),
            phi_red_pct=(100.0 * np.mean([1 - r["phi1"] / r["phi0"] for r in inv
                                          if r["phi0"] > 0 and np.isfinite(r["phi0"])])
                         if inv else float("nan")),
            remig=getattr(self, "n_remig", 0),
            max_mig_per_inv=max((r["mig"] for r in inv), default=0),
        )
        return m
