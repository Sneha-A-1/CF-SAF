import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
"""Cloud-Only vs CF-SAF (Reviewer 2, comment b) and empirical checks of
Lemma 2 (comment d) and Corollary 3 (comment e)."""
import json, sys, time
import numpy as np
from common import *  # noqa
from src.simulation import simulator as S
from src.simulation.simulator import B_GF_KBPS
from simfix import SimFix


def run(system, seed, c=None, df=None, rate=RATE, gaps=None):
    rk, sc = {"Cloud-Only": ("all_cloud", "fcfs"), "CF-SAF": ("idcra", "orsaf"),
              "HEFT-Fog": ("idcra", "heft"), "Threshold-Fog": ("thresh", "fcfs")}[system]
    rng = np.random.default_rng(seed)
    df = FULL.head(NT) if df is None else df
    jobs = load_jobs(df, rate, rng, gaps=None if gaps is None else gaps(rng, len(df)))
    sim = SimFix(build_nodes(cfg), scheduler=sc)
    if c is not None:
        sim._cloud_free = np.zeros(c)
    # instrument Lemma 2: record lambda at dispatch and dispatch case
    rec = {}
    o_start, o_pre = sim._start, sim._try_preempt
    state = {"pre": False}

    def _start(job, node):
        rec[job.tid] = (node.load, node.node_id, state["pre"])
        o_start(job, node)
    sim._start = _start

    def _try_preempt(job):
        state["pre"] = True
        r = o_pre(job)
        state["pre"] = False
        return r
    sim._try_preempt = _try_preempt
    sim.run(jobs, router(rk))
    return sim, rec


def per_task_stats(sim, rec):
    d = sim.done
    lat = np.array([j.latency_ms for j in d])
    urg = np.array([j.label == 1 for j in d])
    dl = np.array([j.delta_s for j in d]) * 1000.0
    pi = np.array([j.pi for j in d])
    cloud = np.array([j.node == -1 for j in d])
    rho = np.array([j.rho_kb for j in d])
    out = dict(
        mean=float(lat.mean()), p95=float(np.percentile(lat, 95)),
        urg_mean=float(lat[urg].mean()), urg_p95=float(np.percentile(lat[urg], 95)),
        non_mean=float(lat[~urg].mean()),
        miss_all=float(100 * (lat > dl).mean()),
        miss_urg=float(100 * (lat[urg] > dl[urg]).mean()),
        tight=int((dl < 1000).sum()),
        miss_tight=float(100 * (lat[dl < 1000] > dl[dl < 1000]).mean()) if (dl < 1000).any() else float("nan"),
        urg_sub100=float(100 * (lat[urg] < 100).mean()),
        priv_offsite=int((pi[cloud] == 1).sum()), priv_total=int(pi.sum()),
        wan_mb=float(rho[cloud].sum() / 1024.0),
        frac_cloud=float(cloud.mean()),
    )
    # Lemma 2 check on fog high-priority tasks dispatched directly or by preemption
    hp, ok, ok_u, n_le, n_pre_after = 0, 0, 0, 0, 0
    nodes = sim.nodes
    kmin = min(n.cpu_mips for n in nodes.values())
    lmax = max(n.latency_ms for n in nodes.values())
    for j in d:
        if j.node < 0 or j.psi < sim.psi_thresh or j.tid not in rec:
            continue
        if j.wait_ms > 1e-9:   # excluded case (re-enqueued) -> no bound
            continue
        if getattr(j, 'preempted', False):   # suspended after dispatch -> outside hypotheses
            n_pre_after = locals().get('n_pre_after', 0) + 1
            continue
        lam, nid, _ = rec[j.tid]
        n = nodes[nid]
        up = j.rho_kb / B_GF_KBPS * 1000.0
        bound = up + n.latency_ms + j.cpu / (n.cpu_mips * (1 - lam)) * 1000.0 + 0.0
        # simulator: done fires at start + exec + latency (latency covers both hops)
        hp += 1
        ok += j.latency_ms <= bound + n.latency_ms + 1e-6
        n_le += lam <= 0.80
        bu = up + 2 * lmax + j.cpu / (kmin * (1 - 0.80)) * 1000.0
        ok_u += (j.latency_ms <= bu + 1e-6) or (lam > 0.80)
    hp_all = sum(1 for j in d if j.node >= 0 and j.psi >= sim.psi_thresh)
    out.update(l2_pre_after=n_pre_after, l2_hp_all=hp_all, l2_hp=hp, l2_ok=ok, l2_lam_le=n_le, l2_uniform_ok=ok_u)
    # Corollary 3: measured utilisation vs bound
    fog = [j for j in d if j.node >= 0]
    if fog:
        T = sim.t
        kmax = max(n.cpu_mips for n in nodes.values())
        N = len(nodes)
        meas = np.mean([sum(j.cpu / nodes[j.node].cpu_mips * j.exec_ms
                            for j in fog if j.node == nid) / T for nid in nodes])
        lb = sum(j.cpu * j.exec_ms for j in fog) / (N * kmax * T)
        lb2 = sum(j.cpu ** 2 / kmax * 1000.0 for j in fog) / (N * kmax * T)
        busy = np.mean([min(1.0, n.busy_ms / max(T, 1e-9)) for n in nodes.values()])
        blb = sum(j.cpu / kmax * 1000.0 for j in fog) / (N * T)
        out.update(c3_meas=float(meas), c3_lb=float(lb), c3_lb2=float(lb2),
                   c3_busy=float(busy), c3_busy_lb=float(blb))
    return out


if __name__ == "__main__":
    res = {"main": {}, "csweep": {}}
    for sysn in ("Cloud-Only", "CF-SAF", "HEFT-Fog", "Threshold-Fog"):
        res["main"][sysn] = [per_task_stats(*run(sysn, s)) for s in SEEDS]
        r = res["main"][sysn]
        print(sysn, {k: round(float(np.mean([x[k] for x in r])), 3) for k in r[0]}, flush=True)
    for c in (26, 28, 30, 32, 36, 43, 60, 1000):
        res["csweep"][c] = {}
        for sysn in ("Cloud-Only", "CF-SAF"):
            res["csweep"][c][sysn] = [per_task_stats(*run(sysn, s, c=c)) for s in SEEDS]
        a = np.mean([x["mean"] for x in res["csweep"][c]["Cloud-Only"]])
        b = np.mean([x["mean"] for x in res["csweep"][c]["CF-SAF"]])
        au = np.mean([x["urg_mean"] for x in res["csweep"][c]["Cloud-Only"]])
        bu = np.mean([x["urg_mean"] for x in res["csweep"][c]["CF-SAF"]])
        print(f"c={c:5d} cloud={a:9.1f} cfsaf={b:9.1f}  urg: cloud={au:9.1f} cfsaf={bu:9.1f}", flush=True)
    json.dump(res, open(_os.path.join(_HERE, 'results', 'cloud_results240.json'), "w"))
