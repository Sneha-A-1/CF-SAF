import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
"""Experiments that activate ORSA-F load rebalancing (Reviewer 2, comment c)."""
import json, sys, time
import numpy as np
from common import *  # noqa
from sim_ext import SimExt

SMALL = [0, 1, 2, 3, 4]       # class A (0,1) and class B (2,3,4)


def affinity_map(jobs, rng, nodes, hot_frac, a_frac):
    """Gateway affinity: device -> gateway -> home node.  A fraction
    hot_frac of devices are attached to gateways on the small nodes.
    A fraction a_frac of tasks are locality-bound, provided their home
    node can host them at all."""
    ndev = 1200
    hot = rng.random(ndev) < hot_frac
    home = np.where(hot, rng.choice(SMALL, ndev),
                    rng.choice([5, 6, 7, 8, 9], ndev))
    caps = {n.node_id: (n.cpu_mips, n.mem_mb) for n in nodes}
    aff = {}
    for j in jobs:
        h = int(home[j.device])
        if rng.random() < a_frac and j.cpu <= caps[h][0] and j.mem <= caps[h][1]:
            aff[j.tid] = h
    return aff


SCEN = {
    # name: (hot_frac, a_frac, background, fail)
    "S0 main config":      (None, 0.0, None, None),
    "S1 hotspot affinity": (0.8, 0.5, None, None),
    "S2 background load":  (0.8, 0.5, {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45, 4: 0.45}, None),
    "S3 node failure":     (0.5, 0.5, None, (6, 15000.0, 35000.0)),
    "S1m hotspot, moderate": (0.8, 0.25, None, None),
    "S4 H1 violated":      (0.8, 0.5, {0: 0.85, 1: 0.45, 2: 0.45, 3: 0.45, 4: 0.45}, None),
}

KEYS = ["latency_ms", "p95_ms", "p99_ms", "urg_mean_ms", "urg_p95_ms",
        "dl_miss_pct", "energy_j", "over_tick_pct", "migrations", "invocations",
        "inv_trigger_noeligible", "inv_conv_pct", "inv_cond_pct",
        "conv_given_cond", "conv_given_notcond", "phi_nonneg_steps",
        "mig_steps", "phi_red_pct", "remig", "max_mig_per_inv", "utilisation"]


def run(scen, mode, seed, track=False):
    hot, a, bg, fail = SCEN[scen]
    rng = np.random.default_rng(seed)
    df = FULL.head(NT)
    jobs = load_jobs(df, RATE, rng)
    nodes = build_nodes(cfg)
    aff = {} if hot is None else affinity_map(jobs, np.random.default_rng(seed + 1),
                                              nodes, hot, a)
    sim = SimExt(nodes, scheduler="orsaf", rb_mode=mode, affinity=aff,
                 background=bg, fail=fail, track=track)
    sim.run(jobs, router())
    out = sim.ext_metrics()
    out["n_affinity"] = len(aff)
    return out, sim


if __name__ == "__main__":
    which = sys.argv[1:] or list(SCEN)
    res = {}
    try:
        res = json.load(open(_os.path.join(_HERE, 'results', 'rebalance_results240.json')))
    except Exception:
        pass
    for sc in which:
        res[sc] = {}
        for mode in ("off", "guarded", "unguarded"):
            t0 = time.time()
            runs = [run(sc, mode, s)[0] for s in SEEDS]
            res[sc][mode] = runs
            sm = summarise(runs, KEYS)
            print(f"{sc:<20} {mode:<9} L={sm['latency_ms'][0]:9.1f}±{sm['latency_ms'][1]:6.1f} "
                  f"P95={sm['p95_ms'][0]:9.1f} urgP95={sm['urg_p95_ms'][0]:9.1f} "
                  f"dl={sm['dl_miss_pct'][0]:5.2f}% over={sm['over_tick_pct'][0]:5.1f}% "
                  f"mig={sm['migrations'][0]:7.1f} inv={sm['invocations'][0]:6.1f} "
                  f"conv={sm['inv_conv_pct'][0]:5.1f}% cond={sm['inv_cond_pct'][0]:5.1f}% "
                  f"c|C={sm['conv_given_cond'][0]:5.1f} c|~C={sm['conv_given_notcond'][0]:5.1f} "
                  f"dPhi>=0:{sm['phi_nonneg_steps'][0]:.1f}/{sm['mig_steps'][0]:.1f} "
                  f"phiRed={sm['phi_red_pct'][0]:.1f}% remig={sm['remig'][0]:.1f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(_os.path.join(_HERE, 'results', 'rebalance_results240.json'), "w"))
