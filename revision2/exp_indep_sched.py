import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
"""Scheduling on independent / real workloads (Reviewer 2, comment f).
W5 heavy-tailed CPU demand, W6 bursty MMPP arrivals, R real farm trace replay.
The GBT used for routing is the one trained on the original synthetic data;
nothing is retrained on these workloads."""
import sys, json
sys.path.insert(0, _HERE)
from regen_main import *  # noqa
from workloads import gen, mmpp_gaps, real_tasks, real_replay

W5 = gen(n=NT, seed=7, cpu_mode="pareto")
BASE_REAL = real_tasks(seed=11)
res = {}
SYS = ('Cloud-Only', 'Threshold-Fog', 'HEFT-Fog', 'CF-SAF')
KEEP = ['latency_ms', 'p95_ms', 'urg_mean_ms', 'urg_p95_ms', 'energy_j', 'utilisation',
        'dl_miss_pct', 'priv_offsite', 'n_fog']
for wl in ('W5 heavy-tailed CPU', 'W6 bursty MMPP (mean 200/s)', 'R real farm trace replay'):
    res[wl] = {}
    for nm in SYS:
        runs = []
        for s in SEEDS:
            if wl.startswith('W5'):
                mt, _ = one(nm, s, df=W5)
            elif wl.startswith('W6'):
                mt, _ = one(nm, s, rate=200, gaps=mmpp_gaps(200, burst=1.5))
            else:
                d, gaps = real_replay(NT, OP, s, base=BASE_REAL)
                mt, _ = one(nm, s, df=d, gaps=lambda rng, n, g=gaps: g)
            runs.append({k: mt[k] for k in KEEP})
        res[wl][nm] = runs
        print(wl, nm, {k: round(float(np.mean([r[k] for r in runs])), 2) for k in KEEP}, flush=True)
    json.dump(res, open(_os.path.join(_HERE, 'results', 'indep_sched.json'), 'w'))
print('ALL DONE')
