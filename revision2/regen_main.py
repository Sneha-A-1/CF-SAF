import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
"""Regenerate the main comparison (Tables VII-VIII, Figs 2-6) with the
corrected simulator at the corrected operating point."""
import sys, json, os, time
sys.path.insert(0, _HERE)
from common import *  # noqa
from simfix import SimFix

OP = 240
SPECS = {'Cloud-Only': ('all_cloud', 'fcfs'), 'Naive Fog': ('all_fog', 'rr'),
         'Threshold-Fog': ('thresh', 'fcfs'), 'HEFT-Fog': ('idcra', 'heft'),
         'CF-SAF': ('idcra', 'orsaf')}
STORE = os.environ.get("STORE", _os.path.join(_HERE, 'results', 'main_results.json'))


def one(nm, seed, rate=OP, n=NT, gaps=None, df=None, simkw=None, rkw=None):
    rk, sc = SPECS[nm]
    rng = np.random.default_rng(seed)
    d = FULL.head(n) if df is None else df
    jobs = load_jobs(d, rate, rng, gaps=None if gaps is None else gaps(rng, len(d)))
    sim = SimFix(build_nodes(cfg), scheduler=sc, **(simkw or {}))
    sim.run(jobs, m.make_router(rk, gbt=GBT, theta=0.22, **(rkw or {})))
    mt = sim.metrics()
    lat = np.array([j.latency_ms for j in sim.done])
    urg = np.array([j.label == 1 for j in sim.done])
    mt['per_node_util'] = [100 * min(1, x.busy_ms / max(sim.t, 1e-9)) for x in sim.nodes.values()]
    mt['urg_mean_ms'] = float(lat[urg].mean())
    mt['urg_p95_ms'] = float(np.percentile(lat[urg], 95))
    mt['non_mean_ms'] = float(lat[~urg].mean())
    mt['dl_miss_pct'] = float(100 * np.mean([j.latency_ms > 1000 * j.delta_s for j in sim.done]))
    mt['priv_offsite'] = int(sum(1 for j in sim.done if j.node == -1 and j.pi == 1))
    mt['wan_mb'] = float(sum(j.rho_kb for j in sim.done if j.node == -1) / 1024)
    mt['urg_sub100_pct'] = float(100 * (lat[urg] < 100).mean())
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in mt.items()}, lat


if __name__ == '__main__':
    res = json.load(open(STORE)) if os.path.exists(STORE) else {}
    for nm in sys.argv[1:] or list(SPECS):
        runs = res.get(nm, [])
        for k in range(len(runs), 10):
            t0 = time.time()
            mt, lat = one(nm, SEEDS[k])
            if k == 0:
                mt['lat_sample'] = [float(x) for x in lat]
            runs.append(mt)
            res = json.load(open(STORE)) if os.path.exists(STORE) else {}
            res[nm] = runs
            json.dump(res, open(STORE, 'w'))
            print(f"{nm} seed {k} L={mt['latency_ms']:.1f} ({time.time()-t0:.0f}s)", flush=True)
