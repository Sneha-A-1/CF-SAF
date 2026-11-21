import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
import sys, os, pickle, numpy as np, pandas as pd, yaml, importlib
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'src', 'utils'), _HERE]
import paths  # noqa
m = importlib.import_module('run_experiments_real')
from src.simulation.simulator import Simulator, Job, build_nodes  # noqa
cfg = yaml.safe_load(open(os.path.join(paths.ROOT, 'config', 'config.yaml')))
FULL = pd.read_csv(paths.DATA)
GBT = pickle.load(open(paths.GBT, 'rb'))
SEEDS = [42 + 13 * k for k in range(10)]
RATE, NT = 240, 12000


def load_jobs(df, rate, rng, gaps=None):
    if gaps is None:
        gaps = rng.exponential(1000.0 / rate, len(df))
    arr = np.cumsum(gaps)
    jobs = []
    for k, r in enumerate(df.itertuples(index=False)):
        j = Job(tid=int(r.task_id), tau=float(r.tau), phi=float(r.phi),
                delta_s=float(r.delta_s), rho_kb=float(r.rho_kb),
                cpu=float(r.cpu_demand_mips), mem=float(r.mem_demand_mb),
                psi=float(r.psi), label=int(r.label), pi=int(r.pi),
                arrival_ms=float(arr[k]))
        j.device = int(r.device_id)
        jobs.append(j)
    return jobs


def router(kind='idcra', theta=0.22, gbt=None):
    return m.make_router(kind, gbt=gbt or GBT, theta=theta)


def summarise(runs, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out[k] = (float('nan'), float('nan'))
            continue
        from scipy import stats
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        out[k] = (float(v.mean()), float(h))
    return out
