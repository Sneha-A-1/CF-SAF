import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
"""Remaining regenerated results: 30-seed power, ablation, sensitivity,
scalability.  Writes rest_results.json."""
import sys, json, os, time
sys.path.insert(0, _HERE)
from regen_main import *  # noqa
from workloads import mmpp_gaps
OUT = _os.path.join(_HERE, 'results', 'rest_results.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, 'w'))
KEEP = ['latency_ms', 'p95_ms', 'exec_ms', 'utilisation', 'throughput', 'energy_j',
        'urg_mean_ms', 'urg_p95_ms', 'preemptions', 'dl_miss_pct']
def slim(mt): return {k: mt[k] for k in KEEP}

if 'power' not in res:
    res['power'] = {}
    for nm in ('CF-SAF', 'HEFT-Fog'):
        res['power'][nm] = [slim(one(nm, 1000 + 7 * k)[0]) for k in range(30)]
    save(); print('power done', flush=True)

VAR = {'CF-SAF (full)': ({}, {}), 'A1: no neural calib.': ({}, {'calib': False}),
       'A2: fixed mu=0.5': ({}, {'fixed_mu': 0.5}), 'A3: no preemption': ({'preemption': False}, {}),
       'A4: no rebalancing': ({'rebalancing': False}, {})}
for regime, gaps, rate in (('poisson240', None, 240), ('mmpp200', mmpp_gaps(200, burst=1.5), 200)):
    key = 'ablation_' + regime
    if key in res: continue
    res[key] = {}
    for v, (skw, rkw) in VAR.items():
        res[key][v] = [slim(one('CF-SAF', s, rate=rate, gaps=gaps, simkw=skw, rkw=rkw)[0]) for s in SEEDS]
    if regime == 'mmpp200':
        for nm in ('HEFT-Fog', 'Cloud-Only', 'Threshold-Fog'):
            res[key][nm] = [slim(one(nm, s, rate=rate, gaps=gaps)[0]) for s in SEEDS]
    save(); print(key, 'done', flush=True)

if 'theta' not in res:
    res['theta'] = {}
    for th in (0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.50, 0.55, 0.65, 0.80):
        L = []
        for s in SEEDS[:5]:
            rng = np.random.default_rng(s); jobs = load_jobs(FULL.head(NT), OP, rng)
            sim = SimFix(build_nodes(cfg), scheduler='orsaf'); sim.run(jobs, m.make_router('idcra', gbt=GBT, theta=th))
            L.append(sim.metrics()['latency_ms'])
        res['theta'][str(th)] = L
        print('theta', th, np.mean(L), flush=True)
    save()

if 'scal' not in res:
    res['scal'] = {}
    for nm in ('Cloud-Only', 'Threshold-Fog', 'HEFT-Fog', 'CF-SAF'):
        res['scal'][nm] = {}
        for rate in (100, 150, 200, 220, 240, 260, 280, 300):
            res['scal'][nm][str(rate)] = [one(nm, s, rate=rate)[0]['latency_ms'] for s in SEEDS[:5]]
            print('scal', nm, rate, np.mean(res['scal'][nm][str(rate)]), flush=True)
            save()
print('ALL DONE', flush=True)
