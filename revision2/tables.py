import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
"""Compute every number used in the revised Section VII from the stored
experiment outputs, and print them (and a JSON of key values)."""
import json, os, numpy as np
from scipy import stats

E = _os.path.join(_HERE, 'results') + _os.sep
J = lambda f: json.load(open(E + f)) if os.path.exists(E + f) else None
main = J('main_results.json') or {}
nv = J('naive_results.json')
if nv and len(nv.get('Naive Fog', [])) >= 2:
    main['Naive Fog'] = nv['Naive Fog']
rest = J('rest_results.json') or {}
K = {}


def arr(runs, k): return np.array([r[k] for r in runs], float)


def ci(v):
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean() - h, v.mean() + h


ORDER = ['Cloud-Only', 'Naive Fog', 'Threshold-Fog', 'HEFT-Fog', 'CF-SAF']
print('=== Table VII (main, 240/s) ===')
for nm in ORDER:
    if nm not in main: continue
    r = main[nm]; L = arr(r, 'latency_ms'); lo, hi = ci(L)
    u = np.mean([np.mean(x['per_node_util']) for x in r]) if nm != 'Cloud-Only' else float('nan')
    row = dict(L=L.mean(), lo=lo, hi=hi, sd=L.std(ddof=1), E=arr(r, 'exec_ms').mean(), U=u,
               T=arr(r, 'throughput').mean(), En=arr(r, 'energy_j').mean(), P95=arr(r, 'p95_ms').mean(),
               urgL=arr(r, 'urg_mean_ms').mean(), urgP95=arr(r, 'urg_p95_ms').mean(),
               dl=arr(r, 'dl_miss_pct').mean(), priv=arr(r, 'priv_offsite').mean(),
               wan=arr(r, 'wan_mb').mean(), sub100=arr(r, 'urg_sub100_pct').mean(),
               nfog=arr(r, 'n_fog').mean())
    K['main_' + nm] = row
    print(f"{nm:<14}" + " ".join(f"{k}={v:.3f}" for k, v in row.items()))
print('\n=== Table VIII (paired t, latency, 10 seeds) ===')
if 'CF-SAF' in main:
    cf = arr(main['CF-SAF'], 'latency_ms')
    for nm in ORDER[:-1]:
        if nm not in main: continue
        n_ = len(main[nm]); b = arr(main[nm], 'latency_ms'); p = stats.ttest_rel(cf[:n_], b).pvalue
        K['t_' + nm] = dict(d=100 * (cf[:n_].mean() - b.mean()) / b.mean(), p=p)
        cu, bu = arr(main['CF-SAF'], 'urg_p95_ms')[:n_], arr(main[nm], 'urg_p95_ms')
        cm, bm = arr(main['CF-SAF'], 'urg_mean_ms')[:n_], arr(main[nm], 'urg_mean_ms')
        K['t_' + nm].update(urgP95_d=100 * (cu.mean() - bu.mean()) / bu.mean(), urgP95_p=stats.ttest_rel(cu, bu).pvalue,
                            urgL_d=100 * (cm.mean() - bm.mean()) / bm.mean(), urgL_p=stats.ttest_rel(cm, bm).pvalue)
        print(nm, {k: (f"{v:.3g}") for k, v in K['t_' + nm].items()})

if 'power' in rest:
    print('\n=== Table IX (30 seeds) ===')
    P = rest['power']; a, b = P['CF-SAF'], P['HEFT-Fog']
    out = {}
    for k in ('latency_ms', 'exec_ms', 'energy_j', 'p95_ms', 'urg_mean_ms', 'urg_p95_ms', 'utilisation', 'throughput'):
        x, y = arr(a, k), arr(b, k)
        out[k] = dict(cf=x.mean(), heft=y.mean(), d=100 * (x.mean() - y.mean()) / y.mean(), p=stats.ttest_rel(x, y).pvalue)
    x, y = arr(a, 'latency_ms'), arr(b, 'latency_ms')
    out['ci_cf'] = ci(x); out['ci_heft'] = ci(y); out['sd'] = (x.std(ddof=1), y.std(ddof=1))
    dd = y - x; out['dz'] = dd.mean() / dd.std(ddof=1); out['wil'] = stats.wilcoxon(x, y).pvalue
    out['wins'] = int((x < y).sum())
    K['power'] = out
    for k, v in out.items(): print(k, v)

for reg in ('ablation_poisson240', 'ablation_mmpp200', 'ablation_mmpp240'):
    if reg not in rest: continue
    print(f'\n=== {reg} ===')
    A = rest[reg]; base = A['CF-SAF (full)']
    K[reg] = {}
    for v, runs in A.items():
        L = arr(runs, 'latency_ms'); bL = arr(base, 'latency_ms')
        p = stats.ttest_rel(L, bL).pvalue if v != 'CF-SAF (full)' and np.any(L != bL) else float('nan')
        row = dict(L=L.mean(), dL=100 * (L.mean() - bL.mean()) / bL.mean(), p=p,
                   U=arr(runs, 'utilisation').mean(), P95=arr(runs, 'p95_ms').mean(),
                   urgP95=arr(runs, 'urg_p95_ms').mean(), pre=arr(runs, 'preemptions').mean(),
                   durgP95=100 * (arr(runs, 'urg_p95_ms').mean() - arr(base, 'urg_p95_ms').mean()) / arr(base, 'urg_p95_ms').mean(),
                   purg=(stats.ttest_rel(arr(runs, 'urg_p95_ms'), arr(base, 'urg_p95_ms')).pvalue
                         if v != 'CF-SAF (full)' and np.any(arr(runs, 'urg_p95_ms') != arr(base, 'urg_p95_ms')) else float('nan')))
        K[reg][v] = row
        print(f"{v:<22}" + " ".join(f"{k}={x:.4g}" for k, x in row.items()))

if 'theta' in rest:
    K['theta'] = {k: float(np.mean(v)) for k, v in rest['theta'].items()}
    print('\ntheta', K['theta'])
if 'scal' in rest:
    K['scal'] = {nm: {r: float(np.mean(v)) for r, v in d.items()} for nm, d in rest['scal'].items()}
    print('\nscal', K['scal'])
st = J('stab_fix.json')
if st:
    K['stab'] = {r: dict(L4=float(np.mean(v['L4k'])), L12=float(np.mean(v['L12k'])),
                         drift=float(100 * (np.mean(v['L12k']) - np.mean(v['L4k'])) / np.mean(v['L4k'])),
                         maxseed=float(np.max(100 * (np.array(v['L12k']) - np.array(v['L4k'])) / np.array(v['L4k']))),
                         nunst=int(np.sum(100 * (np.array(v['L12k']) - np.array(v['L4k'])) / np.array(v['L4k']) > 5)),
                         U=float(np.mean(v['U12k']))) for r, v in st.items()}
    print('\nstab'); [print(r, v) for r, v in K['stab'].items()]
json.dump(K, open(E + 'K.json', 'w'), indent=1, default=float)
