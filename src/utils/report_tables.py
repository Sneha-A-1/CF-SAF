"""
Prints the paper's scheduling tables from stored sweep results.
Run run_full_sweep.py and run_30seed_power.py first.

    python src/utils/report_tables.py
"""
import json, os, numpy as np
from scipy import stats
import paths

ORDER = ['Cloud-Only','Naive Fog','Threshold-Fog','HEFT-Fog','CF-SAF']

def main():
    fp = os.path.join(paths.RESULTS,'full_results.json')
    if not os.path.exists(fp):
        print("Missing results/full_results.json — run run_full_sweep.py first.")
        return
    res = json.load(open(fp))
    have = [k for k in ORDER if k in res]
    def c(n,k): return np.array([r[k] for r in res[n]])
    def ut(n):  return np.array([np.mean(r['per_node_util']) for r in res[n]])

    n = len(res[have[0]])
    print(f"\nTABLE IX  —  rate 270 tasks/s, 12,000 tasks/run, {n} seeds\n")
    hdr = (f"{'System':<16}{'L (ms)':>10}{'95% CI':>20}{'SD':>8}"
           f"{'E (ms)':>9}{'eta%':>7}{'Tput':>8}{'J/task':>9}{'P95':>10}")
    print(hdr); print('-'*len(hdr))
    for nm in have:
        l = c(nm,'latency_ms')
        ci = stats.t.interval(.95, len(l)-1, l.mean(), stats.sem(l)+1e-12)
        u = np.nanmean(ut(nm)); us = '  N/A' if u < .01 else f"{u:6.1f}"
        print(f"{nm:<16}{l.mean():>10.2f}"
              f"{f'({ci[0]:.1f}, {ci[1]:.1f})':>20}{l.std(ddof=1):>8.2f}"
              f"{c(nm,'exec_ms').mean():>9.2f}{us:>7}"
              f"{c(nm,'throughput').mean():>8.1f}"
              f"{c(nm,'energy_j').mean():>9.3f}{c(nm,'p95_ms').mean():>10.1f}")

    print(f"\nTABLE X  —  paired t-tests, CF-SAF vs each baseline (latency)\n")
    cf = c('CF-SAF','latency_ms')
    for nm in have:
        if nm == 'CF-SAF': continue
        b = c(nm,'latency_ms'); t,p = stats.ttest_rel(cf,b)
        d = 100*(b.mean()-cf.mean())/b.mean()
        tag = ('significant (p<0.001)' if p < .001 else
               'p<0.05' if p < .05 else 'not significant')
        print(f"  vs {nm:<16}{d:>+8.1f}%   p={p:.2e}   {tag}")

    # Table XI — 30-seed power comparison
    pp = os.path.join(paths.RESULTS,'heft_30seed_raw.json')
    if os.path.exists(pp):
        d30 = json.load(open(pp))
        ks = sorted(d30['CF-SAF'].keys(), key=int)
        if len(ks) >= 10:
            g = lambda nm,f: np.array([d30[nm][k][f] for k in ks])
            a,b = g('CF-SAF','latency_ms'), g('HEFT-Fog','latency_ms')
            t,p = stats.ttest_rel(a,b); w = stats.wilcoxon(a,b)
            dz = (b-a).mean()/(b-a).std(ddof=1)
            print(f"\nTABLE XI  —  CF-SAF vs HEFT-Fog, {len(ks)} paired seeds\n")
            print(f"  Mean latency   CF-SAF {a.mean():8.2f}   "
                  f"HEFT {b.mean():8.2f}   p={p:.2e}")
            for f,lab in [('exec_ms','Execution   '),('energy_j','Energy      '),
                          ('p95_ms','P95 latency '),('util','Utilisation '),
                          ('throughput','Throughput  ')]:
                x,y = g('CF-SAF',f), g('HEFT-Fog',f)
                _,q = stats.ttest_rel(x,y)
                print(f"  {lab}   CF-SAF {x.mean():8.3f}   "
                      f"HEFT {y.mean():8.3f}   p={q:.2e}")
            print(f"\n  Reduction {100*(b.mean()-a.mean())/b.mean():.1f}%   "
                  f"Cohen's dz={dz:.3f}   Wilcoxon p={w.pvalue:.2e}   "
                  f"CF-SAF lower in {int((a<b).sum())}/{len(ks)} seeds")
    print()

if __name__ == '__main__':
    main()
