import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
"""Regenerate Figs 2-10 for the revised manuscript from stored results."""
import sys, json, os, pickle
sys.path.insert(0, _HERE); sys.path.insert(0, _os.path.join(_REPO, 'data', 'synthetic'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = _os.path.join(_HERE, 'figures'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9.5,
                     'axes.grid': True, 'grid.linestyle': '-', 'grid.alpha': 0.35,
                     'grid.color': '#d9d9d6', 'axes.edgecolor': '#8a8a85',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 2.0, 'lines.markersize': 7})
INK, MUTED = '#2b2b29', '#6b6b66'
ORDER = ['Cloud-Only', 'Naive Fog', 'Threshold-Fog', 'HEFT-Fog', 'CF-SAF']
COL = {'CF-SAF': '#2a78d6', 'HEFT-Fog': '#eb6834', 'Cloud-Only': '#1baf7a',
       'Threshold-Fog': '#eda100', 'Naive Fog': '#e87ba4'}
MK = {'CF-SAF': 'o', 'HEFT-Fog': 's', 'Cloud-Only': 'D', 'Threshold-Fog': '^', 'Naive Fog': 'v'}
E = _os.path.join(_HERE, 'results') + _os.sep
main = json.load(open(E + 'main_results.json'))
main['Naive Fog'] = json.load(open(E + 'naive_results.json'))['Naive Fog']
rest = json.load(open(E + 'rest_results.json'))
st = json.load(open(E + 'stab_fix.json'))
mean = lambda nm, k: float(np.mean([r[k] for r in main[nm]]))


def save(f, name):
    f.savefig(f'{OUT}/{name}', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(f)


def bars(ax, vals, fmt, log=False):
    x = np.arange(len(ORDER))
    ax.bar(x, vals, 0.62, color=[COL[k] for k in ORDER], edgecolor='white', linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v * (1.08 if log else 1.0) + (0 if log else max(vals) * 0.015), fmt.format(v),
                ha='center', va='bottom', fontsize=9.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=20, ha='right')
    if log: ax.set_yscale('log')
    ax.grid(axis='x', visible=False)


# Fig 2 mean latency
f, a = plt.subplots(figsize=(8, 4.4))
v = [mean(k, 'latency_ms') for k in ORDER]
bars(a, v, '{:.1f}', log=True); a.set_ylim(100, max(v) * 3)
a.set_ylabel(r'Mean task latency $\bar{L}$ (ms, log scale)')
save(f, 'Figure2.png')

# Fig 3 normalised metrics as small multiples
f, axs = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
for a, (k, lab) in zip(axs, [('latency_ms', 'Latency'), ('exec_ms', 'Execution time'), ('energy_j', 'Energy per task')]):
    base = mean('Cloud-Only', k); vv = [mean(n, k) / base for n in ORDER]
    bars(a, vv, '{:.2f}', log=True); a.axhline(1, color=MUTED, lw=1, ls=':')
    a.set_title(lab, fontsize=11, color=INK)
axs[0].set_ylabel('Relative to Cloud-Only (log scale)'); axs[0].set_ylim(0.3, 1000)
save(f, 'Figure_3.png')

# Fig 4 CDF
f, a = plt.subplots(figsize=(8, 4.4))
for k in ORDER:
    d = np.sort(main[k][0]['lat_sample']); y = np.arange(1, len(d) + 1) / len(d)
    a.plot(d, y, color=COL[k], lw=2.6 if k == 'CF-SAF' else 1.8, label=k)
a.set_xscale('log'); a.set_xlim(50, 3e5)
a.set_xlabel('Task latency $L$ (ms, log scale)'); a.set_ylabel('Cumulative probability')
a.legend(loc='lower right', frameon=False)
save(f, 'Figure4.png')

# Fig 5 per-node utilisation (fog systems)
show = ['Naive Fog', 'Threshold-Fog', 'HEFT-Fog', 'CF-SAF']
f, a = plt.subplots(figsize=(11, 4.4)); x = np.arange(10); w = 0.2
for n, k in enumerate(show):
    u = np.mean([r['per_node_util'] for r in main[k]], axis=0)
    a.bar(x + (n - 1.5) * w, u, w, color=COL[k], edgecolor='white', linewidth=1.2, label=f'{k} (mean {u.mean():.0f}%)')
a.set_xticks(x); a.set_xticklabels(['A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2'])
a.set_xlabel('Fog node (hardware class and index)'); a.set_ylabel(r'Utilisation $\eta_i$ (%)')
a.set_ylim(0, 108); a.grid(axis='x', visible=False)
a.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, -.16), frameon=False)
save(f, 'Figure5.png')

# Fig 6 energy and throughput, two panels (no dual axis)
f, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
bars(a, [mean(k, 'energy_j') for k in ORDER], '{:.2f}', log=True)
a.set_ylabel(r'Energy $\mathcal{E}$ (J/task, log scale)'); a.set_ylim(5, 200)
bars(b, [mean(k, 'throughput') for k in ORDER], '{:.1f}')
b.set_ylabel(r'Throughput $\Theta$ (tasks/s)'); b.set_ylim(0, 290)
save(f, 'Figure6.png')

# Fig 7 scalability
S = rest['scal']
f, a = plt.subplots(figsize=(8, 4.4))
for k in ['Cloud-Only', 'Threshold-Fog', 'HEFT-Fog', 'CF-SAF']:
    rates = sorted(int(r) for r in S[k]); yv = [np.mean(S[k][str(r)]) for r in rates]
    a.plot(rates, yv, marker=MK[k], color=COL[k], label=k, lw=2.6 if k == 'CF-SAF' else 1.8)
a.axvline(240, color=MUTED, ls=':', lw=1.4)
a.text(243, a.get_ylim()[1] * 0.6 if False else 0, '')
a.set_yscale('log'); a.set_xlabel('Arrival rate (tasks/s)'); a.set_ylabel(r'Mean latency $\bar{L}$ (ms, log scale)')
yl = a.get_ylim(); a.text(237, yl[1] * 0.55, 'stability\nlimit (240/s)', fontsize=9, color=MUTED, ha='right')
a.legend(loc='upper left', frameon=False)
save(f, 'Figure7.png')

# Fig 8 threshold sensitivity: F1 and latency as two panels
import pandas as pd
from generate_dataset import grouped_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
df = pd.read_csv(_os.path.join(_REPO, 'data', 'synthetic', 'tasks.csv')); tr, va, te = grouped_split(df, 42)
FE = ['tau', 'sigma', 'delta_s', 'pi', 'rho_kb', 'phi']
g = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
                               min_samples_leaf=5, random_state=42).fit(tr[FE].values, tr.label.values)
sv = g.predict_proba(va[FE].values)[:, 1]
ths = sorted(float(t) for t in rest['theta'])
f1s = [f1_score(va.label.values, (sv >= t).astype(int)) for t in ths]
lats = [np.mean(rest['theta'][str(t) if str(t) in rest['theta'] else repr(t)]) for t in ths]
json.dump({'theta': ths, 'f1': f1s, 'lat': lats}, open(E + 'theta_curve.json', 'w'))
f, (a, b) = plt.subplots(1, 2, figsize=(12, 4.0))
a.plot(ths, f1s, marker='o', color=COL['CF-SAF']); a.set_ylabel('Urgent-class $F_1$ (validation)')
b.plot(ths, lats, marker='o', color=COL['CF-SAF']); b.set_ylabel(r'Mean latency $\bar{L}$ (ms)')
for ax in (a, b):
    ax.axvline(0.22, color=MUTED, ls=':', lw=1.4); ax.set_xlabel(r'Routing threshold $\theta^*$')
    ax.text(0.235, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), r'$\theta^*=0.22$', fontsize=9.5, color=MUTED)
save(f, 'Figure8.png')

# Fig 9 stability
rates = sorted(int(r) for r in st)
f, a = plt.subplots(figsize=(8, 4.3))
for key, lab, c, mk in (('L4k', '4,000 tasks/run', '#86b6ef', 'o'), ('L12k', '12,000 tasks/run', '#1c5cab', 's')):
    m_ = [np.mean(st[str(r)][key]) for r in rates]
    lo = [np.min(st[str(r)][key]) for r in rates]; hi = [np.max(st[str(r)][key]) for r in rates]
    a.plot(rates, m_, marker=mk, color=c, label=lab + ' (mean of 10 seeds)')
    a.fill_between(rates, lo, hi, color=c, alpha=0.15, lw=0)
a.axvline(240, color=MUTED, ls=':', lw=1.4)
a.text(236, 900, 'highest stable\nrate: 240/s', ha='right', fontsize=9.5, color=INK)
a.set_yscale('log'); a.set_xlabel('Arrival rate (tasks/s)'); a.set_ylabel(r'CF-SAF mean latency $\bar{L}$ (ms, log)')
a.legend(loc='upper left', frameon=False)
save(f, 'Figure9.png')

# Fig 10 rebalancing: (a) max committed load trace under node failure; (b) latency by scenario
from exp_rebalance import run as rb_run, SCEN
R = json.load(open(E + 'rebalance_results240.json'))
f, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={'width_ratios': [1.15, 1]})
MC = {'off': '#8a8a85', 'guarded': '#2a78d6', 'unguarded': '#eb6834'}
ML = {'off': 'no rebalancing', 'guarded': 'guarded (M1)', 'unguarded': 'unguarded'}
for mode in ('off', 'unguarded', 'guarded'):
    _, sim = rb_run('S1 hotspot affinity', mode, 42, track=True)
    t = np.array([x[0] for x in sim.trace]) / 1000; ph = np.array([x[1] for x in sim.trace])
    a.plot(t, np.maximum(ph, 1e-4), color=MC[mode], lw=1.6, label=ML[mode])
a.set_yscale('log'); a.set_xlabel('Time (s)'); a.set_ylabel(r'Potential $\Phi$ before each invocation (log)')
a.set_title('(a) Hotspot affinity (S1), one seed', fontsize=10.5, color=INK)
a.legend(frameon=False, loc='upper left')
sc = [s for s in ['S1 hotspot affinity', 'S1m hotspot, moderate', 'S2 background load', 'S3 node failure', 'S4 H1 violated'] if s in R]
x = np.arange(len(sc)); w = 0.26
for n, mode in enumerate(('off', 'unguarded', 'guarded')):
    yv = [np.mean([r['latency_ms'] for r in R[s][mode]]) for s in sc]
    b.bar(x + (n - 1) * w, yv, w, color=MC[mode], edgecolor='white', linewidth=1.2, label=ML[mode])
b.set_yscale('log'); b.set_xticks(x); b.set_xticklabels([s.split(' ')[0] for s in sc])
b.set_ylabel(r'Mean latency $\bar{L}$ (ms, log)'); b.grid(axis='x', visible=False)
b.set_title('(b) Mean latency by scenario, ten seeds', fontsize=10.5, color=INK)
b.legend(frameon=False, loc='upper right')
save(f, 'Figure10.png')
print('figures written')
