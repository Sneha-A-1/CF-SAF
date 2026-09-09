"""
run_experiments_real.py
-----------------------
Runs the CF-SAF simulation and computes every reported metric.

Usage:
    python run_experiments_real.py --runs 10
    python run_experiments_real.py --runs 10 --ablation
    python run_experiments_real.py --runs 5  --scalability
    python run_experiments_real.py --runs 5  --guard-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.simulation.simulator import Simulator, Job, build_nodes   # noqa: E402

ROOT = Path(__file__).parent
RES = ROOT / "results"


# ------------------------------------------------------------------
def load_jobs(df: pd.DataFrame, rate: float, rng) -> list:
    gaps = rng.exponential(1000.0 / rate, len(df))
    arr = np.cumsum(gaps)
    jobs = []
    for k, (_, r) in enumerate(df.iterrows()):
        jobs.append(Job(tid=int(r.task_id), tau=float(r.tau), phi=float(r.phi),
                        delta_s=float(r.delta_s), rho_kb=float(r.rho_kb),
                        cpu=float(r.cpu_demand_mips), mem=float(r.mem_demand_mb),
                        psi=float(r.psi), label=int(r.label), pi=int(r.pi),
                        arrival_ms=float(arr[k])))
    return jobs


# ---- routers -----------------------------------------------------
def make_router(kind: str, gbt=None, theta: float = 0.47,
                eta_cal: float = 0.5, calib: bool = True,
                fixed_mu=None):
    if kind == "all_cloud":
        return lambda j, s: "cloud"
    if kind == "all_fog":
        return lambda j, s: "fog"
    if kind == "thresh":
        return lambda j, s: "fog" if (j.tau > 0.6 or j.psi >= 0.9) else "cloud"

    def idcra(j, s):
        if j.rho_kb < 0 or getattr(j, "pi", 0) == 1:
            return "fog"
        x = np.array([[j.tau, j.delta_s, j.rho_kb, j.phi,
                       float(getattr(j, "pi", 0)), j.cpu, j.mem]])
        s_gbt = float(gbt.predict_proba(x)[0, 1]) if gbt is not None \
            else float(j.label)
        lam = float(np.mean([n.load for n in s.nodes.values()]))
        if fixed_mu is not None:
            mu = fixed_mu
        else:
            den = lam + (1 - lam) * eta_cal
            mu = lam / den if den > 1e-9 else 0.0
        if not calib:
            mu = 0.0
        # calibrator: down-scores marginal-urgency items as load rises
        s_cal = s_gbt * (1.0 - 0.5 * lam)
        score = (1 - mu) * s_gbt + mu * s_cal
        return "fog" if score >= theta else "cloud"

    return idcra


def train_gbt(df, seed=42):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    X = df[["tau", "delta_s", "rho_kb", "phi", "pi",
            "cpu_demand_mips", "mem_demand_mb"]].values
    y = df["label"].values
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, train_size=0.70,
                                            stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, train_size=0.50,
                                          stratify=ytmp, random_state=seed)
    g = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                   learning_rate=0.1, subsample=0.8,
                                   min_samples_leaf=5, random_state=seed)
    g.fit(Xtr, ytr)

    def fbeta_at(scores, yy, th, beta):
        pred = (scores >= th).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            yy, pred, average="binary", zero_division=0)
        if p + r == 0:
            return 0.0, p, r
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r), p, r

    sv = g.predict_proba(Xva)[:, 1]
    grid = np.arange(0.05, 0.96, 0.01)
    sel = {}
    for beta in (1, 2, 3):
        vals = np.array([fbeta_at(sv, yva, t, beta)[0] for t in grid])
        best = vals.max()
        plateau = grid[vals >= best - 1e-9]
        th = float(np.median(plateau))   # centre of the optimal plateau
        _, p, r = fbeta_at(sv, yva, th, beta)
        sel[f"F{beta}"] = {"theta": round(th, 3),
                           "precision": round(float(p), 4),
                           "recall": round(float(r), 4)}
    st = g.predict_proba(Xte)[:, 1]
    th1 = sel["F1"]["theta"]
    pred = (st >= th1).astype(int)
    p1, r1, f1, _ = precision_recall_fscore_support(
        yte, pred, average="binary", zero_division=0)
    p0, r0, f0, _ = precision_recall_fscore_support(
        1 - yte, 1 - pred, average="binary", zero_division=0)
    cls = {
        "class1": {"precision": round(float(p1), 4), "recall": round(float(r1), 4),
                   "f1": round(float(f1), 4), "fnr": round(1 - float(r1), 4)},
        "class0": {"precision": round(float(p0), 4), "recall": round(float(r0), 4),
                   "f1": round(float(f0), 4), "fnr": round(1 - float(r0), 4)},
        "n_test": int(len(yte)), "n_urgent": int(yte.sum()),
    }
    return g, sel, cls


SYSTEMS = {
    "Cloud-Only":    dict(router="all_cloud", scheduler="fcfs"),
    "Naive-Fog":     dict(router="all_fog",   scheduler="rr"),
    "Threshold-Fog": dict(router="thresh",    scheduler="fcfs"),
    "FuzzyOff":      dict(router="idcra",     scheduler="fuzzy"),
    "HEFT-Fog":      dict(router="idcra",     scheduler="heft"),
    "FuzzyDRL-Fog":  dict(router="idcra",     scheduler="fuzzy"),
    "CF-SAF":        dict(router="idcra",     scheduler="orsaf"),
}


def one_run(df, cfg, spec, seed, rate=None, **kw):
    rng = np.random.default_rng(seed)
    nodes = build_nodes(cfg)
    for n in nodes:
        n.load = float(rng.uniform(0.10, 0.40))
    rate = rate or cfg["dataset"]["arrival_rate_per_s"]
    jobs = load_jobs(df, rate, rng)
    sim = Simulator(nodes, scheduler=spec["scheduler"], rng=rng, **kw)
    router = make_router(spec["router"], gbt=kw.pop("gbt", None)
                         if "gbt" in kw else spec.get("gbt"),
                         theta=spec.get("theta", 0.47),
                         calib=spec.get("calib", True),
                         fixed_mu=spec.get("fixed_mu"))
    sim.run(jobs, router)
    return sim.metrics()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--n_tasks", type=int, default=8000)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--scalability", action="store_true")
    ap.add_argument("--guard-check", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
    df_all = pd.read_csv(ROOT / "data/synthetic/tasks.csv")
    df = df_all.head(a.n_tasks).copy()
    RES.mkdir(exist_ok=True)

    print("Training IDCRA GBT ...")
    gbt, fbeta, cls = train_gbt(df_all)
    print(f"  theta*(F1) = {fbeta['F1']['theta']}   "
          f"urgent recall = {cls['class1']['recall']}")
    json.dump({"fbeta": fbeta, "classification": cls},
              open(RES / "idcra_real.json", "w"), indent=2)

    print(f"\nMain comparison — {a.runs} runs x {len(df)} tasks")
    rows, raw = [], {}
    for name, base in SYSTEMS.items():
        spec = dict(base); spec["gbt"] = gbt
        spec["theta"] = fbeta["F1"]["theta"]
        runs = [one_run(df, cfg, spec, 42 + 13 * k) for k in range(a.runs)]
        raw[name] = runs
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        ci = 1.96 * np.std([r["latency_ms"] for r in runs]) / np.sqrt(a.runs)
        rows.append({"System": name,
                     "latency_ms": round(agg["latency_ms"], 2),
                     "ci95": round(float(ci), 2),
                     "exec_ms": round(agg["exec_ms"], 2),
                     "utilisation_pct": round(agg["utilisation"], 1),
                     "throughput": round(agg["throughput"], 1),
                     "energy_j": round(agg["energy_j"], 4),
                     "p95_ms": round(agg["p95_ms"], 2),
                     "hp_unbounded_pct": round(agg["hp_unbounded_pct"], 2)})
        print(f"  {name:<15} L={agg['latency_ms']:8.2f}  "
              f"E={agg['exec_ms']:7.2f}  U={agg['utilisation']:5.1f}  "
              f"T={agg['throughput']:7.1f}  En={agg['energy_j']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "main_results_real.csv", index=False)
    print(f"\nwrote {RES/'main_results_real.csv'}")

    # significance vs Cloud-Only
    from scipy import stats
    base = [r["latency_ms"] for r in raw["Cloud-Only"]]
    print("\nPaired t-test vs Cloud-Only (latency):")
    for name in SYSTEMS:
        if name == "Cloud-Only":
            continue
        v = [r["latency_ms"] for r in raw[name]]
        t, p = stats.ttest_rel(base, v)
        d = (np.mean(base) - np.mean(v)) / max(np.std(np.array(base) - np.array(v)), 1e-12)
        print(f"  {name:<15} delta={100*(np.mean(v)-np.mean(base))/np.mean(base):+7.1f}%  "
              f"t={t:8.2f}  p={p:.2e}  d={d:6.2f}")

    if a.ablation:
        print("\nAblation:")
        variants = {
            "CF-SAF (full)":      dict(),
            "A1: GBT-only":       dict(calib=False),
            "A2: Fixed mu=0.5":   dict(fixed_mu=0.5),
            "A3: No preemption":  dict(sim=dict(preemption=False)),
            "A4: No rebalancing": dict(sim=dict(rebalancing=False)),
        }
        arows = []
        for nm, v in variants.items():
            spec = dict(SYSTEMS["CF-SAF"]); spec["gbt"] = gbt
            spec["theta"] = fbeta["F1"]["theta"]
            spec["calib"] = v.get("calib", True)
            spec["fixed_mu"] = v.get("fixed_mu")
            simkw = v.get("sim", {})
            runs = [one_run(df, cfg, spec, 42 + 13 * k, **simkw)
                    for k in range(a.runs)]
            agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            arows.append({"Variant": nm,
                          "latency_ms": round(agg["latency_ms"], 2),
                          "utilisation_pct": round(agg["utilisation"], 1),
                          "p95_ms": round(agg["p95_ms"], 2)})
            print(f"  {nm:<20} L={agg['latency_ms']:8.2f}  "
                  f"U={agg['utilisation']:5.1f}  P95={agg['p95_ms']:8.2f}")
        pd.DataFrame(arows).to_csv(RES / "ablation_real.csv", index=False)

    if a.scalability:
        print("\nScalability (latency vs arrival rate):")
        rates = [60, 110, 160, 220, 270, 320, 370, 420, 470, 520]
        srows = []
        for nm in ("HEFT-Fog", "FuzzyDRL-Fog", "CF-SAF"):
            spec = dict(SYSTEMS[nm]); spec["gbt"] = gbt
            spec["theta"] = fbeta["F1"]["theta"]
            for rt in rates:
                runs = [one_run(df, cfg, spec, 42 + 13 * k, rate=rt)
                        for k in range(max(2, a.runs // 3))]
                m = float(np.mean([r["latency_ms"] for r in runs]))
                hp = float(np.mean([r["hp_unbounded_pct"] for r in runs]))
                srows.append({"system": nm, "rate": rt,
                              "latency_ms": round(m, 2),
                              "hp_unbounded_pct": round(hp, 2)})
                print(f"  {nm:<14} {rt:>4}/s  L={m:8.2f}  "
                      f"unbounded={hp:5.2f}%")
        pd.DataFrame(srows).to_csv(RES / "scalability_real.csv", index=False)

    if a.guard_check:
        print("\nMigration guard (M1) on vs off — Theorem 11:")
        for label, g in (("with (M1)", True), ("without (M1)", False)):
            spec = dict(SYSTEMS["CF-SAF"]); spec["gbt"] = gbt
            spec["theta"] = fbeta["F1"]["theta"]
            runs = [one_run(df, cfg, spec, 42 + 13 * k, rate=320, guard_m1=g)
                    for k in range(max(3, a.runs // 2))]
            print(f"  {label:<14} "
                  f"L={np.mean([r['latency_ms'] for r in runs]):8.2f}  "
                  f"U={np.mean([r['utilisation'] for r in runs]):5.1f}  "
                  f"migrations={np.mean([r['migrations'] for r in runs]):6.1f}")


if __name__ == "__main__":
    main()
