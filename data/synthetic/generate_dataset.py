"""
generate_dataset.py
-------------------
Synthetic agricultural IoT dataset generator for CF-SAF.


We model an unobserved DEVICE-level latent context c_dev ~ N(0, sigma_ctx):

    tau_true = tau_observed + c_dev + epsilon_item

Labels threshold tau_true; the classifier sees only tau_observed. Bayes
error is therefore non-zero and theta* is identifiable.

c_dev is a device property, which is why grouped splitting is required:
items from one device share c_dev, so a device spanning train and test
would let the model infer that device's context.

Usage:
    python generate_dataset.py --n_tasks 50000 --seed 42 --sigma_ctx 0.16
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


TAU_LABEL_THRESH = 0.72   # calibrated so the urgent fraction matches the
                          # 28-30% empirical range of Sott et al. (2020)


def compute_urgency(phi, delta, rho, threshold_breach,
                    w1=0.5, w2=0.3, w3=0.2,
                    delta_max=86400.0, rho_max=2867.0):
    """Observable urgency score tau (Eq. 5 of the paper)."""
    t1 = w1 * threshold_breach
    t2 = w2 * (phi / (1.0 + delta / delta_max))
    t3 = w3 * ((rho_max - rho) / rho_max)
    return np.clip(t1 + t2 + t3, 0.0, 1.0)


def generate_dataset(n_tasks=50000, seed=42, n_devices=1200,
                     sigma_ctx=0.16, sigma_item=0.05,
                     arrival_rate=50.0, output_path="tasks.csv"):
    rng = np.random.default_rng(seed)
    print(f"Generating {n_tasks} tasks / {n_devices} devices "
          f"(seed={seed}, sigma_ctx={sigma_ctx})")

    dev_ids   = np.arange(n_devices)
    dev_ctx   = rng.normal(0.0, sigma_ctx, n_devices)   # UNOBSERVED
    dev_sigma = rng.choice([0, 1, 2], n_devices, p=[.50, .35, .15])
    dev_phi   = rng.uniform(0.1, 1.0, n_devices)

    w = rng.gamma(2.0, 1.0, n_devices); w /= w.sum()
    item_dev = rng.choice(dev_ids, n_tasks, p=w)

    sigma = dev_sigma[item_dev]
    phi   = np.clip(dev_phi[item_dev] + rng.normal(0, .03, n_tasks), 0, 1)
    ctx   = dev_ctx[item_dev]

    rho    = np.where(sigma == 2,
                      rng.uniform(500.0, 2867.0, n_tasks),
                      rng.uniform(0.04, 50.0, n_tasks))
    pi     = (rng.random(n_tasks) < 0.08).astype(int)
    breach = (rng.random(n_tasks) < 0.28).astype(int)

    # Deadline is drawn from the OBSERVABLE urgency drivers, not from the
    # label. Drawing delta conditional on the label would make delta a
    # near-perfect proxy for it (delta<60 => urgent), reintroducing leakage
    # through a second channel.
    u_obs = 0.5*breach + 0.3*phi + 0.2*(1.0 - rho/2867.0)
    log_delta = 4.2 - 4.0*u_obs + rng.normal(0.0, 0.60, n_tasks)
    delta = np.clip(10.0**log_delta, 0.5, 86400.0)

    tau = compute_urgency(phi, delta, rho, breach)   # observable (Eq. 5)

    # True urgency adds the unobserved device context
    tau_true = tau + ctx + rng.normal(0.0, sigma_item, n_tasks)
    label = ((tau_true > TAU_LABEL_THRESH) | (pi == 1)).astype(int)

    cpu = rng.uniform(100.0, 8000.0, n_tasks)
    mem = rng.uniform(2.0, 512.0, n_tasks)
    psi = (tau + phi + 1.0 / (delta + 1e-6)) / 3.0
    t_arr = np.cumsum(rng.exponential(1.0 / arrival_rate, n_tasks))

    df = pd.DataFrame({
        "task_id": np.arange(n_tasks),
        "device_id": item_dev,
        "t_arrival_s": np.round(t_arr, 6),
        "tau": np.round(tau, 6),
        "sigma": sigma,
        "delta_s": np.round(delta, 3),
        "pi": pi,
        "rho_kb": np.round(rho, 4),
        "phi": np.round(phi, 6),
        "threshold_breach": breach,
        "cpu_demand_mips": np.round(cpu, 2),
        "mem_demand_mb": np.round(mem, 2),
        "psi": np.round(psi, 6),
        "task_type": np.where(label == 1, 0, 1),
        "label": label,
    })

    n_urg = int(label.sum())
    print(f"  Urgent  : {n_urg} ({100*n_urg/n_tasks:.1f}%)")
    print(f"  Duration: {t_arr[-1]:.1f} s over {len(np.unique(item_dev))} devices")
    naive = ((df.tau > 0.60) | (df.pi == 1)).astype(int)
    print(f"  Naive observed-tau rule agrees with label: "
          f"{100*(naive == df.label).mean():.2f}%  (100% => leakage)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    return df


def grouped_split(df, seed=42, ratios=(0.70, 0.15, 0.15)):
    """Split by device_id: no device appears in more than one split."""
    rng = np.random.default_rng(seed)
    devs = df.device_id.unique().copy()
    rng.shuffle(devs)
    n = len(devs)
    n_tr, n_va = int(round(ratios[0]*n)), int(round(ratios[1]*n))
    tr = df[df.device_id.isin(set(devs[:n_tr]))]
    va = df[df.device_id.isin(set(devs[n_tr:n_tr+n_va]))]
    te = df[df.device_id.isin(set(devs[n_tr+n_va:]))]
    assert not (set(tr.device_id) & set(te.device_id))
    assert not (set(tr.device_id) & set(va.device_id))
    assert not (set(va.device_id) & set(te.device_id))
    return tr, va, te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_devices", type=int, default=1200)
    ap.add_argument("--sigma_ctx", type=float, default=0.16)
    ap.add_argument("--output", type=str, default="tasks.csv")
    a = ap.parse_args()
    df = generate_dataset(a.n_tasks, a.seed, a.n_devices,
                          a.sigma_ctx, output_path=a.output)
    tr, va, te = grouped_split(df, a.seed)
    print(f"Grouped split: train={len(tr)} val={len(va)} test={len(te)}")
