"""
train_idcra.py
--------------
Training script for IDCRA (GBT + Neural Calibrator).

Usage:
    python src/idcra/train_idcra.py --config config/config.yaml
"""

import argparse
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, f1_score,
                              confusion_matrix)

from feature_engineering import build_feature_matrix


# ----------------------------------------------------------------
# Neural Urgency Calibrator
# ----------------------------------------------------------------
class NeuralCalibrator(nn.Module):
    """
    Two-layer MLP: [input -> 16 -> 8 -> 1 (sigmoid)]
    Input: 9 features = 8 (feature vector) + 1 (lambda_avg)
    """
    def __init__(self, input_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),         nn.ReLU(),
            nn.Linear(8, 1),          nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------
# Threshold selection (F1 maximisation on validation set)
# ----------------------------------------------------------------
def select_threshold(scores: np.ndarray, labels: np.ndarray,
                     thresholds: np.ndarray = None) -> float:
    if thresholds is None:
        thresholds = np.arange(0.20, 0.81, 0.01)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    print(f"  Best threshold: {best_thresh:.2f}  (val F1 = {best_f1:.4f})")
    return best_thresh


# ----------------------------------------------------------------
# Main training
# ----------------------------------------------------------------
def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]["numpy"]
    np.random.seed(seed)
    torch.manual_seed(cfg["seed"]["torch"])

    # -- Load dataset --
    data_path = Path("data/synthetic/tasks.csv")
    if not data_path.exists():
        print("Dataset not found. Run: python data/synthetic/generate_dataset.py")
        return

    df = pd.read_csv(data_path)
    X = build_feature_matrix(df)
    y = df["label"].values

    # -- Stratified split --
    ratios = cfg["dataset"]["split"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, train_size=ratios[0], stratify=y, random_state=seed)
    val_frac = ratios[1] / (1 - ratios[0])
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, train_size=val_frac, stratify=y_tmp, random_state=seed)

    print(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

    # ------------------------------------------------------------------
    # Step 1: Train GBT
    # ------------------------------------------------------------------
    gbt_cfg = cfg["idcra"]["gbt"]
    print("\nTraining GBT ...")
    gbt = GradientBoostingClassifier(
        n_estimators=gbt_cfg["n_estimators"],
        max_depth=gbt_cfg["max_depth"],
        learning_rate=gbt_cfg["learning_rate"],
        subsample=gbt_cfg["subsample"],
        min_samples_leaf=gbt_cfg["min_samples_leaf"],
        random_state=seed
    )
    gbt.fit(X_tr, y_tr)

    # GBT validation scores
    gbt_val_scores = gbt.predict_proba(X_val)[:, 1]

    # ------------------------------------------------------------------
    # Step 2: Train Neural Calibrator
    # ------------------------------------------------------------------
    nn_cfg = cfg["idcra"]["neural"]
    print("Training Neural Calibrator ...")

    # Append dummy lambda_avg=0.3 for training (mean initial load)
    lambda_avg_train = 0.3
    X_tr_nn = np.hstack([X_tr, np.full((len(X_tr), 1), lambda_avg_train)])
    X_val_nn = np.hstack([X_val, np.full((len(X_val), 1), lambda_avg_train)])

    model = NeuralCalibrator(input_dim=X_tr_nn.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=nn_cfg["lr"],
        weight_decay=nn_cfg["weight_decay"]
    )
    criterion = nn.BCELoss()

    Xt = torch.tensor(X_tr_nn, dtype=torch.float32)
    yt = torch.tensor(y_tr,    dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_val_nn, dtype=torch.float32)

    batch_size = nn_cfg["batch_size"]
    for epoch in range(nn_cfg["epochs"]):
        model.train()
        idx = np.random.permutation(len(Xt))
        total_loss = 0.0
        for start in range(0, len(Xt), batch_size):
            batch_idx = idx[start:start + batch_size]
            xb, yb = Xt[batch_idx], yt[batch_idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{nn_cfg['epochs']} "
                  f"loss={total_loss/len(Xt)*batch_size:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Select threshold on validation set
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        nn_val_scores = model(Xv).squeeze().numpy()

    eta_cal = cfg["idcra"]["eta_cal"]
    lambda_avg = lambda_avg_train
    mu = lambda_avg / (lambda_avg + (1 - lambda_avg) * eta_cal)
    combined_val = (1 - mu) * gbt_val_scores + mu * nn_val_scores

    print("\nSelecting routing threshold (F1 maximisation on val set):")
    theta_star = select_threshold(combined_val, y_val)

    # ------------------------------------------------------------------
    # Step 4: Evaluate on test set
    # ------------------------------------------------------------------
    X_te_nn = np.hstack([X_te, np.full((len(X_te), 1), lambda_avg_train)])
    Xt_te = torch.tensor(X_te_nn, dtype=torch.float32)

    gbt_te_scores = gbt.predict_proba(X_te)[:, 1]
    with torch.no_grad():
        nn_te_scores = model(Xt_te).squeeze().numpy()

    combined_te = (1 - mu) * gbt_te_scores + mu * nn_te_scores
    y_pred = (combined_te >= theta_star).astype(int)

    print("\n=== Test Set Results ===")
    print(classification_report(y_te, y_pred,
          target_names=["Non-urgent (0)", "Urgent (1)"]))
    cm = confusion_matrix(y_te, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    print(f"False Negative Rate (Urgent class): {fnr:.4f}")

    # ------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------
    import pickle
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "gbt_model.pkl", "wb") as f:
        pickle.dump(gbt, f)
    torch.save(model.state_dict(), out_dir / "neural_calibrator.pt")
    with open(out_dir / "idcra_threshold.txt", "w") as f:
        f.write(f"{theta_star:.4f}\n")
    print(f"\nModels saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    train(args.config)
