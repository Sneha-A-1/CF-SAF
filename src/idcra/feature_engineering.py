"""
feature_engineering.py
-----------------------
Feature vector construction for IDCRA.
"""

import numpy as np


SENSOR_MODALITY_ENCODING = {
    0: [1, 0, 0],   # soil moisture / nutrient
    1: [0, 1, 0],   # atmospheric / temperature
    2: [0, 0, 1],   # imaging / drone / multispectral
}


def compute_urgency_score(phi: float, delta: float, rho: float,
                           threshold_breach: int,
                           w1: float = 0.5, w2: float = 0.3,
                           w3: float = 0.2,
                           delta_max: float = 86400.0,
                           rho_max: float = 2867.0) -> float:
    """
    Compute urgency score tau (Equation 5 in the paper).

    Args:
        phi:               Source criticality in [0,1]
        delta:             Deadline in seconds
        rho:               Payload size in KB
        threshold_breach:  1 if sensor crossed agronomic threshold else 0
        w1, w2, w3:        Weights summing to 1.0
        delta_max:         Maximum deadline (86400 s = 24 h)
        rho_max:           Maximum payload (2867 KB ~ 2.8 MB)

    Returns:
        tau: urgency score in [0, 1]
    """
    assert abs(w1 + w2 + w3 - 1.0) < 1e-6, "Weights must sum to 1.0"
    term1 = w1 * threshold_breach
    term2 = w2 * (phi / (1.0 + delta / delta_max))
    term3 = w3 * ((rho_max - rho) / rho_max)
    return float(np.clip(term1 + term2 + term3, 0.0, 1.0))


def build_feature_vector(tau: float, sigma: int, delta: float,
                          pi: int, rho: float, phi: float) -> np.ndarray:
    """
    Build 6-dimensional feature vector x_d = [tau, sigma_enc, delta, pi, rho, phi].
    sigma_enc is a 3-bit one-hot encoding (expands to 8 total features).

    Args:
        tau:   Urgency score
        sigma: Sensor modality type (0, 1, or 2)
        delta: Deadline (seconds)
        pi:    Privacy flag (0 or 1)
        rho:   Payload size (KB)
        phi:   Source criticality

    Returns:
        x: numpy array of shape (8,)
           [tau, sigma_0, sigma_1, sigma_2, delta, pi, rho, phi]
    """
    sigma_enc = SENSOR_MODALITY_ENCODING.get(sigma, [0, 0, 0])
    x = np.array([tau] + sigma_enc + [delta, float(pi), rho, phi],
                 dtype=np.float32)
    return x


def build_feature_matrix(df) -> np.ndarray:
    """
    Build feature matrix from a pandas DataFrame with columns:
    tau, sigma, delta_s, pi, rho_kb, phi, threshold_breach.

    Returns:
        X: numpy array of shape (n_samples, 8)
    """
    rows = []
    for _, row in df.iterrows():
        x = build_feature_vector(
            tau=row["tau"], sigma=int(row["sigma"]),
            delta=row["delta_s"], pi=int(row["pi"]),
            rho=row["rho_kb"], phi=row["phi"]
        )
        rows.append(x)
    return np.vstack(rows)
