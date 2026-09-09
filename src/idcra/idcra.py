"""
idcra.py
--------
Intelligent Data Classification and Routing Algorithm (IDCRA).


Combines:
  - Gradient-Boosted Tree (GBT) classifier
  - Neural urgency calibrator
  - Load-adaptive mixing coefficient mu (Equation 7)
"""

import numpy as np
from .feature_engineering import build_feature_vector


class IDCRA:
    """
    IDCRA: Intelligent Data Classification and Routing Algorithm.

    Parameters
    ----------
    gbt_model       : Trained sklearn GradientBoostingClassifier
    neural_model    : Trained PyTorch neural calibrator (nn.Module)
    threshold       : Routing threshold theta* (default 0.47)
    eta_cal         : Calibration constant (default 0.5)
    """

    def __init__(self, gbt_model, neural_model,
                 threshold: float = 0.47,
                 eta_cal: float = 0.50):
        self.gbt = gbt_model
        self.neural = neural_model
        self.theta_star = threshold
        self.eta_cal = eta_cal

    def compute_mu(self, lambda_avg: float) -> float:
        """
        Load-adaptive mixing coefficient (Equation 7).
        mu = lambda_avg / (lambda_avg + (1 - lambda_avg) * eta_cal)
        """
        denom = lambda_avg + (1.0 - lambda_avg) * self.eta_cal
        if denom < 1e-9:
            return 0.0
        return lambda_avg / denom

    def score(self, x: np.ndarray, lambda_avg: float) -> float:
        """
        Compute IDCRA routing score (Equation 6).
        s_IDCRA = (1 - mu) * s_GBT + mu * s_neural
        """
        # GBT score: probability of class 1 (urgent)
        s_gbt = float(self.gbt.predict_proba(x.reshape(1, -1))[0, 1])

        # Neural calibration score
        import torch
        with torch.no_grad():
            inp = torch.tensor(
                np.append(x, lambda_avg), dtype=torch.float32
            ).unsqueeze(0)
            s_neural = float(self.neural(inp).squeeze())

        mu = self.compute_mu(lambda_avg)
        return (1.0 - mu) * s_gbt + mu * s_neural

    def route(self, data_item: dict,
              fog_loads: list) -> str:
        """
        Route a single data item to 'fog' or 'cloud'.
        Implements Algorithm 1.

        Parameters
        ----------
        data_item : dict with keys tau, sigma, delta, pi, rho, phi
        fog_loads : list of current load fractions lambda_i per fog node

        Returns
        -------
        'fog' or 'cloud'
        """
        # Privacy override (line 3 of Algorithm 1)
        if data_item.get("pi", 0) == 1:
            return "fog"

        # Build feature vector
        x = build_feature_vector(
            tau=data_item["tau"],
            sigma=data_item["sigma"],
            delta=data_item["delta"],
            pi=data_item["pi"],
            rho=data_item["rho"],
            phi=data_item["phi"]
        )

        # Average fog load
        lambda_avg = float(np.mean(fog_loads)) if fog_loads else 0.0

        # Compute routing score
        s = self.score(x, lambda_avg)

        return "fog" if s >= self.theta_star else "cloud"

    def route_batch(self, data_items: list,
                    fog_loads: list) -> list:
        """Route a batch of data items."""
        return [self.route(d, fog_loads) for d in data_items]
