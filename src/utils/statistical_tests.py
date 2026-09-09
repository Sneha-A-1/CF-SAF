"""
statistical_tests.py
--------------------
Paired t-test significance testing for CF-SAF experiment results.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def paired_ttest(system_a: List[float],
                 system_b: List[float],
                 alpha: float = 0.001,
                 alternative: str = "two-sided") -> Dict:
    """
    Perform a paired two-sided t-test between two systems.

    Parameters
    ----------
    system_a   : Metric values for system A (10 runs)
    system_b   : Metric values for system B (10 runs)
    alpha      : Significance level (default 0.001)
    alternative: 'two-sided', 'less', or 'greater'

    Returns
    -------
    dict with t_stat, p_value, significant, mean_a, mean_b, reduction_pct
    """
    assert len(system_a) == len(system_b), "Must have same number of runs"
    t_stat, p_value = stats.ttest_rel(system_a, system_b,
                                       alternative=alternative)
    mean_a = np.mean(system_a)
    mean_b = np.mean(system_b)
    reduction = 100.0 * (mean_a - mean_b) / mean_a if mean_a != 0 else 0.0

    return {
        "mean_a":        round(mean_a, 4),
        "mean_b":        round(mean_b, 4),
        "t_stat":        round(float(t_stat), 4),
        "p_value":       float(p_value),
        "significant":   bool(p_value < alpha),
        "alpha":         alpha,
        "reduction_pct": round(reduction, 2),
    }


def confidence_interval_95(values: List[float]) -> Tuple[float, float]:
    """Compute 95% confidence interval."""
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return round(ci[0], 3), round(ci[1], 3)


def run_all_significance_tests(results: Dict[str, List[float]],
                                baseline_key: str = "Cloud-Only",
                                alpha: float = 0.001) -> None:
    """
    Run significance tests for all systems vs. a baseline.

    Parameters
    ----------
    results     : {system_name: [metric_value per run]}
    baseline_key: Name of the baseline system
    alpha       : Significance level
    """
    baseline = results[baseline_key]
    print(f"\n{'System':<20} {'Mean':>8} {'vs Baseline':>12} "
          f"{'p-value':>12} {'Significant':>12}")
    print("-" * 68)
    for name, values in results.items():
        mean = np.mean(values)
        if name == baseline_key:
            print(f"{name:<20} {mean:>8.3f} {'---':>12} {'---':>12} {'---':>12}")
            continue
        test = paired_ttest(baseline, values, alpha=alpha)
        sig = "YES (p<0.001)" if test["significant"] else "NO"
        print(f"{name:<20} {mean:>8.3f} "
              f"{test['reduction_pct']:>11.1f}% "
              f"{test['p_value']:>12.2e} {sig:>12}")
