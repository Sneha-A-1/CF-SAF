"""
Trains the IDCRA gradient-boosted tree on the full dataset and saves it.
All scheduling scripts load this model, so run this first.

    python src/utils/train_and_save_gbt.py
"""
import pickle, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import paths

FEATURES = ['tau', 'delta_s', 'rho_kb', 'phi', 'pi',
            'cpu_demand_mips', 'mem_demand_mb']

def main():
    df = pd.read_csv(paths.DATA)
    gbt = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=5, random_state=42)
    gbt.fit(df[FEATURES].values, df.label.values)
    with open(paths.GBT, 'wb') as f:
        pickle.dump(gbt, f)
    print(f"GBT trained on {len(df)} items -> {paths.GBT}")

if __name__ == '__main__':
    main()
