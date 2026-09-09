import sys, os, numpy as np, pandas as pd, yaml, importlib, pickle
import paths
m = importlib.import_module('run_experiments_real')
from src.simulation.simulator import Simulator, build_nodes
from sklearn.ensemble import GradientBoostingClassifier
cfg=yaml.safe_load(open(os.path.join(paths.ROOT,'config','config.yaml')))
full=pd.read_csv(paths.DATA)
F=['tau','delta_s','rho_kb','phi','pi','cpu_demand_mips','mem_demand_mb']
gbt=GradientBoostingClassifier(n_estimators=100,max_depth=4,learning_rate=.1,
     subsample=.8,min_samples_leaf=5,random_state=42).fit(full[F].values, full.label.values)
pickle.dump(gbt, open(paths.GBT,'wb'))
print("Stability check: does mean latency converge as run length grows?")
print("A stable operating point gives the SAME latency at n=4k and n=12k.\n")
print(f"{'rate':>5} {'n=4000':>10} {'n=12000':>10} {'drift':>10} {'util%':>7}")
for rate in [250,260,270,280]:
    r=[]
    for n in [4000,12000]:
        df=full.head(n); rng=np.random.default_rng(42)
        jobs=m.load_jobs(df,rate,rng)
        sim=Simulator(build_nodes(cfg),scheduler='orsaf')
        sim.run(jobs, m.make_router('idcra',gbt=gbt,theta=0.22))
        r.append(sim.metrics()['latency_ms'])
    d=100*(r[1]-r[0])/r[0]
    print(f"{rate:>5} {r[0]:>10.1f} {r[1]:>10.1f} {d:>9.1f}%")
