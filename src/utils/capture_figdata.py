import sys, os, numpy as np, pandas as pd, yaml, importlib, pickle
import paths
m = importlib.import_module('run_experiments_real')
from src.simulation.simulator import Simulator, build_nodes
from sklearn.ensemble import GradientBoostingClassifier
cfg = yaml.safe_load(open(os.path.join(paths.ROOT,'config','config.yaml')))
full = pd.read_csv(paths.DATA)
F=['tau','delta_s','rho_kb','phi','pi','cpu_demand_mips','mem_demand_mb']
gbt = pickle.load(open(paths.GBT,'rb'))
df = full.head(4000)
RATE=300
SPECS = {'Cloud-Only':('all_cloud','fcfs'),'Naive Fog':('all_fog','rr'),
 'Threshold-Fog':('thresh','fcfs'),'FuzzyOff':('idcra','fuzzy'),
 'HEFT-Fog':('idcra','heft'),'FuzzyDRL-Fog':('idcra','fuzzy'),
 'CF-SAF':('idcra','orsaf')}
out={}
for nm,(rk,sc) in SPECS.items():
    rng=np.random.default_rng(42); jobs=m.load_jobs(df,RATE,rng)
    sim=Simulator(build_nodes(cfg), scheduler=sc)
    sim.run(jobs, m.make_router(rk, gbt=gbt, theta=0.22))
    lat=np.array([j.latency_ms for j in sim.done])
    util=[100.0*min(1.0,n.busy_ms/max(sim.t,1e-9)) for n in sim.nodes.values()]
    out[nm]={'lat':lat,'util':util,'metrics':sim.metrics()}
    print(f"{nm:<15} n={len(lat):5d} mean={lat.mean():9.2f} nodes={len(util)}")
pickle.dump(out, open(os.path.join(paths.FIGDATA,'main.pkl'),'wb'))
print("saved")
