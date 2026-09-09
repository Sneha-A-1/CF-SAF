import sys, json, os, numpy as np, pandas as pd, yaml, importlib, pickle
import paths
m=importlib.import_module('run_experiments_real')
from src.simulation.simulator import Simulator, build_nodes
cfg=yaml.safe_load(open(os.path.join(paths.ROOT,'config','config.yaml')))
full=pd.read_csv(paths.DATA)
gbt=pickle.load(open(paths.GBT,'rb'))
RATE, N, NT = 270, 10, 12000   # paper operating point
df=full.head(NT)
SPECS={'Cloud-Only':('all_cloud','fcfs'),'Naive Fog':('all_fog','rr'),
 'Threshold-Fog':('thresh','fcfs'),'FuzzyOff':('idcra','fuzzy'),
 'HEFT-Fog':('idcra','heft'),'FuzzyDRL-Fog':('idcra','fuzzy'),
 'CF-SAF':('idcra','orsaf')}
STORE=os.path.join(paths.RESULTS,'full_results.json')
res=json.load(open(STORE)) if os.path.exists(STORE) else {}
todo=sys.argv[1:] or list(SPECS.keys())
for nm in todo:
    rk,sc=SPECS[nm]; runs=[]
    for k in range(N):
        rng=np.random.default_rng(42+13*k); jobs=m.load_jobs(df,RATE,rng)
        sim=Simulator(build_nodes(cfg),scheduler=sc)
        sim.run(jobs, m.make_router(rk,gbt=gbt,theta=0.22))
        mt=sim.metrics()
        mt['per_node_util']=[100*min(1,n.busy_ms/max(sim.t,1e-9))
                             for n in sim.nodes.values()]
        if k==0: mt['lat_sample']=[float(j.latency_ms) for j in sim.done]
        runs.append({kk:(vv if not isinstance(vv,np.floating) else float(vv))
                     for kk,vv in mt.items()})
    res[nm]=runs
    lat=[r['latency_ms'] for r in runs]
    print(f"{nm:<15} lat={np.mean(lat):9.2f} +/-{np.std(lat):6.2f}  "
          f"util={np.nanmean([np.mean(r['per_node_util']) for r in runs]):5.1f}")
    json.dump(res, open(STORE,'w'))
print("chunk saved")
