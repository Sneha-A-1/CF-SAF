import sys, json, os, numpy as np, pandas as pd, yaml, importlib, pickle
import paths
m=importlib.import_module('run_experiments_real')
from src.simulation.simulator import Simulator, build_nodes
cfg=yaml.safe_load(open(os.path.join(paths.ROOT,'config','config.yaml')))
full=pd.read_csv(paths.DATA); gbt=pickle.load(open(paths.GBT,'rb'))
RATE, NT = 270, 12000
df=full.head(NT)
P=os.path.join(paths.RESULTS,'heft_30seed_raw.json')
acc=json.load(open(P)) if os.path.exists(P) else {}
sched={'CF-SAF':'orsaf','HEFT-Fog':'heft'}
lo,hi=int(sys.argv[1]),int(sys.argv[2])
for nm,sc in sched.items():
    acc.setdefault(nm,{})
    for k in range(lo,hi):
        if str(k) in acc[nm]: continue
        rng=np.random.default_rng(1000+7*k); jobs=m.load_jobs(df,RATE,rng)
        sim=Simulator(build_nodes(cfg),scheduler=sc)
        sim.run(jobs, m.make_router('idcra',gbt=gbt,theta=0.22))
        mt=sim.metrics()
        acc[nm][str(k)]={kk:float(mt[kk]) for kk in
            ['latency_ms','exec_ms','throughput','energy_j','p95_ms']}
        acc[nm][str(k)]['util']=float(np.mean(
            [100*min(1,n.busy_ms/max(sim.t,1e-9)) for n in sim.nodes.values()]))
        json.dump(acc,open(P,'w'))
    print(f"{nm}: {len(acc[nm])} seeds stored")
