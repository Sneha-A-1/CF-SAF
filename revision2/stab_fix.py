import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
import sys, json; sys.path.insert(0,_HERE)
from common import *
from simfix import SimFix
out={}
for rate in [150,180,200,220,240,250,260,270,280]:
    row={}
    for n in (4000,12000):
        L=[];U=[]
        for s in SEEDS:
            rng=np.random.default_rng(s); jobs=load_jobs(FULL.head(n),rate,rng)
            sim=SimFix(build_nodes(cfg),scheduler='orsaf'); sim.run(jobs,router())
            mt=sim.metrics(); L.append(mt['latency_ms']); U.append(mt['utilisation'])
        row[n]=(L,U)
    a,b=np.mean(row[4000][0]),np.mean(row[12000][0])
    out[rate]={'L4k':row[4000][0],'L12k':row[12000][0],'U12k':row[12000][1]}
    print(rate, round(a,1), round(b,1), f"{100*(b-a)/a:.1f}%", 'util',round(np.mean(row[12000][1]),1), 'per-seed12k',[round(x) for x in row[12000][0]], flush=True)
    json.dump(out,open(_os.path.join(_HERE, 'results', 'stab_fix.json'),'w'))
