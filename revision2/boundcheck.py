import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
import sys, json; sys.path.insert(0,_HERE)
from exp_cloud import run, per_task_stats, SEEDS
from workloads import mmpp_gaps
import numpy as np
out={}
for lab,kw in (('poisson240',dict(rate=240)),('poisson260',dict(rate=260)),('mmpp240',dict(rate=240,gaps=mmpp_gaps(240,burst=1.5)))):
    rs=[per_task_stats(*run('CF-SAF',s,**kw)) for s in SEEDS]
    out[lab]={k:float(np.mean([r[k] for r in rs])) for k in ('l2_pre_after','l2_hp_all','l2_hp','l2_ok','l2_lam_le','l2_uniform_ok','c3_meas','c3_lb','c3_lb2','c3_busy','c3_busy_lb')}
    out[lab]['c3_ok']=int(sum(r['c3_meas']>=r['c3_lb']-1e-12 and r['c3_lb']>=r['c3_lb2']-1e-12 for r in rs)); out[lab]['l2_allok']=int(sum(r['l2_ok']==r['l2_hp'] for r in rs))
    print(lab,out[lab],flush=True)
json.dump(out,open(_os.path.join(_HERE, 'results', 'boundcheck.json'),'w'),indent=1)
