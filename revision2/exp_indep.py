import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
"""IDCRA on independent and real-data workloads (Reviewer 2, comment f)."""
import sys, json; sys.path.insert(0,_HERE); sys.path.insert(0,_os.path.join(_REPO, 'data', 'synthetic'))
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from generate_dataset import grouped_split
from workloads import *

FEATS=['tau','sigma','delta_s','pi','rho_kb','phi']
base=pd.read_csv(_os.path.join(_REPO, 'data', 'synthetic', 'tasks.csv'))
tr,va,te=grouped_split(base,42)
mk=lambda: GradientBoostingClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,subsample=0.8,min_samples_leaf=5,random_state=42)
g=mk().fit(tr[FEATS].values,tr.label.values)
TH=0.22
def ev(d, scores, th, extra=None):
    y=d.label.values; p=(scores>=th).astype(int)
    pr,rc,f,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    acc=(p==y).mean()
    out=dict(n=len(d),urgent=float(y.mean()),prec=pr,rec=rc,f1=f,acc=acc,auc=roc_auc_score(y,scores) if 0<y.mean()<1 else float('nan'))
    # naive observed-tau rule (Threshold-Fog's criterion) for reference
    nv=((d.tau.values>0.6)|(d.pi.values==1)).astype(int)
    p2,r2,f2,_=precision_recall_fscore_support(y,nv,average='binary',zero_division=0)
    out.update(naive_prec=p2,naive_rec=r2,naive_f1=f2)
    return out
res={}
res['T0 original test split (Table VI)']=ev(te,g.predict_proba(te[FEATS].values)[:,1],TH)
for nm,kw in INDEP.items():
    d=gen(**kw); _,_,dte=grouped_split(d,7)
    res[nm]=ev(dte,g.predict_proba(dte[FEATS].values)[:,1],TH)
# real data
r=real_tasks(seed=11)
res['R real farm trace (transfer, no retraining)']=ev(r,g.predict_proba(r[FEATS].values)[:,1],TH)
# in-domain reference: temporal split 70/15/15 on real data, theta by F1 on val
o=np.argsort(r.ts.values); n=len(r); i1,i2=int(.7*n),int(.85*n)
rtr,rva,rte=r.iloc[o[:i1]],r.iloc[o[i1:i2]],r.iloc[o[i2:]]
g2=mk().fit(rtr[FEATS].values,rtr.label.values)
sv=g2.predict_proba(rva[FEATS].values)[:,1]
best=max(np.arange(0.05,0.96,0.01),key=lambda t: precision_recall_fscore_support(rva.label.values,(sv>=t).astype(int),average='binary',zero_division=0)[2])
res['R real farm trace (retrained, temporal split)']=ev(rte,g2.predict_proba(rte[FEATS].values)[:,1],best); res['R real farm trace (retrained, temporal split)']['theta']=float(best)
res['R real (transfer) on same temporal test split']=ev(rte,g.predict_proba(rte[FEATS].values)[:,1],TH)
for k,v in res.items(): print(f"{k:<48}", {kk:round(vv,3) for kk,vv in v.items()})
json.dump(res,open(_os.path.join(_HERE, 'results', 'indep_cls.json'),'w'),indent=1)
