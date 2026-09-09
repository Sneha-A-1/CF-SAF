import numpy as np, pandas as pd, json, sys
sys.path.insert(0,'data/synthetic')
from generate_dataset import grouped_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

df = pd.read_csv('data/synthetic/tasks.csv')
tr, va, te = grouped_split(df, 42)
print(f"train={len(tr)} val={len(va)} test={len(te)}")
print(f"class balance tr/va/te: {tr.label.mean():.3f} {va.label.mean():.3f} {te.label.mean():.3f}")

FEATS = ['tau','sigma','delta_s','pi','rho_kb','phi']
def X(d): return d[FEATS].values
gbt = GradientBoostingClassifier(n_estimators=100, max_depth=4,
        learning_rate=0.1, subsample=0.8, min_samples_leaf=5,
        random_state=42)
gbt.fit(X(tr), tr.label.values)

s_va = gbt.predict_proba(X(va))[:,1]
s_te = gbt.predict_proba(X(te))[:,1]

def fbeta_at(s, y, thr, beta):
    p = (s>=thr).astype(int)
    pr,rc,_,_ = precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    b2=beta*beta
    f = (1+b2)*pr*rc/(b2*pr+rc) if (pr+rc)>0 else 0
    return pr,rc,f

grid = np.arange(0.05,0.96,0.01)
out={}
for beta in [1,2,3]:
    best=(0,0.5)
    for t in grid:
        _,_,f = fbeta_at(s_va, va.label.values, t, beta)
        if f>best[0]: best=(f,t)
    f,t = best
    pr,rc,_ = fbeta_at(s_va, va.label.values, t, beta)
    out[f"F{beta}"]={"theta":round(float(t),2),"precision":round(pr,3),
                    "recall":round(rc,3),"fbeta":round(f,4)}
    print(f"F{beta}: theta*={t:.2f} P={pr:.3f} R={rc:.3f} F={f:.4f}")

theta = out["F1"]["theta"]
pred = (s_te>=theta).astype(int)
pr,rc,f1,sup = precision_recall_fscore_support(te.label.values,pred,zero_division=0)
cm = confusion_matrix(te.label.values,pred)
fnr = cm[:,0]/cm.sum(1)  # per-class miss
print("\n=== TEST SET (theta* from F1 on val) ===")
for c in [0,1]:
    print(f"Class {c}: P={pr[c]:.3f} R={rc[c]:.3f} F1={f1[c]:.3f} "
          f"FNR={1-rc[c]:.3f} n={sup[c]}")
mp,mr,mf,_ = precision_recall_fscore_support(te.label.values,pred,average='macro',zero_division=0)
wp,wr,wf,_ = precision_recall_fscore_support(te.label.values,pred,average='weighted',zero_division=0)
print(f"Macro : P={mp:.3f} R={mr:.3f} F1={mf:.3f}")
print(f"Weight: P={wp:.3f} R={wr:.3f} F1={wf:.3f}")
print(f"n_test={len(te)}  urgent={int(te.label.sum())}  nonurgent={int((1-te.label).sum())}")

json.dump({"theta":theta,"fbeta":out,
  "test":{"class0":{"precision":round(pr[0],3),"recall":round(rc[0],3),
                    "f1":round(f1[0],3),"fnr":round(1-rc[0],3),"n":int(sup[0])},
          "class1":{"precision":round(pr[1],3),"recall":round(rc[1],3),
                    "f1":round(f1[1],3),"fnr":round(1-rc[1],3),"n":int(sup[1])},
          "macro":{"precision":round(mp,3),"recall":round(mr,3),"f1":round(mf,3)},
          "weighted":{"precision":round(wp,3),"recall":round(wr,3),"f1":round(wf,3)},
          "n_test":len(te)}},
  open('results/idcra_real.json','w'), indent=2)
