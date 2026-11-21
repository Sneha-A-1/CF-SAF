import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
import sys, json; sys.path.insert(0,_HERE)
from regen_rest import *  # reuses VAR, one, slim, res, save (skips completed keys)
key='ablation_mmpp240'
if key not in res:
    gaps=mmpp_gaps(240,burst=1.5); res[key]={}
    for v,(skw,rkw) in VAR.items():
        res[key][v]=[slim(one('CF-SAF',s,rate=240,gaps=gaps,simkw=skw,rkw=rkw)[0]) for s in SEEDS]
    for nm in ('HEFT-Fog','Cloud-Only','Threshold-Fog'):
        res[key][nm]=[slim(one(nm,s,rate=240,gaps=gaps)[0]) for s in SEEDS]
    save(); print('mmpp240 done')
