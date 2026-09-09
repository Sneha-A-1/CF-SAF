"""Repository-relative paths. Import this instead of hardcoding."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA    = os.path.join(ROOT, 'data', 'synthetic', 'tasks.csv')
RESULTS = os.path.join(ROOT, 'results')
MODELS  = os.path.join(RESULTS, 'models')
FIGDATA = os.path.join(RESULTS, 'figdata')
FIGURES = os.path.join(RESULTS, 'figures')
GBT     = os.path.join(MODELS, 'gbt.pkl')
for d in (RESULTS, MODELS, FIGDATA, FIGURES):
    os.makedirs(d, exist_ok=True)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
