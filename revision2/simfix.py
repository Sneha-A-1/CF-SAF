import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
_sys.path.insert(0, _HERE)
"""Compatibility alias.  The preemption fix is now part of
src/simulation/simulator.py, so SimFix is simply the corrected Simulator."""
from src.simulation.simulator import Simulator, EPS_PRE_MS  # noqa: F401


class SimFix(Simulator):
    pass
