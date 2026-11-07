"""Wind speed -> turbine power, so scenarios can be read as energy (planning, battery sizing)."""
from __future__ import annotations

import numpy as np


def hub_height_speed(v10, hub=100.0, alpha=1 / 7):
    """Power-law extrapolation of 10 m wind to hub height (alpha ~1/7 for open terrain)."""
    return np.asarray(v10, dtype=np.float64) * (hub / 10.0) ** alpha


def capacity_factor(v_hub, cut_in=3.0, rated=12.0, cut_out=25.0):
    """Generic turbine curve, output in [0, 1]: cubic between cut-in and rated, 0 above cut-out."""
    v = np.asarray(v_hub, dtype=np.float64)
    cf = np.clip((v ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3), 0.0, 1.0)
    return np.where(v >= cut_out, 0.0, cf)


def wind_to_power(v10, **kw):
    """10 m wind (m/s) -> capacity factor (fraction of rated power)."""
    curve = {k: kw.pop(k) for k in ("cut_in", "rated", "cut_out") if k in kw}
    return capacity_factor(hub_height_speed(v10, **kw), **curve)
