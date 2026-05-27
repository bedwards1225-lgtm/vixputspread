"""Strike selection monotonicity: higher target delta -> higher (less OTM) put strike."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from vrp.pricing.surface import fit_surface
from vrp.signals.entry import _short_strike_for_delta


def _flat_surface(spot=500.0):
    as_of = date(2024, 1, 2)
    expiry = as_of + timedelta(days=30)
    strikes = np.arange(spot * 0.6, spot * 1.4, 1.0)
    rows = [{"strike": float(k), "expiry": expiry, "iv": 0.20} for k in strikes]
    return fit_surface(pd.DataFrame(rows), as_of=as_of, spot=spot, rate=0.04, div_yield=0.0), expiry


def test_strike_monotonic_in_target_delta():
    surface, expiry = _flat_surface()
    deltas = [0.10, 0.15, 0.20, 0.25]
    strikes = [
        _short_strike_for_delta(surface, expiry, d, rate=0.04, div_yield=0.0)[0]
        for d in deltas
    ]
    # Higher target |delta| means LESS OTM put -> HIGHER strike.
    assert strikes == sorted(strikes), strikes
