"""Vol surface fit/evaluate round-trip."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from vrp.pricing.surface import fit_surface


def test_surface_round_trip_small_error():
    as_of = date(2024, 1, 2)
    expiries = [as_of + timedelta(days=d) for d in (14, 30, 60)]
    rows = []
    spot = 100.0
    for exp in expiries:
        for strike in np.arange(80, 121, 2):
            log_money = np.log(strike / spot)
            # Known smile shape: parabolic in log-money + linear skew.
            iv = 0.18 + 0.10 * log_money**2 - 0.05 * log_money
            rows.append({"strike": strike, "expiry": exp, "iv": iv})
    chain = pd.DataFrame(rows)

    surface = fit_surface(chain, as_of=as_of, spot=spot, rate=0.04, div_yield=0.01)

    # Round-trip: evaluate at the same strikes/expiries; error should be tiny.
    for _, row in chain.iterrows():
        iv_hat = surface.iv(row["strike"], row["expiry"])
        assert abs(iv_hat - row["iv"]) < 0.005, f"{row['strike']} {row['expiry']}: {iv_hat} vs {row['iv']}"


def test_surface_extrapolation_holds_flat():
    as_of = date(2024, 1, 2)
    rows = [
        {"strike": 95, "expiry": as_of + timedelta(days=30), "iv": 0.22},
        {"strike": 100, "expiry": as_of + timedelta(days=30), "iv": 0.20},
        {"strike": 105, "expiry": as_of + timedelta(days=30), "iv": 0.18},
    ]
    surface = fit_surface(pd.DataFrame(rows), as_of=as_of, spot=100, rate=0.04, div_yield=0.0)
    # Far-OTM strike outside the range should hold flat at the boundary IV.
    iv_far_otm = surface.iv(50, as_of + timedelta(days=30))
    assert iv_far_otm == pytest.approx(0.22, abs=1e-6)
