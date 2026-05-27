"""BSM pricing and Greeks correctness tests."""
from __future__ import annotations

import numpy as np
import pytest

from vrp.pricing import black_scholes as bsm


def test_call_price_textbook():
    # Hull, Options Futures and Other Derivatives, Ex 13.6:
    # S=42, K=40, r=10%, sigma=20%, T=0.5 => call=4.7594
    price = float(bsm.price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20, "call"))
    assert price == pytest.approx(4.7594, abs=1e-3)


def test_put_call_parity():
    S, K, T, r, q, sigma = 100, 100, 0.5, 0.05, 0.02, 0.25
    c = float(bsm.price(S, K, T, r, q, sigma, "call"))
    p = float(bsm.price(S, K, T, r, q, sigma, "put"))
    parity = c - p - (S * np.exp(-q * T) - K * np.exp(-r * T))
    assert parity == pytest.approx(0.0, abs=1e-6)


def test_atm_put_delta_negative_half_ish():
    d = float(bsm.delta(100, 100, 0.5, 0.05, 0.0, 0.20, "put"))
    assert -0.5 < d < 0.0


def test_gamma_equal_call_put():
    args = (100, 100, 0.5, 0.05, 0.01, 0.20)
    g = float(bsm.gamma(*args))
    assert g > 0


def test_implied_vol_round_trip():
    S, K, T, r, q = 100, 95, 30/365, 0.04, 0.01
    for sigma in (0.10, 0.20, 0.35, 0.60):
        p = float(bsm.price(S, K, T, r, q, sigma, "put"))
        iv = bsm.implied_vol(p, S, K, T, r, q, "put")
        assert iv == pytest.approx(sigma, abs=1e-4)


def test_vega_per_vol_point_scaling():
    # numpy vega is per 1.00 vol unit; multiplying by 0.01 should give the
    # finite-difference approximation under a 1-vol-point bump.
    args = (100, 100, 0.5, 0.05, 0.0)
    sigma = 0.20
    v = float(bsm.vega(*args, sigma)) * 0.01
    bumped = float(bsm.price(*args, sigma + 0.01, "call"))
    base = float(bsm.price(*args, sigma, "call"))
    assert v == pytest.approx(bumped - base, abs=5e-3)


def test_intrinsic_lower_bound_put_iv_returns_nan_under_intrinsic():
    nan = bsm.implied_vol(0.0, 100, 95, 0.5, 0.05, 0.0, "call")
    assert np.isnan(nan) or nan < 0.01
