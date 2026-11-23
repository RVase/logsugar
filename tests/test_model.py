"""Checks that model.py implements Eqs. (1)-(4) as written.

Run with `python -m pytest` or `python tests/test_model.py`.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import Params, heaviside, simulate, step  # noqa: E402

P = Params(**json.loads((ROOT / "params.json").read_text()))


def test_one_step_matches_equations():
    A, B, K = 0.3, 0.2, 1.4
    a, b, k = step(A, B, K, P)
    N = A + B
    assert np.isclose(a, max(0, P.r_A * A * (1 - N / K)))
    assert np.isclose(b, max(0, P.r_B * B * (1 - N / K)))
    bonus = P.dK_plus if A / N >= P.theta else 0.0
    assert np.isclose(k, max(P.K_min, K - P.dK_minus + bonus))


def test_heaviside_convention():
    assert heaviside(0.0) == 1.0 and heaviside(1e-12) == 1.0 and heaviside(-1e-12) == 0.0


def test_threshold_at_exactly_theta_gives_bonus():
    p = replace(P, theta=0.5)
    _, _, k = step(0.25, 0.25, 2.0, p)  # R = 0.5 exactly
    assert np.isclose(k, 2.0 - p.dK_minus + p.dK_plus)


def test_non_negativity_on_overshoot():
    a, b, _ = step(2.0, 2.0, 1.0, P)  # N > K
    assert a == 0.0 and b == 0.0


def test_capacity_never_below_K_min():
    out = simulate(replace(P, dK_plus=0.0, dK_minus=0.1, K0=1.5, T=200))
    assert out["K"].min() >= P.K_min


def test_cooperator_defector_ratio_is_geometric():
    # The density factor is shared, so A_t/B_t = (r_A/r_B)^t * A_0/B_0 while both are positive.
    out = simulate(P)
    t = out["t"]
    mask = (out["A"] > 0) & (out["B"] > 0)
    ratio = out["A"][mask] / out["B"][mask]
    expected = (P.r_A / P.r_B) ** t[mask] * (P.A0 / P.B0)
    assert np.allclose(ratio, expected, rtol=1e-9)


def test_simulation_shapes():
    out = simulate(P)
    for key in ("t", "A", "B", "K", "N", "R"):
        assert out[key].shape == (P.T + 1,)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
