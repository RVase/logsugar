"""Discrete-time logistic model of cooperation, defection and a dynamic carrying capacity.

Implements Eqs. (1)-(4) of the paper, and nothing else:

    A_{t+1} = max(0, r_A * A_t * (1 - N_t / K_t))                       (1)
    B_{t+1} = max(0, r_B * B_t * (1 - N_t / K_t))                       (2)
    K_{t+1} = max(K_min, K_t - dK_minus + dK_plus * H(R_t - theta))     (3)
    H(x)    = 1 if x >= 0 else 0                                        (4)

with N_t = A_t + B_t and R_t = A_t / N_t. Here dK_minus is the degradation
rate (delta K in the paper) and dK_plus the cooperation bonus (Delta K).

All three variables at t+1 are computed from the state at t (synchronous update).
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Params:
    r_A: float        # growth rate of cooperators
    r_B: float        # growth rate of defectors (r_B >= r_A)
    K_min: float      # baseline carrying capacity
    dK_minus: float   # delta K: degradation per step
    dK_plus: float    # Delta K: cooperation bonus per step
    theta: float      # cooperation threshold on R_t
    A0: float         # initial cooperators
    B0: float         # initial defectors
    K0: float         # initial carrying capacity
    T: int            # number of steps

    def as_dict(self):
        return asdict(self)


def heaviside(x):
    """Eq. (4): H(x) = 1 for x >= 0, 0 otherwise."""
    return 1.0 if x >= 0 else 0.0


def step(A, B, K, p):
    """One synchronous update of Eqs. (1)-(3). Returns (A', B', K')."""
    N = A + B
    density = 1.0 - N / K
    A_next = max(0.0, p.r_A * A * density)
    B_next = max(0.0, p.r_B * B * density)
    # R_t is undefined when N_t = 0; an extinct population has no cooperators,
    # so we take R_t = 0 and the bonus is off.
    R = A / N if N > 0 else 0.0
    K_next = max(p.K_min, K - p.dK_minus + p.dK_plus * heaviside(R - p.theta))
    return A_next, B_next, K_next


def simulate(p):
    """Iterate the map for p.T steps.

    Returns a dict of arrays of length T+1 (t = 0..T):
    t, A, B, K, N = A + B, R = A / N (0 where N = 0).
    """
    A = np.empty(p.T + 1)
    B = np.empty(p.T + 1)
    K = np.empty(p.T + 1)
    A[0], B[0], K[0] = p.A0, p.B0, p.K0
    for t in range(p.T):
        A[t + 1], B[t + 1], K[t + 1] = step(A[t], B[t], K[t], p)
    N = A + B
    R = np.divide(A, N, out=np.zeros_like(A), where=N > 0)
    return {"t": np.arange(p.T + 1), "A": A, "B": B, "K": K, "N": N, "R": R}
