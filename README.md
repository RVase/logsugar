# LogSugar — three logistic equations for cooperation, defection and a dynamic carrying capacity

Code for

> Vasellini, R., Cecconi, F., & Vilone, D. *Cooperation, Defection, and the Commons: Reproducing
> an Agent-Based Model of Aksum with Three Logistic Equations.* Submitted to *Games* (MDPI).

It implements Eqs. (1)–(4) of the paper and regenerates every figure computed from them
(Figures 1, 2 and 3b). Figure 3a shows output of the agent-based model of Vasellini et al. (2025),
*Frontiers in Complex Systems* 3:1590952, whose NetLogo code is at
[RVase/aksum-abm-netlogo](https://github.com/RVase/aksum-abm-netlogo).

## The model

Cooperators $A_t$, defectors $B_t$ and carrying capacity $K_t$ evolve as

```math
A_{t+1} = \max\left(0,\; r_A A_t \left(1 - \frac{N_t}{K_t}\right)\right) \qquad (1)
```

```math
B_{t+1} = \max\left(0,\; r_B B_t \left(1 - \frac{N_t}{K_t}\right)\right) \qquad (2)
```

```math
K_{t+1} = \max\left(K_{\min},\; K_t - \delta K + \Delta K\,\Theta(R_t - \theta)\right) \qquad (3)
```

with $N_t = A_t + B_t$, $R_t = A_t/N_t$, and $\Theta(x) = 1$ for $x \ge 0$, $0$ otherwise (4).

Implementation conventions, which the equations leave implicit:

- All three variables at $t+1$ are computed from the state at $t$ (synchronous update).
- $\Theta(0) = 1$: the bonus is paid when $R_t$ equals $\theta$ exactly.
- If the population is extinct ($N_t = 0$), $R_t$ is taken as $0$.

## Install

Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

## Reproducing the figures

```bash
python figures.py
```

writes `figures/fig1`, `fig2` and `fig3b` (PDF and PNG) in about five seconds, and prints the
numbers quoted in the text:

| figure | what is varied | printed check |
|---|---|---|
| 1 | $r_B \in \{1.5, 2.5, 2.995, 3.6\}$ at fixed $r_A/r_B = 0.955$ | $R_t < \theta$ first at $t = 90$ in all four panels |
| 2a | $r_B \in [1, 4]$ at fixed $r_A/r_B$; 1400 iterations, first 1280 discarded | cooperators absent from the long-run state |
| 2b, 2c | $r_A/r_B \in [0.90, 0.99]$ at $r_B = 2.995$ | peak/final 1.11 → 2.37; steps with $R_t \ge \theta$ 40 → 413 |
| 3b | $\Delta K = 4.83 \times 10^{-3}$ vs $\Delta K = 0$ | peak / equilibrium = 1.29 |

## Running the model directly

```bash
python run.py trajectory                          # one run with params.json
python run.py trajectory --set r_A=1.43 --set r_B=1.5
python run.py sweep r_B 1.0 4.0                   # long-run states as r_B varies
```

Results are written to `results/`:

| command | files |
|---|---|
| `trajectory` | `trajectory.csv` (columns `t, A, B, N, K, R`), `trajectory_params.json`, `trajectory.png` |
| `sweep NAME LO HI` | `sweep_NAME.csv` (final `--keep` steps for each of `--n` values), `sweep_NAME.png` |

`--set NAME=VALUE` overrides any entry of `params.json`, `--params FILE` loads another parameter
file, `--out DIR` changes the output folder, and `--no-plot` writes CSV only. The model itself is
in `model.py`: `simulate(Params(...))` returns the arrays.

## Parameters

`params.json` holds Table 1 of the paper. All quantities are in units of $K_{\min} = 1$; the values
are illustrative, not fitted to the agent-based model.

| symbol | name in code | value | meaning |
|---|---|---|---|
| $K_{\min}$ | `K_min` | 1 | baseline carrying capacity (unit) |
| $K_0$ | `K0` | 1 | initial carrying capacity |
| $\delta K$ | `dK_minus` | 1.5 × 10⁻³ | degradation per time step |
| $\Delta K$ | `dK_plus` | 4.83 × 10⁻³ | cooperation bonus per time step |
| $\theta$ | `theta` | 0.5 | PGG success threshold on $R_t$ |
| $r_B$ | `r_B` | 2.995 | defector growth rate |
| $r_A$ | `r_A` | 2.860 | cooperator growth rate ($r_A/r_B = 0.955$) |
| $A_0$ | `A0` | 0.020 | initial cooperators |
| $B_0$ | `B0` | 3.18 × 10⁻⁴ | initial defectors ($R_0 = 0.98$) |
| — | `T` | 500 | number of time steps |

## Tests

```bash
python -m pytest            # or: python tests/test_model.py
```

The tests check one step against Eqs. (1)–(4) written out by hand, the non-negativity and
$K_{\min}$ floors, the $\Theta(0) = 1$ convention, and the identity
$A_t/B_t = (r_A/r_B)^t A_0/B_0$, which follows from (1)–(2) because the density factor is shared.

## Citation

Please cite the paper; see `CITATION.cff`.

## Licence

MIT — see `LICENSE`.
