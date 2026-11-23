"""Regenerate Figures 1, 2 and 3(b) of the paper from Eqs. (1)-(4).

    python figures.py            # writes figures/fig1.pdf, fig2.pdf, fig3b.pdf (+ .png)

All runs use params.json (Table 1); departures are those stated in the figure captions.
Figure 3(a) shows output of the agent-based model (Vasellini et al., 2025) and is not
produced here.
"""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import Params, simulate

HERE = Path(__file__).resolve().parent
OUT = Path("figures")

BLUE, ORANGE, GREEN, GREY, PINK = "#0072B2", "#D55E00", "#009E73", "#888888", "#CC79A7"


def load():
    return Params(**json.loads((HERE / "params.json").read_text()))


def crossing_step(out, theta):
    """First t at which R_t < theta (the bonus switches off from t+1)."""
    below = np.nonzero(out["R"] < theta)[0]
    return int(below[0]) if below.size else None


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf / .png")


# --------------------------------------------------------------------------- Figure 1
def figure1(p):
    """Temporal dynamics for four r_B at fixed r_A / r_B."""
    ratio = p.r_A / p.r_B
    r_Bs = [1.5, 2.5, p.r_B, 3.6]
    fig = plt.figure(figsize=(10, 7.5))
    outer = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.18)
    for i, rB in enumerate(r_Bs):
        q = replace(p, r_B=rB, r_A=ratio * rB)
        out = simulate(q)
        tc = crossing_step(out, q.theta)
        inner = outer[i // 2, i % 2].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        ax.plot(out["t"], out["K"], "k--", lw=1.2, label="capacity $K_t$")
        ax.plot(out["t"], out["B"], color=ORANGE, lw=1, label="defectors $B_t$")
        ax.plot(out["t"], out["A"], color=BLUE, lw=1, label="cooperators $A_t$")
        axr.plot(out["t"], out["R"], color=GREEN, lw=1.2)
        axr.axhline(q.theta, color=GREY, lw=0.8, ls="-.")
        for a in (ax, axr):
            if tc is not None:
                a.axvline(tc, color=PINK, lw=0.8, ls=":")
        ax.set_title(f"({'abcd'[i]}) $r_B = {rB:g}$, $r_A = {q.r_A:.3g}$", loc="left", fontsize=10)
        ax.set_ylim(0, 1.4)
        axr.set_ylim(0, 1.05)
        axr.set_ylabel("$R_t$")
        plt.setp(ax.get_xticklabels(), visible=False)
        if i % 2 == 0:
            ax.set_ylabel("population, capacity [$K_{min}$]")
        if i // 2 == 1:
            axr.set_xlabel("time step $t$")
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="center right")
        print(f"  Fig 1({'abcd'[i]}): r_B={rB}, R_t < theta first at t={tc}")
    save(fig, "fig1")


# --------------------------------------------------------------------------- Figure 2
def figure2(p, n_rB=600, T=1400, discard=1280, n_cost=60):
    """(a) long-run state vs r_B at fixed r_A/r_B; (b, c) peak/final and bonus duration vs cost."""
    ratio = p.r_A / p.r_B
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.22)

    # (a) bifurcation diagram
    ax = fig.add_subplot(gs[0, :])
    xs, As, Bs = [], [], []
    for rB in np.linspace(1.0, 4.0, n_rB):
        out = simulate(replace(p, r_B=rB, r_A=ratio * rB, T=T))
        keep = slice(discard, T + 1)
        k = T + 1 - discard
        xs.append(np.full(k, rB))
        As.append(out["A"][keep])
        Bs.append(out["B"][keep])
    xs, As, Bs = map(np.concatenate, (xs, As, Bs))
    ax.scatter(xs, Bs, s=0.2, lw=0, color=ORANGE, label="defectors $B_t$")
    ax.scatter(xs, As, s=0.2, lw=0, color=BLUE, label="cooperators $A_t$")
    ax.axvline(3.0, color=GREY, lw=0.8, ls="--")
    ax.set_xlim(1, 4)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"growth rate $r_B$  (at fixed $r_A/r_B = {ratio:.3g}$)")
    ax.set_ylabel("long-run population [$K_{min}$]")
    ax.set_title("(a) Long-run state of the system", loc="left", fontsize=10)
    ax.legend(frameon=False, markerscale=15, fontsize=8, loc="upper left")
    print(f"  Fig 2(a): max long-run cooperators = {As.max():.3g}")

    # (b), (c) sensitivity to the cost of cooperation at fixed r_B
    costs = np.linspace(0.90, 0.99, n_cost)
    peak_over_final, steps_on = [], []
    for c in costs:
        # long enough for K to return to K_min and N to settle
        out = simulate(replace(p, r_A=c * p.r_B, T=4000))
        final = out["N"][-200:].mean()
        peak_over_final.append(out["N"].max() / final)
        steps_on.append(crossing_step(out, p.theta))
    axb = fig.add_subplot(gs[1, 0])
    axb.plot(costs, peak_over_final, color=PINK)
    axb.set_xlabel("cost of cooperation $r_A/r_B$")
    axb.set_ylabel("peak population / final")
    axb.set_title("(b) A cheaper dilemma buys a bigger boom ...", loc="left", fontsize=10)
    axc = fig.add_subplot(gs[1, 1])
    axc.plot(costs, steps_on, color=PINK)
    axc.set_xlabel("cost of cooperation $r_A/r_B$")
    axc.set_ylabel(r"steps with $R_t \geq \theta$")
    axc.set_title("(c) ... and buys it for longer", loc="left", fontsize=10)
    print(
        f"  Fig 2(b): peak/final from {peak_over_final[0]:.3f} to {peak_over_final[-1]:.3f}; "
        f"Fig 2(c): steps from {steps_on[0]} to {steps_on[-1]}"
    )
    save(fig, "fig2")


# --------------------------------------------------------------------------- Figure 3(b)
def figure3b(p):
    """Total population with the cooperation bonus and with Delta K = 0."""
    on = simulate(p)
    off = simulate(replace(p, dK_plus=0.0))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(off["t"], off["N"], color=ORANGE, lw=1, label="without PGG ($\\Delta K = 0$)")
    ax.plot(on["t"], on["N"], color=BLUE, lw=1, label="with PGG")
    final = off["N"][-100:].mean()
    ratio = on["N"].max() / final
    ax.axhline(final, color=GREY, lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("generations (arbitrary units)")
    ax.set_ylabel("population [$K_{min}$]")
    ax.set_title(f"(b) Three equations — peak / equilibrium = {ratio:.2f}×", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    print(f"  Fig 3(b): peak / equilibrium = {ratio:.3f}")
    save(fig, "fig3b")


if __name__ == "__main__":
    params = load()
    figure1(params)
    figure2(params)
    figure3b(params)
