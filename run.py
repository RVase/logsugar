"""Command-line entry point: simulate the model and write the results.

    python run.py trajectory                   # one run with params.json
    python run.py trajectory --set r_A=1.5     # override any parameter
    python run.py sweep r_A 1.0 4.0           # long-run states vs one parameter

Outputs go to ./results/ as CSV (always) and PNG (unless --no-plot).
"""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from model import Params, simulate

HERE = Path(__file__).resolve().parent


def load_params(path, overrides):
    with open(path) as f:
        p = Params(**json.load(f))
    for item in overrides or []:
        key, value = item.split("=", 1)
        if key not in p.as_dict():
            raise SystemExit(f"unknown parameter: {key}")
        p = replace(p, **{key: int(value) if key == "T" else float(value)})
    return p


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def cmd_trajectory(args):
    p = load_params(args.params, args.set)
    out = simulate(p)
    cols = ["t", "A", "B", "N", "K", "R"]
    path = args.out / "trajectory.csv"
    write_csv(path, cols, zip(*(out[c] for c in cols)))
    (args.out / "trajectory_params.json").write_text(json.dumps(p.as_dict(), indent=2))
    print(f"wrote {path}")
    if not args.no_plot:
        import matplotlib.pyplot as plt

        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax.plot(out["t"], out["A"], label="Cooperators $A_t$", color="#0072B2")
        ax.plot(out["t"], out["B"], label="Defectors $B_t$", color="#D55E00", ls="--")
        ax.plot(out["t"], out["K"], label="Carrying capacity $K_t$", color="#009E73", ls=":")
        ax.set_ylabel("population / capacity")
        ax.legend(frameon=False)
        axr.plot(out["t"], out["R"], color="black")
        axr.axhline(p.theta, color="grey", lw=0.8, ls=":")
        axr.set_ylabel("$R_t$")
        axr.set_xlabel("time step $t$")
        fig.tight_layout()
        fig.savefig(args.out / "trajectory.png", dpi=200)
        print(f"wrote {args.out / 'trajectory.png'}")


def cmd_sweep(args):
    p = load_params(args.params, args.set)
    if args.name not in p.as_dict() or args.name == "T":
        raise SystemExit(f"cannot sweep: {args.name}")
    values = np.linspace(args.lo, args.hi, args.n)
    rows = []
    for v in values:
        out = simulate(replace(p, **{args.name: float(v)}))
        for t in range(p.T + 1 - args.keep, p.T + 1):
            rows.append((v, t, out["A"][t], out["B"][t], out["N"][t], out["K"][t]))
    path = args.out / f"sweep_{args.name}.csv"
    write_csv(path, [args.name, "t", "A", "B", "N", "K"], rows)
    print(f"wrote {path}  (last {args.keep} of {p.T} steps per value)")
    if not args.no_plot:
        import matplotlib.pyplot as plt

        arr = np.array(rows)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(arr[:, 0], arr[:, 3], s=0.3, lw=0, color="#D55E00", label="Defectors $B$")
        ax.scatter(arr[:, 0], arr[:, 2], s=0.3, lw=0, color="#0072B2", label="Cooperators $A$")
        ax.set_xlabel(args.name)
        ax.set_ylabel(f"state over the last {args.keep} steps")
        ax.legend(frameon=False, markerscale=15)
        fig.tight_layout()
        fig.savefig(args.out / f"sweep_{args.name}.png", dpi=200)
        print(f"wrote {args.out / f'sweep_{args.name}.png'}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--params", default=HERE / "params.json", type=Path)
    common.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a parameter (repeatable)")
    common.add_argument("--out", default=Path("results"), type=Path)
    common.add_argument("--no-plot", action="store_true")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("trajectory", parents=[common], help="single run")

    sw = sub.add_parser("sweep", parents=[common], help="long-run states as one parameter varies")
    sw.add_argument("name")
    sw.add_argument("lo", type=float)
    sw.add_argument("hi", type=float)
    sw.add_argument("--n", type=int, default=600, help="number of parameter values")
    sw.add_argument("--keep", type=int, default=100, help="final steps recorded per value")

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"trajectory": cmd_trajectory, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
