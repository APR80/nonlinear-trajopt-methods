"""Run every method compare them.
Each solver reports its own cost, which is computed on its own discretization --
RK4 + trapezoid for the shooting methods, Hermite-Simpson for collocation, an
adaptive mesh for the PMP boundary-value solve.  Those numbers are not
comparable: a coarse transcription can report a lower cost than it actually
achieves.

So every method is additionally scored by acrobot.evaluate, which takes only
its control signal, integrates the true continuous dynamics with DOP853 at
rtol=1e-12, and evaluates the continuous cost.  That column is the one to
compare.

"""

from __future__ import annotations
import argparse
import time
import numpy as np
import acrobot
from acrobot import PROBLEM, Problem


def _referee(res, prob: Problem):
    """Score one solver's control on the continuous problem."""
    u = np.asarray(res["u"]).reshape(-1)
    if len(u) == prob.N:  # one control per interval: ZOH
        return acrobot.evaluate(prob.t_grid, np.append(u, u[-1]), prob, kind="zoh")
    return acrobot.evaluate(res.get("t", prob.t_grid), u, prob, kind="linear")


def run_all(prob: Problem = PROBLEM, skip=()):
    import dircol
    import ilqr_ddp

    rows = []

    def add(name, res):
        res["referee"] = _referee(res, prob)
        res["name"] = name
        rows.append(res)
        print(f"  {name:<22s} done  ({res['wall'] * 1e3:8.1f} ms)")

    print("running solvers...")
    if "ilqr" not in skip:
        add("iLQR", ilqr_ddp.solve(prob, mode="ilqr", verbose=False))
    if "ddp" not in skip:
        add("DDP (2nd order)", ilqr_ddp.solve(prob, mode="ddp", verbose=False))
    if "dircol" not in skip:
        add("Dircol (separated)", dircol.solve(prob, form="separated", verbose=False))
        add("Dircol (compressed)", dircol.solve(prob, form="compressed", verbose=False))
    if "pmp" not in skip:
        import pmp

        add("PMP (indirect)", pmp.solve(prob, verbose=False))
    return rows


def print_table(rows, prob: Problem):
    w = (22, 10, 7, 15, 15, 12)
    hdr = ("method", "wall [ms]", "iters", "own cost", "referee cost", "|x(T)-xg|")
    line = "  ".join(h.ljust(n) for h, n in zip(hdr, w))
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        ref = r["referee"]
        cells = (
            r["name"],
            f"{r['wall'] * 1e3:.1f}",
            str(r.get("n_iter", "-")),
            f"{r['cost']:.8f}",
            f"{ref['cost']:.8f}",
            f"{ref['terminal_error']:.3e}",
        )
        print("  ".join(c.ljust(n) for c, n in zip(cells, w)))
    print("-" * len(line))

    refs = np.array([r["referee"]["cost"] for r in rows])
    spread = (refs.max() - refs.min()) / refs.min()
    print(
        f"referee-cost spread across methods: {spread:.3e} relative "
        f"(min {refs.min():.8f}, max {refs.max():.8f})"
    )
    print(f"grid: T={prob.T}s  h={prob.h}s  N={prob.N}  |u|<={prob.u_max}  R={prob.R}")


def refinement_study(prob: Problem, hs=(0.05, 0.025, 0.0125, 0.00625)):
    """Show that every method converges to the same continuous optimum."""
    import dircol
    import ilqr_ddp
    import pmp

    print("\nmesh refinement -- referee cost of each method as h -> 0")
    print(
        f"  {'h':<10s} {'N':<6s} {'iLQR':<14s} {'DDP':<14s} {'dircol':<14s} "
        f"{'PMP':<14s} {'spread':<10s}"
    )
    print("  " + "-" * 86)
    for h in hs:
        p = prob.replace(h=h)
        c = [
            _referee(ilqr_ddp.solve(p, mode="ilqr", verbose=False), p)["cost"],
            _referee(ilqr_ddp.solve(p, mode="ddp", verbose=False), p)["cost"],
            _referee(dircol.solve(p, form="compressed", verbose=False), p)["cost"],
            _referee(pmp.solve_stage2(p, verbose=False), p)["cost"],
        ]
        print(
            f"  {h:<10.5f} {p.N:<6d} "
            + " ".join(f"{v:<14.8f}" for v in c)
            + f"{(max(c) - min(c)) / min(c):<10.2e}"
        )


def overlay(rows, prob: Problem, savepath="results/compare.png"):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for r in rows:
        x = np.asarray(r["x"])
        u = np.asarray(r["u"]).reshape(-1)
        t = np.asarray(r.get("t", prob.t_grid))
        tu = t[:-1] if len(u) == len(t) - 1 else t
        axs[0].plot(t, x[:, 0], lw=1.4, label=r["name"])
        axs[1].plot(t, x[:, 1], lw=1.4, label=r["name"])
        axs[2].plot(tu, u, lw=1.4, label=r["name"])

    axs[0].axhline(np.pi, color="k", ls="--", lw=1)
    axs[0].set_ylabel(r"$\theta_1$ [rad]")
    axs[1].axhline(0.0, color="k", ls="--", lw=1)
    axs[1].set_ylabel(r"$\theta_2$ [rad]")
    axs[2].axhline(prob.u_max, color="r", ls="--", lw=1)
    axs[2].axhline(-prob.u_max, color="r", ls="--", lw=1)
    axs[2].set_ylabel("torque [N·m]")
    axs[2].set_xlabel("time [s]")
    for a in axs:
        a.grid(True, alpha=0.3)
    axs[0].legend(loc="best", fontsize=8)
    fig.suptitle("Acrobot swing-up: all methods on the canonical problem")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="compare all trajopt methods")
    ap.add_argument("--h", type=float, default=None, help="override the grid spacing")
    ap.add_argument("--skip", nargs="*", default=[], help="solvers to skip")
    ap.add_argument(
        "--refine", action="store_true", help="also run the refinement study"
    )
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-animate", action="store_true")
    args = ap.parse_args()

    prob = PROBLEM if args.h is None else PROBLEM.replace(h=args.h)
    t0 = time.perf_counter()
    rows = run_all(prob, skip=args.skip)
    print_table(rows, prob)
    if args.refine:
        refinement_study(prob)
    print(f"\ntotal wall time: {time.perf_counter() - t0:.1f} s")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        overlay(rows, prob)
        ani = None
        if not args.no_animate:
            r = rows[-1]
            ani = acrobot.animate(
                np.asarray(r.get("t", prob.t_grid)),
                np.asarray(r["x"]),
                prob,
                title=f"{r['name']} swing-up",
            )
        plt.show()
