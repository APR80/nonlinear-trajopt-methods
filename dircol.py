"""Hermite-Simpson direct collocation"""

from __future__ import annotations
import argparse
import time
import casadi as ca
import numpy as np
import acrobot
from acrobot import PROBLEM, Problem, f_casadi


def _dyn_function(prob: Problem) -> ca.Function:
    x = ca.MX.sym("x", 4)
    u = ca.MX.sym("u")
    return ca.Function("f", [x, u], [f_casadi(x, u, prob.params)], ["x", "u"], ["xdot"])


def _initial_guess(prob: Problem, n_mid: int):
    """A swing-up-shaped guess: ramp theta1 up, leave the rest at zero.

    All-zeros is a poor guess here because it sits exactly at the unstable-free
    hanging equilibrium, where the gravity gradient vanishes.
    """
    N, t = prob.N, prob.t_grid
    s = t / prob.T
    X = np.zeros((4, N + 1))
    X[0] = np.pi * (3 * s**2 - 2 * s**3)  # smooth 0 -> pi
    X[2] = np.gradient(X[0], t)
    U = np.zeros(N + 1)
    Xm = 0.5 * (X[:, :-1] + X[:, 1:]) if n_mid else None
    return X, U, Xm


def build(prob: Problem, form: str = "separated", hessian: str = "exact"):
    """Build the NLP."""
    N, h = prob.N, prob.h
    f = _dyn_function(prob)
    xg = prob.xgoal

    # --- decision variables ---
    X = ca.MX.sym("X", 4, N + 1)  # states at nodes
    U = ca.MX.sym("U", 1, N + 1)  # controls at nodes (u is continuous)
    variables = [ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)]

    x1, x2 = X[:, :-1], X[:, 1:]
    u1, u2 = U[:, :-1], U[:, 1:]
    um = 0.5 * (u1 + u2)

    fmap = f.map(N)
    f1 = fmap(x1, u1)
    f2 = fmap(x2, u2)

    if form == "separated":
        Xm = ca.MX.sym("Xm", 4, N)  # midpoint states, explicit
        variables.append(ca.reshape(Xm, -1, 1))
        fm = fmap(Xm, um)
        interp = Xm - (0.5 * (x1 + x2) + (h / 8.0) * (f1 - f2))
        simpson = (x2 - x1) - (h / 6.0) * (f1 + 4.0 * fm + f2)
        defects = ca.vertcat(ca.reshape(interp, -1, 1), ca.reshape(simpson, -1, 1))
    elif form == "compressed":
        Xm = 0.5 * (x1 + x2) + (h / 8.0) * (f1 - f2)
        fm = fmap(Xm, um)
        xdot_m = (-1.5 / h) * (x1 - x2) - 0.25 * (f1 + f2)
        defects = ca.reshape(fm - xdot_m, -1, 1)
    else:
        raise ValueError("form must be 'separated' or 'compressed'")

    w = ca.vertcat(*variables)

    Q, R, Qn = ca.DM(prob.Q), prob.R, ca.DM(prob.Qn)

    def stage(xs, us):
        d = xs - ca.repmat(ca.DM(xg), 1, xs.shape[1])
        return 0.5 * ca.sum1(d * (Q @ d)) + 0.5 * R * us**2

    g1, g2, gm = stage(x1, u1), stage(x2, u2), stage(Xm, um)
    J = (h / 6.0) * ca.sum2(g1 + 4.0 * gm + g2)
    dT = X[:, -1] - ca.DM(xg)
    J = J + 0.5 * ca.dot(dT, Qn @ dT)

    # --- constraints: dynamics + initial condition ---
    g = ca.vertcat(defects, X[:, 0] - ca.DM(prob.x0))

    nlp = {"x": w, "f": J, "g": g}
    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-10,
        "ipopt.acceptable_tol": 1e-8,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
        "ipopt.sb": "yes",
    }
    if hessian == "limited-memory":
        opts["ipopt.hessian_approximation"] = "limited-memory"
    solver = ca.nlpsol("dircol", "ipopt", nlp, {**opts, "expand": True})

    n_mid = 4 * N if form == "separated" else 0
    sizes = {
        "nx": 4 * (N + 1),
        "nu": N + 1,
        "nm": n_mid,
        "nw": w.numel(),
        "ng": g.numel(),
    }
    return solver, sizes


def solve(
    prob: Problem = PROBLEM,
    form: str = "separated",
    hessian: str = "exact",
    verbose: bool = True,
):
    """Solve the canonical problem by Hermite-Simpson collocation."""
    t_build = time.perf_counter()
    solver, sz = build(prob, form, hessian)
    t_build = time.perf_counter() - t_build

    N = prob.N
    nx, nu, nm = sz["nx"], sz["nu"], sz["nm"]
    Xg, Ug, Xmg = _initial_guess(prob, nm)

    w0 = np.concatenate(
        [Xg.reshape(-1, order="F"), Ug.reshape(-1)]
        + ([Xmg.reshape(-1, order="F")] if nm else [])
    )
    lbw = np.full(sz["nw"], -np.inf)
    ubw = np.full(sz["nw"], np.inf)
    lbw[nx : nx + nu] = -prob.u_max  # torque bounds, both forms
    ubw[nx : nx + nu] = prob.u_max

    t0 = time.perf_counter()
    sol = solver(
        x0=w0, lbx=lbw, ubx=ubw, lbg=np.zeros(sz["ng"]), ubg=np.zeros(sz["ng"])
    )
    wall = time.perf_counter() - t0
    stats = solver.stats()

    wv = np.asarray(sol["x"]).reshape(-1)
    X = wv[:nx].reshape((4, N + 1), order="F")
    U = wv[nx : nx + nu]

    res = {
        "x": X.T,
        "u": U,
        "cost": float(sol["f"]),
        "n_iter": stats.get("iter_count", -1),
        "status": stats.get("return_status", "?"),
        "wall": wall,
        "build": t_build,
        "form": form,
    }
    if verbose:
        print(
            f"DIRCOL ({form}, Hermite-Simpson, {sz['nw']} vars, {sz['ng']} constraints)"
        )
        print(f"  status         : {res['status']}  ({res['n_iter']} iterations)")
        print(f"  cost           : {res['cost']:.10f}")
        print(f"  final state    : {np.array2string(res['x'][-1], precision=6)}")
        print(f"  |x(T) - xg|    : {np.linalg.norm(res['x'][-1] - prob.xgoal):.3e}")
        print(f"  max |u|        : {np.max(np.abs(res['u'])):.4f}")
        print(
            f"  wall           : {wall * 1e3:.1f} ms  (+{t_build * 1e3:.0f} ms build)"
        )
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hermite-Simpson direct collocation")
    ap.add_argument(
        "--form", choices=["separated", "compressed", "both"], default="both"
    )
    ap.add_argument("--hessian", choices=["exact", "limited-memory"], default="exact")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-animate", action="store_true")
    args = ap.parse_args()

    forms = ["separated", "compressed"] if args.form == "both" else [args.form]
    out = {}
    for fm in forms:
        out[fm] = solve(PROBLEM, form=fm, hessian=args.hessian)
        ref = acrobot.evaluate(PROBLEM.t_grid, out[fm]["u"], PROBLEM, kind="linear")
        out[fm]["referee"] = ref
        print(
            f"  referee cost   : {ref['cost']:.8f}   "
            f"|x(T)-xg| = {ref['terminal_error']:.3e}"
        )
        print()

    if len(out) == 2:
        du = np.max(np.abs(out["separated"]["u"] - out["compressed"]["u"]))
        dx = np.max(np.abs(out["separated"]["x"] - out["compressed"]["x"]))
        print(
            f"separated vs compressed:  max|du| = {du:.3e}   max|dx| = {dx:.3e}"
            f"   cost gap = {abs(out['separated']['cost'] - out['compressed']['cost']):.3e}"
        )

    if not args.no_plot:
        import matplotlib.pyplot as plt
        import os

        os.makedirs("results", exist_ok=True)

        for fm in forms:
            r = out[fm]
            acrobot.plot_solution(
                PROBLEM.t_grid,
                r["x"],
                PROBLEM.t_grid,
                r["u"],
                PROBLEM,
                title=f"Direct collocation ({fm})",
                savepath="results/dircol.png"
                if fm == forms[-1]
                else f"results/dircol_{fm}.png",
            )
        ani = None
        if not args.no_animate:
            ani = acrobot.animate(
                PROBLEM.t_grid,
                out[forms[-1]]["x"],
                PROBLEM,
                title="Direct collocation swing-up",
            )
        plt.show()
