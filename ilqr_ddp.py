"""iLQR and full DDP."""

from __future__ import annotations
import argparse
import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import acrobot
from acrobot import PROBLEM, Problem, rk4_step

# Step-size ladder for the line search; evaluated in parallel.
ALPHAS = 0.5 ** jnp.arange(12)


def build_solver(prob: Problem, mode: str = "ilqr"):
    """Build a jitted iLQR/DDP solver. Returns solve(u_init, reg0) -> dic"""
    if mode not in ("ilqr", "ddp"):
        raise ValueError("mode must be 'ilqr' or 'ddp'")

    Q = jnp.asarray(prob.Q)
    Qn = jnp.asarray(prob.Qn)
    xg = jnp.asarray(prob.xgoal)
    x0 = jnp.asarray(prob.x0)
    h, R, N, u_max = prob.h, prob.R, prob.N, prob.u_max
    p_phys = prob.params

    step = lambda x, u: rk4_step(x, u, h, p_phys)  # noqa: E731
    w = jnp.ones(N + 1).at[0].set(0.5).at[-1].set(0.5)  # trapezoid weights
    max_iters, tol = 300, 1e-12

    # --- cost ---
    def total_cost(xtraj, useq):
        dx = xtraj - xg[None, :]
        node = 0.5 * jnp.einsum("ni,ij,nj->n", dx, Q, dx)
        return (
            h * jnp.sum(w * node)
            + h * jnp.sum(0.5 * R * useq**2)
            + 0.5 * dx[-1] @ Qn @ dx[-1]
        )

    def rollout(useq):
        def fwd(x, u):
            xn = step(x, u)
            return xn, xn

        _, xs = jax.lax.scan(fwd, x0, useq)
        return jnp.vstack([x0[None, :], xs])

    # --- derivatives ---
    def _sd(z):
        return step(z[:4], z[4])

    hess_step = jax.hessian(_sd)  # (4, 5, 5); only used in DDP mode

    def linearize(xtraj, useq):
        A = jax.vmap(jax.jacfwd(step, 0))(xtraj[:-1], useq)
        B = jax.vmap(jax.jacfwd(step, 1))(xtraj[:-1], useq)
        if mode == "ilqr":
            # Gauss-Newton: the second-order dynamics terms are omitted.
            return (
                A,
                B,
                jnp.zeros((N, 4, 4, 4)),
                jnp.zeros((N, 4, 4)),
                jnp.zeros((N, 4)),
            )
        Z = jnp.concatenate([xtraj[:-1], useq[:, None]], axis=1)
        H = jax.vmap(hess_step)(Z)  # (N, 4, 5, 5)
        return A, B, H[:, :, :4, :4], H[:, :, 4, :4], H[:, :, 4, 4]

    # --- backward pass ---
    def backward(xtraj, useq, A, B, fxx, fux, fuu, reg):
        dxT = xtraj[-1] - xg
        p_T = Qn @ dxT + h * w[-1] * (Q @ dxT)
        P_T = Qn + h * w[-1] * Q

        def bstep(carry, inp):
            p_next, P_next, d1, d2, bad = carry
            x, u, Ak, Bk, Fxx, Fux, Fuu, wk = inp
            dx = x - xg
            lx = (h * wk) * (Q @ dx)
            lu = h * R * u
            lxx = (h * wk) * Q
            luu = h * R

            # Levenberg-Marquardt regularization
            P_reg = P_next + reg * jnp.eye(4)

            gx = lx + Ak.T @ p_next
            gu = lu + Bk @ p_next

            Gxx = lxx + Ak.T @ P_reg @ Ak
            Guu = luu + Bk @ P_reg @ Bk
            Gux = Bk @ P_reg @ Ak

            # Full DDP
            Gxx = Gxx + jnp.einsum("i,ijk->jk", p_next, Fxx)
            Gux = Gux + jnp.einsum("i,ij->j", p_next, Fux)
            Guu = Guu + jnp.dot(p_next, Fuu)

            bad = bad | (Guu <= 1e-12) | ~jnp.isfinite(Guu)
            Guu_safe = jnp.where(Guu > 1e-12, Guu, 1.0)

            d = gu / Guu_safe  # feedforward
            K = Gux / Guu_safe  # feedback (4,)

            p_k = gx - K * gu + K * (Guu * d) - Gux * d
            P_k = Gxx + jnp.outer(K, Guu * K) - jnp.outer(Gux, K) - jnp.outer(K, Gux)
            P_k = 0.5 * (P_k + P_k.T)

            return (p_k, P_k, d1 + d * gu, d2 + d * Guu * d, bad), (d, K, p_k)

        inputs = (xtraj[:-1], useq, A, B, fxx, fux, fuu, w[:-1])
        (_, _, d1, d2, bad), (d, K, p_seq) = jax.lax.scan(
            bstep, (p_T, P_T, 0.0, 0.0, False), inputs, reverse=True
        )
        return d, K, d1, d2, bad, jnp.vstack([p_seq, p_T[None, :]])

    # --- forward pass ---
    def forward(xtraj, useq, d, K, alpha):
        def fstep(x, inp):
            xk, uk, dk, Kk = inp
            u_new = jnp.clip(uk - alpha * dk - Kk @ (x - xk), -u_max, u_max)
            return step(x, u_new), (step(x, u_new), u_new)

        _, (xs, us) = jax.lax.scan(fstep, x0, (xtraj[:-1], useq, d, K))
        return jnp.vstack([x0[None, :], xs]), us

    forward_batch = jax.vmap(forward, in_axes=(None, None, None, None, 0))

    # --- main loop ---
    def solve_traced(u_init, reg0):
        x_init = rollout(u_init)
        J_init = total_cost(x_init, u_init)

        # carry: x, u, J, reg, it, done, n_accept
        def cond(c):
            _, _, _, _, it, done, _ = c
            return (it < max_iters) & ~done

        def body(c):
            xtraj, useq, J, reg, it, done, n_acc = c

            A, B, fxx, fux, fuu = linearize(xtraj, useq)
            d, K, d1, d2, bad, _ = backward(xtraj, useq, A, B, fxx, fux, fuu, reg)

            xs, us = forward_batch(xtraj, useq, d, K, ALPHAS)
            Js = jax.vmap(total_cost)(xs, us)

            expected = ALPHAS * d1 - 0.5 * ALPHAS**2 * d2
            actual = J - Js
            ok = jnp.isfinite(Js) & (expected > 0) & (actual > 1e-4 * expected) & ~bad
            idx = jnp.argmax(ok)  # first acceptable alpha
            accepted = ok[idx]

            xn = jnp.where(accepted, xs[idx], xtraj)
            un = jnp.where(accepted, us[idx], useq)
            Jn = jnp.where(accepted, Js[idx], J)

            rel = jnp.abs(J - Jn) / jnp.maximum(jnp.abs(J), 1.0)
            conv = accepted & (rel < tol) & (expected[idx] < tol * jnp.abs(J) + 1e-12)

            reg_n = jnp.where(
                accepted, jnp.maximum(reg * 0.5, 1e-9), jnp.minimum(reg * 10.0, 1e10)
            )
            failed = ~accepted & (reg >= 1e10)

            return (xn, un, Jn, reg_n, it + 1, conv | failed, n_acc + accepted)

        init = (x_init, u_init, J_init, reg0, 0, False, 0)
        xtraj, useq, J, reg, it, done, n_acc = jax.lax.while_loop(cond, body, init)

        # final backward pass to expose the costate at the solution.
        A, B, fxx, fux, fuu = linearize(xtraj, useq)
        _, _, d1, _, _, p = backward(xtraj, useq, A, B, fxx, fux, fuu, 0.0)
        return xtraj, useq, J, it, n_acc, p, d1, J_init

    return jax.jit(solve_traced)


def solve(prob: Problem = PROBLEM, mode: str = "ilqr", u_init=None, verbose=True):
    solver = build_solver(prob, mode)
    if u_init is None:
        u_init = jnp.zeros(prob.N)
    u_init = jnp.asarray(u_init, dtype=float).reshape(prob.N)

    t0 = time.perf_counter()
    out = solver(u_init, 1e-6)
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = solver(u_init, 1e-6)
    jax.block_until_ready(out)
    wall = time.perf_counter() - t0

    xtraj, useq, J, it, n_acc, p, d1, J_init = out
    res = {
        "x": np.asarray(xtraj),
        "u": np.asarray(useq),
        "lam": np.asarray(p),
        "cost": float(J),
        "cost_init": float(J_init),
        "n_iter": int(it),
        "n_accept": int(n_acc),
        "wall": wall,
        "compile": t_compile,
        "mode": mode,
        "grad_proxy": float(d1),
    }
    if verbose:
        print(
            f"{mode.upper()}  ({'full second-order DDP' if mode == 'ddp' else 'Gauss-Newton iLQR'})"
        )
        print(f"  iterations     : {res['n_iter']} ({res['n_accept']} accepted)")
        print(f"  cost           : {res['cost_init']:.6e}  ->  {res['cost']:.10f}")
        print(f"  final state    : {np.array2string(res['x'][-1], precision=6)}")
        print(f"  |x(T) - xg|    : {np.linalg.norm(res['x'][-1] - prob.xgoal):.3e}")
        print(f"  max |u|        : {np.max(np.abs(res['u'])):.4f}")
        print(f"  wall           : {wall * 1e3:.1f} ms  (+{t_compile:.2f} s JIT)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="iLQR / DDP acrobot swing-up")
    ap.add_argument("--mode", choices=["ilqr", "ddp", "both"], default="both")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-animate", action="store_true")
    args = ap.parse_args()

    modes = ["ilqr", "ddp"] if args.mode == "both" else [args.mode]
    results = {}
    for m in modes:
        results[m] = solve(PROBLEM, mode=m)
        ref = acrobot.evaluate(
            PROBLEM.t_grid,
            np.append(results[m]["u"], results[m]["u"][-1]),
            PROBLEM,
            kind="zoh",
        )
        results[m]["referee"] = ref
        print(
            f"  referee cost   : {ref['cost']:.8f}   "
            f"|x(T)-xg| = {ref['terminal_error']:.3e}"
        )
        print()

    if len(results) == 2:
        du = np.max(np.abs(results["ilqr"]["u"] - results["ddp"]["u"]))
        print(
            f"max |u_iLQR - u_DDP| = {du:.3e}  "
            f"(cost gap {abs(results['ilqr']['cost'] - results['ddp']['cost']):.3e})"
        )

    if not args.no_plot:
        import matplotlib.pyplot as plt
        import os

        os.makedirs("results", exist_ok=True)

        for m in modes:
            r = results[m]
            acrobot.plot_solution(
                PROBLEM.t_grid,
                r["x"],
                PROBLEM.t_grid[:-1],
                r["u"],
                PROBLEM,
                title=f"{m.upper()} acrobot swing-up",
                savepath=f"results/{m}.png",
            )
        ani = None
        if not args.no_animate:
            r = results[modes[-1]]
            ani = acrobot.animate(
                PROBLEM.t_grid, r["x"], PROBLEM, title=f"{modes[-1].upper()} swing-up"
            )
        plt.show()
