"""Pontryagin Maximum Principle (indirect method)."""

from __future__ import annotations
import argparse
import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
import acrobot
from acrobot import PROBLEM, Problem, Minv_B, f, rk4_step  # noqa: E402


# Stage 1: discrete adjoint sweep (forward-backward) + quasi-Newton
def build_adjoint_sweep(prob: Problem):
    """Return a jitted forward-backward PMP sweep for the discrete problem."""
    Q = jnp.asarray(prob.Q)
    Qn = jnp.asarray(prob.Qn)
    xg = jnp.asarray(prob.xgoal)
    x0 = jnp.asarray(prob.x0)
    h, R, N = prob.h, prob.R, prob.N
    p_phys = prob.params

    step = lambda x, u: rk4_step(x, u, h, p_phys)  # noqa: E731
    A_fn = jax.jacfwd(step, 0)
    B_fn = jax.jacfwd(step, 1)

    # Trapezoid weights for the state running cost
    w = jnp.ones(N + 1).at[0].set(0.5).at[-1].set(0.5)

    @jax.jit
    def sweep(useq):
        useq = jnp.reshape(useq, (N,))

        def fwd(x, u):
            xn = step(x, u)
            return xn, xn

        _, xs = jax.lax.scan(fwd, x0, useq)
        xtraj = jnp.vstack([x0[None, :], xs])

        # Linearizations for the whole trajectory in one batched call.
        A = jax.vmap(A_fn)(xtraj[:-1], useq)  # (N, 4, 4)
        B = jax.vmap(B_fn)(xtraj[:-1], useq)  # (N, 4)

        dx = xtraj - xg[None, :]
        node_cost = 0.5 * jnp.einsum("ni,ij,nj->n", dx, Q, dx)
        J = (
            h * jnp.sum(w * node_cost)
            + h * jnp.sum(0.5 * R * useq**2)
            + 0.5 * dx[-1] @ Qn @ dx[-1]
        )

        # p_k = dJ/dx_k.
        p_N = Qn @ dx[-1] + h * w[-1] * (Q @ dx[-1])

        def bwd(p_next, inp):
            dxk, uk, Ak, Bk, wk = inp
            grad_k = h * R * uk + Bk @ p_next  # dJ/du_k = h * H_u
            p_k = h * wk * (Q @ dxk) + Ak.T @ p_next
            return p_k, (p_k, grad_k)

        inputs = (dx[:-1], useq, A, B, w[:-1])
        _, (p_seq, grad) = jax.lax.scan(bwd, p_N, inputs, reverse=True)
        ptraj = jnp.vstack([p_seq, p_N[None, :]])

        return J, grad, xtraj, ptraj

    return sweep


def _projected_grad(grad, useq, u_max):
    """At a bound only the inward gradient component counts."""
    proj = np.asarray(grad).copy()
    at_lo = useq <= -u_max + 1e-12
    at_hi = useq >= u_max - 1e-12
    proj[at_lo] = np.minimum(proj[at_lo], 0.0)
    proj[at_hi] = np.maximum(proj[at_hi], 0.0)
    return proj


def solve_stage1(prob: Problem = PROBLEM, u_init=None, maxiter=800, verbose=True):
    sweep = build_adjoint_sweep(prob)
    N = prob.N
    u_init = np.zeros(N) if u_init is None else np.asarray(u_init, float).reshape(N)

    n_eval = 0

    def fun(u):
        nonlocal n_eval
        n_eval += 1
        J, g, _, _ = sweep(jnp.asarray(u))
        return float(J), np.asarray(g, dtype=float)

    t0 = time.perf_counter()
    fun(u_init)  # trigger JIT compilation outside the timed region
    t_compile = time.perf_counter() - t0
    n_eval = 0

    t0 = time.perf_counter()
    res = minimize(
        fun,
        u_init,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-prob.u_max, prob.u_max)] * N,
        options={
            "maxiter": maxiter,
            "maxfun": 20 * maxiter,
            "ftol": 1e-16,
            "gtol": 1e-14,
        },
    )
    wall = time.perf_counter() - t0

    useq = np.clip(res.x, -prob.u_max, prob.u_max)
    J, grad, xtraj, ptraj = sweep(jnp.asarray(useq))
    proj = _projected_grad(grad, useq, prob.u_max)

    out = {
        "u": useq,
        "x": np.asarray(xtraj),
        "lam": np.asarray(ptraj),
        "cost": float(J),
        "grad_inf": float(np.max(np.abs(proj))),
        "n_iter": int(res.nit),
        "n_eval": n_eval,
        "wall": wall,
        "compile": t_compile,
    }
    if verbose:
        print("Stage 1 -- adjoint sweep + L-BFGS-B (first-order indirect)")
        print(f"  iterations         : {out['n_iter']}  ({out['n_eval']} sweeps)")
        print(f"  discrete cost J    : {out['cost']:.10f}")
        print(f"  ||proj dJ/du||_inf : {out['grad_inf']:.3e}")
        print(f"  |x(T) - xg|        : {np.linalg.norm(out['x'][-1] - prob.xgoal):.3e}")
        print(
            f"  wall               : {out['wall']:.3f} s (+{out['compile']:.2f} s JIT)"
        )
    return out


# Stage 2: indirect collocation -- Newton on the discretized PMP system
def build_tpbvp(prob: Problem):
    """Residual and sparse Jacobian of the discrete PMP boundary-value system."""
    h, R, N = prob.h, prob.R, prob.N
    Q, Qn = jnp.asarray(prob.Q), jnp.asarray(prob.Qn)
    xg, x0 = jnp.asarray(prob.xgoal), jnp.asarray(prob.x0)
    par = prob.params
    w = jnp.ones(N + 1).at[0].set(0.5).at[-1].set(0.5)

    step = lambda x, u: rk4_step(x, u, h, par)  # noqa: E731
    A_fn, B_fn = jax.jacfwd(step, 0), jax.jacfwd(step, 1)
    # Hessian of (x,u) -> F(x,u).p, i.e. the derivatives of A'p and B'p.
    W_fn = jax.hessian(lambda z, p: jnp.dot(step(z[:4], z[4]), p), 0)

    nX, nP = 4 * (N + 1), 4 * N
    n = nX + nP + N

    def unpack(z):
        z = np.asarray(z)
        return z[:nX].reshape(N + 1, 4), z[nX : nX + nP].reshape(N, 4), z[nX + nP :]

    @jax.jit
    def parts(z, act, bnd):
        X = z[:nX].reshape(N + 1, 4)
        Pm = z[nX : nX + nP].reshape(N, 4)  # Pm[j] = p_{j+1}
        U = z[nX + nP :]
        Xk = X[:-1]

        F = jax.vmap(step)(Xk, U)
        A = jax.vmap(A_fn)(Xk, U)
        B = jax.vmap(B_fn)(Xk, U)
        W = jax.vmap(W_fn)(jnp.concatenate([Xk, U[:, None]], 1), Pm)

        r_ic = X[0] - x0
        r_dyn = X[1:] - F
        dxk = X[1:N] - xg
        r_cs = Pm[: N - 1] - (
            h * w[1:N, None] * (dxk @ Q.T) + jnp.einsum("kij,ki->kj", A[1:N], Pm[1:N])
        )
        dxT = X[N] - xg
        r_tr = Pm[N - 1] - (Qn @ dxT + h * w[N] * (Q @ dxT))
        r_u = jnp.where(act, U - bnd, h * R * U + jnp.einsum("ki,ki->k", B, Pm))
        r = jnp.concatenate([r_ic, r_dyn.reshape(-1), r_cs.reshape(-1), r_tr, r_u])

        Wxx, Wxu, Wuu = W[:, :4, :4], W[:, :4, 4], W[:, 4, 4]
        free = 1.0 - act
        return (
            r,
            A,
            B,
            Wxx,
            Wxu,
            jnp.where(act, 1.0, h * R + Wuu),  # d r_u / d u_k
            Wxu * free[:, None],  # d r_u / d x_k  (= W[4,:4])
            B * free[:, None],
        )  # d r_u / d p_{k+1}

    # ---- constant sparsity pattern ---
    rows, cols = [], []
    e4 = np.arange(4)

    def blk(r0, c0, nr=4, nc=4):
        rr = (r0[:, None, None] + e4[None, :nr, None]).repeat(nc, 2)
        cc = (c0[:, None, None] + e4[None, None, :nc]).repeat(nr, 1)
        rows.append(rr.reshape(-1))
        cols.append(cc.reshape(-1))

    k, m = np.arange(N), np.arange(N - 1)
    xi, pi, ui = lambda i: 4 * i, lambda j: nX + 4 * j, lambda j: nX + nP + j
    o_dyn = 4
    o_cs = o_dyn + 4 * N
    o_tr = o_cs + 4 * (N - 1)
    o_u = o_tr + 4

    blk(np.array([0]), np.array([xi(0)]))
    blk(o_dyn + 4 * k, xi(k + 1))
    blk(o_dyn + 4 * k, xi(k))
    blk(o_dyn + 4 * k, ui(k), nc=1)
    blk(o_cs + 4 * m, pi(m))
    blk(o_cs + 4 * m, pi(m + 1))
    blk(o_cs + 4 * m, xi(m + 1))
    blk(o_cs + 4 * m, ui(m + 1), nc=1)
    blk(np.array([o_tr]), np.array([pi(N - 1)]))
    blk(np.array([o_tr]), np.array([xi(N)]))
    blk(o_u + k, ui(k), nr=1, nc=1)
    blk(o_u + k, xi(k), nr=1)
    blk(o_u + k, pi(k), nr=1)
    R_idx, C_idx = np.concatenate(rows), np.concatenate(cols)
    I4 = np.eye(4)
    hwQ = h * np.asarray(w)[1:N, None, None] * np.asarray(prob.Q)[None]
    QnhwQ = np.asarray(prob.Qn) + h * float(w[N]) * np.asarray(prob.Q)

    def jac(pp):
        _, A, B, Wxx, Wxu, ruu, rux, rup = (np.asarray(v) for v in pp)
        vals = np.concatenate(
            [
                I4.reshape(-1),
                np.tile(I4, (N, 1, 1)).reshape(-1),
                (-A).reshape(-1),
                (-B).reshape(-1),
                np.tile(I4, (N - 1, 1, 1)).reshape(-1),
                (-A[1:N].transpose(0, 2, 1)).reshape(-1),
                (-hwQ - Wxx[1:N]).reshape(-1),
                (-Wxu[1:N]).reshape(-1),
                I4.reshape(-1),
                (-QnhwQ).reshape(-1),
                ruu.reshape(-1),
                rux.reshape(-1),
                rup.reshape(-1),
            ]
        )
        return sp.csc_matrix((vals, (R_idx, C_idx)), shape=(n, n))

    return parts, jac, unpack, n


def _newton(parts, jac, z, act, bnd, max_iter=40, tol=1e-11):
    """Damped Newton with an Armijo line search on ||r||."""
    aj, bj = jnp.asarray(act), jnp.asarray(bnd)
    nr = np.inf
    for it in range(max_iter):
        pp = parts(jnp.asarray(z), aj, bj)
        r = np.asarray(pp[0])
        nr = float(np.linalg.norm(r))
        if not np.isfinite(nr):
            return z, nr, it
        if nr < tol:
            return z, nr, it
        try:
            dz = spla.spsolve(jac(pp), -r)
        except Exception:
            return z, nr, it
        if not np.all(np.isfinite(dz)):
            return z, nr, it
        for a in 0.5 ** np.arange(35):
            zn = z + a * dz
            rn = float(np.linalg.norm(np.asarray(parts(jnp.asarray(zn), aj, bj)[0])))
            if np.isfinite(rn) and rn < (1.0 - 1e-4 * a) * nr:
                z = zn
                break
        else:
            return z, nr, it
    return z, nr, max_iter


def _prolong(prob_c: Problem, prob_f: Problem, X, Pm, U):
    """Interpolate from a coarse control mesh onto a finer one."""
    tc, tf = prob_c.t_grid, prob_f.t_grid
    Xf = np.column_stack([np.interp(tf, tc, X[:, i]) for i in range(4)])
    # p is stored at nodes 1..N; pad with p_1 so the interpolation covers t = 0.
    Pc = np.vstack([Pm[0], Pm])
    Pf = np.column_stack([np.interp(tf, tc, Pc[:, i]) for i in range(4)])[1:]
    # u is a zero-order hold, so interpolate it at interval midpoints.
    Uf = np.interp(tf[:-1] + 0.5 * prob_f.h, tc[:-1] + 0.5 * prob_c.h, U)
    return Xf, Pf, Uf


def _mesh_ladder(N, coarsest=40):
    """Coarse-to-fine control meshes, doubling from coarsest up to N."""
    levels, n = [], coarsest
    while n < N:
        levels.append(n)
        n *= 2
    return levels + [N]


def solve_stage2(
    prob: Problem = PROBLEM,
    u_init=None,
    x_init=None,
    lam_init=None,
    coarsest=40,
    verbose=True,
):
    t0 = time.perf_counter()
    levels = _mesh_ladder(prob.N, coarsest)
    X, Pm, U, prev = x_init, None, u_init, None
    if lam_init is not None:
        Pm = np.asarray(lam_init)[1:]

    hist, t_compile = [], 0.0
    for lvl in levels:
        pr = prob.replace(h=prob.T / lvl)
        if U is None:
            s1 = solve_stage1(pr, maxiter=200, verbose=False)
            X, Pm, U = s1["x"], np.asarray(s1["lam"])[1:], s1["u"]
            t_compile += s1["compile"]
        else:
            X, Pm, U = _prolong(prev if prev is not None else prob, pr, X, Pm, U)

        parts, jac, unpack, nz = build_tpbvp(pr)
        U = np.clip(U, -pr.u_max, pr.u_max)
        act, bnd = np.zeros(lvl), np.zeros(lvl)

        tc = time.perf_counter()
        jax.block_until_ready(
            parts(jnp.zeros(nz), jnp.asarray(act), jnp.asarray(bnd))[0]
        )
        t_compile += time.perf_counter() - tc

        # Outer active-set loop
        for _ in range(6):
            z = np.concatenate([X.reshape(-1), Pm.reshape(-1), U])
            z, nr, nit = _newton(parts, jac, z, act, bnd)
            X, Pm, U = unpack(z)
            mult = np.asarray(
                parts(jnp.asarray(z), jnp.asarray(act), jnp.asarray(bnd))[0]
            )[-lvl:]
            new_act, new_bnd = act.copy(), bnd.copy()
            hi, lo = U > pr.u_max, U < -pr.u_max
            new_act[hi | lo] = 1.0
            new_bnd[hi], new_bnd[lo] = pr.u_max, -pr.u_max
            release = ((act == 1) & (bnd > 0) & (mult > 1e-9)) | (
                (act == 1) & (bnd < 0) & (mult < -1e-9)
            )
            new_act[release] = 0.0
            if np.array_equal(new_act, act):
                break
            act, bnd = new_act, new_bnd
            U = np.clip(U, -pr.u_max, pr.u_max)

        hist.append(
            {"N": lvl, "residual": nr, "n_iter": nit, "n_active": int(act.sum())}
        )
        if verbose:
            print(
                f"    N={lvl:5d}:  {nit:2d} Newton steps   ||r|| = {nr:.2e}   "
                f"|x(T)-xg| = {np.linalg.norm(X[-1] - pr.xgoal):.3e}"
            )
        prev = pr

    sweep = build_adjoint_sweep(prob)
    J, grad, xs, ps = sweep(jnp.asarray(U))
    wall = time.perf_counter() - t0 - t_compile

    out = {
        "u": np.asarray(U),
        "x": np.asarray(xs),
        "lam": np.asarray(ps),
        "cost": float(J),
        "residual": hist[-1]["residual"],
        "grad_inf": float(np.max(np.abs(_projected_grad(grad, U, prob.u_max)))),
        "n_iter": sum(l["n_iter"] for l in hist),
        "levels": hist,
        "n_active": hist[-1]["n_active"],
        "wall": wall,
        "compile": t_compile,
    }
    if verbose:
        print("Stage 2 -- indirect collocation + Newton (mesh continuation)")
        print(f"  mesh ladder        : {levels}")
        print(f"  TPBVP residual     : {out['residual']:.3e}")
        print(f"  ||proj dJ/du||_inf : {out['grad_inf']:.3e}")
        print(f"  discrete cost J    : {out['cost']:.10f}")
        print(f"  |x(T) - xg|        : {np.linalg.norm(out['x'][-1] - prob.xgoal):.3e}")
        print(
            f"  wall               : {out['wall']:.3f} s "
            f"(+{out['compile']:.2f} s XLA compile over {len(levels)} meshes)"
        )
    return out


# Optimality certificate
def hamiltonian(prob: Problem, x, u, lam):
    """``H(t) = l(x,u) + lam' f(x,u)`` at every node (u held over the interval)."""
    x, lam = np.asarray(x), np.asarray(lam)
    uu = np.append(np.asarray(u), np.asarray(u)[-1])
    dx = x - prob.xgoal
    run = 0.5 * np.einsum("ni,ij,nj->n", dx, prob.Q, dx) + 0.5 * prob.R * uu**2
    fx = np.array([acrobot.f_np(x[k], uu[k], prob.params) for k in range(len(x))])
    return run + np.einsum("ni,ni->n", lam, fx)


def certificate(prob: Problem, res):
    """Independent evidence that the result satisfies the PMP conditions."""
    s2 = res["stage2"]
    x, u, lam = s2["x"], s2["u"], s2["lam"]
    H = hamiltonian(prob, x, u, lam)
    drift = lambda a: float(np.max(np.abs(a - np.mean(a))))
    m = max(3, prob.N // 40)
    # Transversality is imposed on the discrete costate, which also carries the
    # final node's quadrature weight; the continuous condition returns as h -> 0.
    trans = float(np.max(np.abs(lam[-1] - prob.Qn @ (x[-1] - prob.xgoal))))

    print("\nOptimality certificate")
    print("-" * 70)
    print(f"  TPBVP residual  ||r||_2              : {s2['residual']:.3e}   (-> 0)")
    print(f"  minimum condition  ||proj H_u||_inf  : {s2['grad_inf']:.3e}   (-> 0)")
    print(f"  transversality  |lam(T) - Qn(x_T-xg)|: {trans:.3e}   (O(h))")
    print(f"  Hamiltonian constancy  max|H-mean H| : {drift(H):.3e}  full trajectory")
    print(
        f"                            interior   : {drift(H[m:-m]):.3e}  (O(h) -- it "
        f"halves when h does)"
    )
    print(f"  nodes on the torque bound            : {s2['n_active']} / {prob.N}")
    print("-" * 70)
    print("  The first two are exact statements about the discrete problem and are at")
    print("  round-off.  The last two are continuous-time conditions read off a")
    print("  discrete solution, so they are O(h) by construction, not solver error:")
    print("  the drift is dominated by the first few nodes, where |f| is largest.")
    return {
        "H": H,
        "H_drift": drift(H),
        "H_drift_interior": drift(H[m:-m]),
        "transversality": trans,
    }


# Driver
def solve(prob: Problem = PROBLEM, verbose=True, stage1_iters=300, stage2=True):
    if verbose:
        print("=" * 66)
        print("PONTRYAGIN MAXIMUM PRINCIPLE -- acrobot swing-up")
        print("=" * 66)

    s1 = solve_stage1(prob, maxiter=stage1_iters, verbose=verbose)
    if verbose:
        print()

    s2 = solve_stage2(prob, verbose=verbose) if stage2 else None

    best = s2 if s2 is not None else s1
    ref = acrobot.evaluate(
        prob.t_grid, np.append(best["u"], best["u"][-1]), prob, kind="zoh"
    )
    ref1 = acrobot.evaluate(
        prob.t_grid, np.append(s1["u"], s1["u"][-1]), prob, kind="zoh"
    )

    res = {
        "stage1": s1,
        "stage2": s2,
        "referee": ref,
        "referee_stage1": ref1,
        "t": prob.t_grid,
        "x": best["x"],
        "u": best["u"],
        "lam": best["lam"],
        "cost": best["cost"],
        "referee_cost": ref["cost"],
        "wall": s1["wall"] + (s2["wall"] if s2 else 0.0),
        "compile": s1["compile"] + (s2["compile"] if s2 else 0.0),
        "n_iter": (s2 or s1)["n_iter"],
    }
    if verbose:
        cert = certificate(prob, res) if s2 else {}
        res.update(cert)
        print("\nReferee evaluation (DOP853, rtol=1e-12, continuous cost)")
        print(
            f"  stage 1 (first-order)  J = {ref1['cost']:.8f}   "
            f"|x(T)-xg| = {ref1['terminal_error']:.3e}"
        )
        print(
            f"  stage 2 (Newton)       J = {ref['cost']:.8f}   "
            f"|x(T)-xg| = {ref['terminal_error']:.3e}"
        )
        ok = ref["terminal_error"] < 5e-2
        print(
            f"\n  SWING-UP {'ACHIEVED' if ok else 'FAILED'}  "
            f"(terminal error {ref['terminal_error']:.3e})"
        )
    elif s2:
        res.update({"H": hamiltonian(prob, best["x"], best["u"], best["lam"])})
    return res


# Self-checks
def _self_check(prob: Problem = PROBLEM):
    """Costate gradient vs autodiff, and the sparse Jacobian vs dense autodiff."""
    sweep = build_adjoint_sweep(prob)
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.normal(size=prob.N) * 2.0)

    J_sweep, g_sweep, _, _ = sweep(u)

    def cost_of_u(uu):
        xs = acrobot.rollout(jnp.asarray(prob.x0), uu, prob.h, prob.params)
        return acrobot.discrete_cost(xs, uu, prob)

    J_ad, g_ad = cost_of_u(u), jax.grad(cost_of_u)(u)
    scale = float(jnp.max(jnp.abs(g_ad)))
    dJ = float(abs(J_sweep - J_ad)) / max(float(abs(J_ad)), 1.0)
    dg = float(jnp.max(jnp.abs(g_sweep - g_ad))) / scale
    print("pmp.py self-check")
    print(f"  costate gradient vs autodiff : cost {dJ:.3e}, gradient {dg:.3e}")
    assert dJ < 1e-12 and dg < 1e-10, "adjoint sweep disagrees with autodiff"

    # Jacobian of the TPBVP residual, on a small mesh where dense autodiff is cheap.
    small = prob.replace(h=prob.T / 6)
    parts, jac, _, n = build_tpbvp(small)
    z = rng.normal(size=n) * 0.3
    act, bnd = jnp.zeros(small.N), jnp.zeros(small.N)
    Jd = np.asarray(jax.jacfwd(lambda zz: parts(zz, act, bnd)[0])(jnp.asarray(z)))
    Js = jac(parts(jnp.asarray(z), act, bnd)).toarray()
    rel = float(np.max(np.abs(Jd - Js)) / np.max(np.abs(Jd)))
    print(f"  sparse TPBVP Jacobian vs autodiff : relative {rel:.3e}")
    assert rel < 1e-10, "sparse Jacobian assembly is wrong"
    print("  OK\n")


def _measurements(prob: Problem = PROBLEM):
    R = prob.R
    res = solve(prob, verbose=False)
    x, lam = res["x"], res["lam"]

    def flow(y):
        xx, ll = y[:4], y[4:]
        uu = -(Minv_B(xx, prob.params) @ ll[2:]) / R
        _, vjp = jax.vjp(lambda zz: f(zz, uu, prob.params), xx)
        return jnp.concatenate(
            [
                f(xx, uu, prob.params),
                -(jnp.asarray(prob.Q) @ (xx - jnp.asarray(prob.xgoal))) - vjp(ll)[0],
            ]
        )

    jac_flow = jax.jit(jax.jacfwd(flow))
    mx = max(
        float(
            np.max(
                np.linalg.eigvals(
                    np.asarray(
                        jac_flow(
                            jnp.concatenate([jnp.asarray(x[k]), jnp.asarray(lam[k])])
                        )
                    )
                ).real
            )
        )
        for k in range(0, prob.N + 1, max(1, prob.N // 40))
    )
    g = np.array([acrobot.Minv_B_np(x[k], prob.params) for k in range(prob.N)])
    dot = np.einsum("ki,ki->k", g, lam[1:, 2:])
    cancel = np.abs(g * lam[1:, 2:]).sum(1) / np.maximum(np.abs(dot), 1e-300)

    print("Why shooting fails on this problem")
    print("-" * 66)
    print(f"  max Re(eig) of the Hamiltonian flow  : {mx:.1f} 1/s")
    print(f"  single-shooting amplification over T : exp({mx * prob.T:.0f})")
    for M in (10, 100):
        print(
            f"    with {M:3d} multiple-shooting segments : "
            f"{np.exp(mx * prob.T / M):.2e} per segment"
        )
    print(
        f"  cancellation in u* = -(M^-1 B . lam_v)/R : median "
        f"{np.median(cancel):.0f}x, max {np.max(cancel):.0f}x"
    )
    print(
        f"  |du/dlam_v| = |M^-1 B|/R             : up to "
        f"{np.max(np.linalg.norm(g, axis=1)) / R:.1e}"
    )
    print("-" * 66)
    print()
    _reduced_conditioning(prob, res)


def _reduced_conditioning(prob: Problem, res: dict):
    """Why first-order methods on the reduced map ``u -> J`` stall."""
    sweep = build_adjoint_sweep(prob)
    hess = jax.jit(jax.jacfwd(lambda uu: sweep(uu)[1]))

    def at(u):
        u = jnp.asarray(u)
        H = np.asarray(hess(u))
        H = 0.5 * (H + H.T)
        return np.asarray(sweep(u)[1]), H, np.linalg.eigvalsh(H)

    stall = solve_stage1(prob, maxiter=300, verbose=False)
    gs, Hs, ws = at(stall["u"])
    alpha = float(gs @ gs / (gs @ Hs @ gs))  # exact Cauchy (steepest-descent) step
    _, _, wo = at(res["u"])

    print("Why first-order methods on the reduced map u -> J stall")
    print("-" * 66)
    print(
        f"  at the 300-iteration L-BFGS-B iterate (J = {stall['cost']:.6f}, "
        f"{stall['cost'] - res['cost']:.3f} above the optimum):"
    )
    print(f"    reduced Hessian |d2J/du2|_2        : {ws[-1]:.4e}")
    print(f"    eigenvalue range                   : [{ws[0]:.4g}, {ws[-1]:.4g}]")
    print(f"    ||H_u||_inf                        : {np.max(np.abs(gs)):.4g}")
    print(
        f"    exact steepest-descent step        : moves u by "
        f"{np.max(np.abs(alpha * gs)):.2e}, lowers J by {0.5 * alpha * (gs @ gs):.2e}"
    )
    print(f"  at the converged solution (J = {res['cost']:.6f}):")
    print(f"    eigenvalue range                   : [{wo[0]:.4g}, {wo[-1]:.4g}]")
    print(f"    condition number                   : {wo[-1] / wo[0]:.3e}")
    print(f"    positive definite (2nd-order suff.): {bool(wo[0] > 0)}")
    print("-" * 66)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage1-only", action="store_true", help="skip the TPBVP solve")
    ap.add_argument("--stage1-iters", type=int, default=300)
    ap.add_argument(
        "--measurements",
        action="store_true",
        help="reproduce the conditioning numbers from the docstring",
    )
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-animate", action="store_true")
    args = ap.parse_args()

    _self_check()
    res = solve(PROBLEM, stage1_iters=args.stage1_iters, stage2=not args.stage1_only)
    if args.measurements:
        print()
        _measurements()

    if not args.no_plot:
        import matplotlib.pyplot as plt

        acrobot.plot_solution(
            res["t"],
            res["x"],
            res["t"][:-1],
            res["u"],
            PROBLEM,
            title="PMP (indirect) acrobot swing-up",
            savepath="results/pmp.png",
        )
        if res["stage2"] is not None:
            f2, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            for i, lbl in enumerate(
                [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"]
            ):
                ax[0].plot(res["t"], res["lam"][:, i], label=lbl)
            ax[0].set_ylabel("costate")
            ax[0].legend(ncol=4)
            ax[0].grid(True, alpha=0.3)
            ax[0].set_title("costate from the indirect solve")
            ax[1].plot(res["t"], res["H"], color="tab:purple")
            ax[1].set_ylabel("$H(t)$")
            ax[1].set_xlabel("time [s]")
            ax[1].set_title("the Hamiltonian must be constant along an extremal")
            ax[1].grid(True, alpha=0.3)
            f2.tight_layout()
            f2.savefig("results/pmp_costate.png", dpi=130)

        ani = None
        if not args.no_animate:
            ani = acrobot.animate(res["t"], res["x"], PROBLEM, title="PMP swing-up")
        plt.show()
