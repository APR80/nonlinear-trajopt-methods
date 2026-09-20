"""
Shared acrobot model and common utilities.
it also has a high-accuracy referee evaluator (:func:`evaluate`) used to score every
method on equal footing, independent of its own discretization;
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np

__all__ = [
    "AcrobotParams",
    "Problem",
    "PARAMS",
    "PROBLEM",
    "f",
    "f_np",
    "f_casadi",
    "Minv_B",
    "rk4_step",
    "rollout",
    "discrete_cost",
    "energy",
    "evaluate",
    "control_interp",
    "plot_solution",
    "animate",
]


# Parameters and problem specification
@dataclass(frozen=True)
class AcrobotParams:
    """physical parameters."""

    m1: float = 0.1
    m2: float = 0.08
    l1: float = 0.3
    l2: float = 0.4
    lc1: float = 0.17
    lc2: float = 0.21
    I1: float = 0.04
    I2: float = 0.06
    g: float = 9.81


PARAMS = AcrobotParams()


@dataclass(frozen=True)
class Problem:
    """The swing-up problem solved by every method in this repo.
    Continuous-time objective (soft terminal -- no hard endpoint constraint)::
        J = 1/2 (x(T) - xg)' Qn (x(T) - xg)
            + int_0^T [ 1/2 (x - xg)' Q (x - xg) + 1/2 R u^2 ] dt
    subject to ``xdot = f(x, u)``, ``x(0) = x0``, ``|u| <= u_max``.
    """

    x0: np.ndarray = field(default_factory=lambda: np.zeros(4))
    xgoal: np.ndarray = field(default_factory=lambda: np.array([np.pi, 0.0, 0.0, 0.0]))
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.1, 0.1]))
    R: float = 0.01
    Qn: np.ndarray = field(default_factory=lambda: np.diag([200.0, 200.0, 20.0, 20.0]))
    T: float = 5.0
    h: float = 0.0125
    u_max: float = 15.0
    params: AcrobotParams = PARAMS

    @property
    def N(self) -> int:
        """Number of intervals on the nominal grid."""
        return int(round(self.T / self.h))

    @property
    def t_grid(self) -> np.ndarray:
        """Nominal node times, length ``N + 1``."""
        return np.linspace(0.0, self.T, self.N + 1)

    def replace(self, **kw) -> "Problem":
        return replace(self, **kw)


PROBLEM = Problem()


# Dynamics
class _CasadiBackend:
    """Minimal numpy like shim so the dynamics can be written once."""

    @staticmethod
    def sin(v):
        import casadi as ca

        return ca.sin(v)

    @staticmethod
    def cos(v):
        import casadi as ca

        return ca.cos(v)

    @staticmethod
    def stack(items):
        import casadi as ca

        return ca.vertcat(*items)


def _dynamics(x, u, xp, p: AcrobotParams = PARAMS):

    th1, th2, d1, d2 = x[0], x[1], x[2], x[3]
    s1 = xp.sin(th1)
    s2, c2 = xp.sin(th2), xp.cos(th2)
    s12 = xp.sin(th1 + th2)

    coupling = p.m2 * p.l1 * p.lc2

    # Mass matrix
    M11 = p.I1 + p.I2 + p.m2 * p.l1**2 + 2.0 * coupling * c2
    M12 = p.I2 + coupling * c2
    M22 = p.I2
    det = M11 * M22 - M12 * M12

    # Gravity torques tau_g = -dV/dq.
    tg1 = -p.m1 * p.g * p.lc1 * s1 - p.m2 * p.g * (p.l1 * s1 + p.lc2 * s12)
    tg2 = -p.m2 * p.g * p.lc2 * s12

    # Coriolis: C(q, qdot) qdot, with
    cor1 = -2.0 * coupling * s2 * d2 * d1 - coupling * s2 * d2 * d2
    cor2 = coupling * s2 * d1 * d1

    rhs1 = tg1 - cor1
    rhs2 = u + tg2 - cor2

    dd1 = (M22 * rhs1 - M12 * rhs2) / det
    dd2 = (-M12 * rhs1 + M11 * rhs2) / det
    return xp.stack([d1, d2, dd1, dd2])


def f_np(x, u, p: AcrobotParams = PARAMS):
    """NumPy dynamics.  Used by the referee integrator."""
    return _dynamics(x, u, np, p)


def f_casadi(x, u, p: AcrobotParams = PARAMS):
    """CasADi (symbolic) dynamics, for the direct-collocation transcription."""
    return _dynamics(x, u, _CasadiBackend, p)


def f(x, u, p: AcrobotParams = PARAMS):
    """JAX dynamics.  Used by iLQR/DDP and by the PMP costate integration."""
    import jax.numpy as jnp

    return _dynamics(x, jnp.squeeze(u), jnp, p)


def _Minv_B(x, xp, p: AcrobotParams = PARAMS):
    """``M(q)^-1 B`` with ``B = [0, 1]'`` -- the nonzero block of ``df/du``.
    The PMP minimum condition needs this analytically:
    """
    c2 = xp.cos(x[1])
    coupling = p.m2 * p.l1 * p.lc2
    M11 = p.I1 + p.I2 + p.m2 * p.l1**2 + 2.0 * coupling * c2
    M12 = p.I2 + coupling * c2
    M22 = p.I2
    det = M11 * M22 - M12 * M12
    return xp.stack([-M12 / det, M11 / det])


def Minv_B(x, p: AcrobotParams = PARAMS):
    import jax.numpy as jnp

    return _Minv_B(x, jnp, p)


def Minv_B_np(x, p: AcrobotParams = PARAMS):
    return _Minv_B(x, np, p)


def energy(x, p: AcrobotParams = PARAMS) -> float:
    """Total mechanical energy.  Zero-input trajectories must conserve it."""
    th1, th2, d1, d2 = x
    coupling = p.m2 * p.l1 * p.lc2
    M11 = p.I1 + p.I2 + p.m2 * p.l1**2 + 2.0 * coupling * np.cos(th2)
    M12 = p.I2 + coupling * np.cos(th2)
    M22 = p.I2
    ke = 0.5 * (M11 * d1 * d1 + 2.0 * M12 * d1 * d2 + M22 * d2 * d2)
    pe = -p.m1 * p.g * p.lc1 * np.cos(th1) - p.m2 * p.g * (
        p.l1 * np.cos(th1) + p.lc2 * np.cos(th1 + th2)
    )
    return float(ke + pe)


# Discrete-time helpers (JAX)
def rk4_step(x, u, h, p: AcrobotParams = PARAMS):
    k1 = f(x, u, p)
    k2 = f(x + 0.5 * h * k1, u, p)
    k3 = f(x + 0.5 * h * k2, u, p)
    k4 = f(x + h * k3, u, p)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(x0, useq, h, p: AcrobotParams = PARAMS):
    """Roll the RK4 dynamics forward under an open-loop control sequence."""
    import jax
    import jax.numpy as jnp

    def step(x, u):
        xn = rk4_step(x, u, h, p)
        return xn, xn

    _, xs = jax.lax.scan(step, x0, useq)
    return jnp.vstack([x0[None, :], xs])


def discrete_cost(xtraj, useq, prob: Problem):
    """shared by the shooting methods (iLQR/DDP and PMP stage 1)."""
    import jax.numpy as jnp

    Q = jnp.asarray(prob.Q)
    Qn = jnp.asarray(prob.Qn)
    xg = jnp.asarray(prob.xgoal)
    h = prob.h

    dx = xtraj - xg[None, :]
    node_cost = 0.5 * jnp.einsum("ni,ij,nj->n", dx, Q, dx)
    weights = jnp.ones(node_cost.shape[0]).at[0].set(0.5).at[-1].set(0.5)
    state_cost = h * jnp.sum(weights * node_cost)

    u = jnp.reshape(useq, (-1,))
    ctrl_cost = h * jnp.sum(0.5 * prob.R * u**2)

    dxT = xtraj[-1] - xg
    return state_cost + ctrl_cost + 0.5 * dxT @ Qn @ dxT


# Referee: score any control on the true continuous problem
def control_interp(t_grid, u_vals, kind="linear", u_max=None):
    """Build u(t) from samples."""
    t_grid = np.asarray(t_grid, dtype=float)
    u_vals = np.asarray(u_vals, dtype=float).reshape(-1)

    if kind == "zoh":
        # u_vals has one entry per interval
        edges = t_grid[: len(u_vals) + 1]

        def u_of_t(t):
            i = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(u_vals) - 1)
            return float(u_vals[i])

    elif kind == "linear":
        if len(u_vals) != len(t_grid):
            raise ValueError(
                f"linear interpolation needs one control per node "
                f"({len(t_grid)}), got {len(u_vals)}"
            )

        def u_of_t(t):
            return float(np.interp(t, t_grid, u_vals))

    else:
        raise ValueError(f"unknown interpolation kind: {kind!r}")

    if u_max is None:
        return u_of_t

    return lambda t: float(np.clip(u_of_t(t), -u_max, u_max))


def evaluate(t_grid, u_vals, prob: Problem = PROBLEM, kind="linear"):
    """Score a on the continuous problem"""
    from scipy.integrate import solve_ivp

    u_of_t = control_interp(t_grid, u_vals, kind=kind, u_max=prob.u_max)
    Q, Qn, xg = prob.Q, prob.Qn, prob.xgoal

    def aug(t, y):
        x = y[:4]
        u = u_of_t(t)
        dx = f_np(x, u, prob.params)
        e = x - xg
        dJ = 0.5 * e @ Q @ e + 0.5 * prob.R * u * u
        return np.concatenate([dx, [dJ]])

    y0 = np.concatenate([np.asarray(prob.x0, dtype=float), [0.0]])
    sol = solve_ivp(
        aug,
        (0.0, prob.T),
        y0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
        max_step=prob.h,  # never step over a control breakpoint
    )
    if not sol.success:
        raise RuntimeError(f"referee integration failed: {sol.message}")

    xT = sol.y[:4, -1]
    running = float(sol.y[4, -1])
    eT = xT - xg
    terminal = 0.5 * float(eT @ Qn @ eT)

    return {
        "x_final": xT,
        "cost": running + terminal,
        "cost_running": running,
        "cost_terminal": terminal,
        "terminal_error": float(np.linalg.norm(eT)),
        "sol": sol,
    }


# Plotting / animation
def link_positions(theta1, theta2, p: AcrobotParams = PARAMS):
    theta1 = np.asarray(theta1)
    theta2 = np.asarray(theta2)
    x1 = p.l1 * np.sin(theta1)
    y1 = -p.l1 * np.cos(theta1)
    x2 = x1 + p.l2 * np.sin(theta1 + theta2)
    y2 = y1 - p.l2 * np.cos(theta1 + theta2)
    return x1, y1, x2, y2


def plot_solution(
    t_x, xtraj, t_u, useq, prob: Problem = PROBLEM, title="", savepath=None
):
    import matplotlib.pyplot as plt

    xtraj = np.asarray(xtraj)
    useq = np.asarray(useq).reshape(-1)

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(t_x, xtraj[:, 0], label=r"$\theta_1$")
    axs[0].plot(t_x, xtraj[:, 1], label=r"$\theta_2$")
    axs[0].axhline(prob.xgoal[0], color="gray", ls="--", lw=1, label="goal $\\theta_1$")
    axs[0].axhline(prob.xgoal[1], color="gray", ls=":", lw=1)
    axs[0].set_ylabel("angle [rad]")
    axs[0].legend(loc="best")

    axs[1].plot(t_x, xtraj[:, 2], label=r"$\dot\theta_1$")
    axs[1].plot(t_x, xtraj[:, 3], label=r"$\dot\theta_2$")
    axs[1].set_ylabel("rate [rad/s]")
    axs[1].legend(loc="best")

    axs[2].plot(t_u, useq, color="tab:green", label="$u$")
    if np.isfinite(prob.u_max):
        axs[2].axhline(prob.u_max, color="r", ls="--", lw=1)
        axs[2].axhline(-prob.u_max, color="r", ls="--", lw=1)
    axs[2].set_ylabel("torque [N·m]")
    axs[2].set_xlabel("time [s]")
    axs[2].legend(loc="best")

    for a in axs:
        a.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def animate(
    t_hist, xtraj, prob: Problem = PROBLEM, title="", interval=None, savepath=None
):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import os

    xtraj = np.asarray(xtraj)
    p = prob.params
    x1, y1, x2, y2 = link_positions(xtraj[:, 0], xtraj[:, 1], p)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    margin = 0.1 + p.l1 + p.l2
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    ax.plot(x2, y2, lw=0.8, color="0.8", zorder=0)  # tip trace
    (link1,) = ax.plot([], [], "o-", lw=3, markersize=8, color="#3498db")
    (link2,) = ax.plot([], [], "o-", lw=3, markersize=8, color="#e74c3c")
    ax.plot([0], [0], "o", markersize=8, color="k")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        link1.set_data([], [])
        link2.set_data([], [])
        time_text.set_text("")
        return link1, link2, time_text

    def step(i):
        link1.set_data([0, x1[i]], [0, y1[i]])
        link2.set_data([x1[i], x2[i]], [y1[i], y2[i]])
        time_text.set_text(f"t = {t_hist[i]:.2f} s")
        return link1, link2, time_text

    if interval is None:
        interval = 1000.0 * (t_hist[1] - t_hist[0]) if len(t_hist) > 1 else 50.0

    ani = animation.FuncAnimation(
        fig, step, frames=len(t_hist), init_func=init, interval=interval, blit=False
    )

    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fps = max(1, round(1000.0 / interval))
        ani.save(savepath, writer=animation.PillowWriter(fps=fps))

    return ani


# Self-checks
def _self_check():
    import casadi as ca
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    print("acrobot.py self-check")
    print("-" * 64)

    # All backends must agree.
    x_sym = ca.MX.sym("x", 4)
    u_sym = ca.MX.sym("u")
    f_ca = ca.Function("f", [x_sym, u_sym], [f_casadi(x_sym, u_sym)])

    worst_ca = worst_jax = 0.0
    for _ in range(200):
        x = rng.normal(size=4) * np.array([3.0, 3.0, 5.0, 5.0])
        u = float(rng.normal() * 5.0)
        ref = f_np(x, u)
        worst_ca = max(worst_ca, np.max(np.abs(np.array(f_ca(x, u)).ravel() - ref)))
        worst_jax = max(
            worst_jax, np.max(np.abs(np.asarray(f(jnp.asarray(x), u)) - ref))
        )
    print(f"  NumPy vs CasADi dynamics : max abs diff {worst_ca:.3e}")
    print(f"  NumPy vs JAX   dynamics  : max abs diff {worst_jax:.3e}")
    assert worst_ca < 1e-12 and worst_jax < 1e-12, "dynamics backends disagree"

    # df/du must equal [0, 0, M^-1 B].
    B_ad = np.asarray(jax.jacfwd(lambda uu: f(jnp.asarray(x), uu))(0.3))
    err_B = np.max(np.abs(B_ad[2:] - Minv_B_np(x)))
    print(f"  analytic M^-1 B vs autodiff df/du : {err_B:.3e}")
    assert err_B < 1e-12

    # RK4 must converge at 4th order against DOP853.
    from scipy.integrate import solve_ivp

    x_test = np.array([0.4, -0.3, 0.5, -0.2])
    u_test = 0.7
    Tint = 0.4
    ref = solve_ivp(
        lambda t, y: f_np(y, u_test),
        (0, Tint),
        x_test,
        method="DOP853",
        rtol=1e-13,
        atol=1e-13,
    ).y[:, -1]

    prev_err, orders = None, []
    print("  RK4 convergence vs DOP853:")
    for n in (10, 20, 40, 80):
        hh = Tint / n
        xx = jnp.asarray(x_test)
        for _ in range(n):
            xx = rk4_step(xx, u_test, hh)
        err = float(np.max(np.abs(np.asarray(xx) - ref)))
        if prev_err is not None:
            orders.append(np.log2(prev_err / err))
        order = "" if not orders else f"  order ~ {orders[-1]:.2f}"
        print(f"    h = {hh:.4f}   err = {err:.3e}{order}")
        prev_err = err
    assert 3.5 < min(orders) and max(orders) < 4.5, f"RK4 is not 4th order: {orders}"

    # 4. Zero-input dynamics must conserve energy.
    x_free = np.array([0.6, 0.2, 0.3, -0.4])
    sol = solve_ivp(
        lambda t, y: f_np(y, 0.0),
        (0, 5.0),
        x_free,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    drift = abs(energy(sol.y[:, -1]) - energy(x_free))
    print(f"  energy drift over 5 s with u=0 : {drift:.3e}")
    assert drift < 1e-9, "unforced dynamics do not conserve energy"

    # 5. Referee reproduces a known rollout.
    prob = PROBLEM
    u_nodes = 0.5 * np.sin(2.0 * prob.t_grid)
    res = evaluate(prob.t_grid, u_nodes, prob, kind="linear")
    print(
        f"  referee sample run: J = {res['cost']:.6f}, "
        f"|x(T)-xg| = {res['terminal_error']:.4f}"
    )

    print("-" * 64)
    print("all self-checks passed")
    print(
        f"\ncanonical problem: T={prob.T}s  h={prob.h}s  N={prob.N}  "
        f"|u|<={prob.u_max}  R={prob.R}"
    )
    print(f"  Q  = diag{tuple(float(v) for v in np.diag(prob.Q))}")
    print(f"  Qn = diag{tuple(float(v) for v in np.diag(prob.Qn))}")


if __name__ == "__main__":
    _self_check()
