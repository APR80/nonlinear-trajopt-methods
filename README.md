# Nonlinear trajectory optimization algorithms
This repo contains implementations of trajectory optimization methods applied to the acrobot swing-up, with a shared dynamics model and a common referee for comparing them on equal terms.
| method | file | family | backend |
|---|---|---|---|
| iLQR / DDP | [`ilqr_ddp.py`](ilqr_ddp.py) | shooting / dynamic programming | JAX |
| Direct collocation | [`dircol.py`](dircol.py) | direct transcription (Hermite–Simpson) | CasADi + IPOPT |
| pontryagin maximum principle (indirect) | [`pmp.py`](pmp.py) | necessary conditions / indirect | JAX |
| Compare | [`compare.py`](compare.py) | runs all of them side by side | — |

All four land on the same trajectory: 

<img src="results/compare.png" alt="all four methods" width="600">
<img width="400" height="400" alt="swingup" src="https://github.com/user-attachments/assets/00b90cda-bcf4-4bb0-9701-17bf0d2f5a3c" />

## note

Every method optimizes the *identical* objective — same `Q`, `R`, `Qn`, horizon, and a
**soft-only** terminal cost (no hard terminal constraint) — which is what makes their
costs directly comparable. Each solver's own reported cost is on its own discretization,
and that is why the referee(referee (an independent
DOP853 re-integration at `rtol=1e-12`) is for.

## PMP: why is it so bad?

PMP is the slowest and by far the hardest for me to get converging, which was sad because that is 
the method I learned in my optimal control course(kirk's book). here are the reasons I descovered:

- **The reduced map `u → J` is brutally ill-conditioned** (condition number ~3.3e11), so
  plain gradient descent on the adjoint-sweep gradient stalls — even a good first-order
  method (L-BFGS-B) is still 2.67 above the optimum after 300 iterations.
- **Shooting on the Hamiltonian system is numerically impossible.** The costate dynamics
  double the spectrum of the state dynamics; a single-shooting error amplifies by `e^756`
  over the horizon, and even 100 multiple-shooting segments still amplify ~2000× each.
- **Recovering `u` from the costate `λ` is near-total cancellation** (up to 3×10⁵×), so an
  extremely accurate costate is needed just to get a usable control.

The fix here is **indirect collocation**: treat state, costate, and control as
independent unknowns and solve the discretized necessary conditions directly with Newton
(well-conditioned, unlike the reduced problem), climbing a mesh-continuation ladder
from a coarse solve seeded by a short first-order (L-BFGS-B) run.

## Running it

```bash
pip install numpy scipy matplotlib jax casadi

python acrobot.py                     # self-checks on the shared core
python ilqr_ddp.py                    # --mode ilqr|ddp|both
python dircol.py                      # --form separated|compressed|both
python pmp.py                         # --stage1-only, --measurements
python compare.py --refine            # full comparison + mesh-refinement study


