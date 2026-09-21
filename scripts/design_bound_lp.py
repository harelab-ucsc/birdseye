#!/usr/bin/env python3
"""
Trace-based design bound: certified constraint variants and a design heuristic.

The deployed design bound in ``pose_uncertainty_prop.design_bound`` uses the
submultiplicative step

    lambda_u,max  <=  sup_X ||J_xi(X_w)||_2^2 * lambda_max(Sigma_0),

which is tight only when the dominant right singular vector of ``J_xi`` aligns
with the eigenvector of the noisiest DOF.  This module replaces that step with
the trace family

    tr(J_xi Sigma_0 J_xi^T) = sum_i sigma_i^2 ||j_i(X_w)||_2^2 = s . g(X_w),

so every bound variant is a *linear* functional of the design vector
``s = [sigma_1^2 ... sigma_6^2]``, with ``g_i(X_w) = ||j_i(X_w)||_2^2``.  Each
variant therefore induces a polyhedron in design space and the admissible
noise budget is the solution of a small linear (or convex) program.

The constraint registry admits **certified variants only**: every functional
``f`` here satisfies ``f(s) <= lambda_u,target  =>  lambda_u,max <=
lambda_u,target``, i.e. it is a true upper bound on the worst-case pixel
variance.  Descriptive statistics of the trace field (mean, quantiles, CVaR)
are *not* bounds and are not registered as constraints; they appear on the
distribution page as descriptive markers only.

Every constraint functional is positively homogeneous of degree 1 in ``s`` and
convex, so each subgradient is a *valid global* cut (``f(s) = a.s`` at the
evaluation point) and a handful of cutting planes replaces the 2.3M-row
constraint matrix exactly.

The payload is the design heuristic: admissible per-DOF 1-sigma pose accuracy
as a function of altitude and desired pixel fidelity.

Study execution and reporting are separate: ``run_study()`` computes the sweep
and returns a ``Study`` (saved as JSON), while ``build_report()`` renders the
PDF from a ``Study`` held in memory or reloaded from disk with ``--render
STUDY.json``; the CLI renders a PDF only when ``--pdf`` is passed.

``pose_uncertainty_prop.py`` is imported, never modified.
"""

import argparse
import json
import os
import time
import textwrap
from dataclasses import dataclass
import numpy as np
import cvxpy as cp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pose_uncertainty_prop import Camera, CovarianceModel, ProjectionModel, SE3

DOF_LABELS = ["x", "y", "z", "roll", "pitch", "yaw"]

DEFAULT_CAMERA = dict(w=1920, h=1200,
                      fx=4264.494512341911, fy=4262.892739736864,
                      cx=958.4594068961055, cy=592.50331737885)
DEFAULT_ALTITUDES = [5.0, 10.0, 20.0, 50.0]
DEFAULT_STD_DEVS = [0.01, 0.01, 0.06, 0.04, 0.04, 0.13]
DEFAULT_LAM_TARGET = 100.0        # = (30 px 3-sigma radius / 3)^2 = (10 px 1-sigma)^2
DEFAULT_RADII = [5.0, 10.0, 20.0, 30.0, 50.0]


# ==========================================================
# Definitions (contract shared with scripts/design_bound_lp.md)
# ==========================================================

DEFINITIONS = [
    ("X_w", "A point on the ground plane Z = z in world coordinates; the set X is the "
            "back-projection of the image rectangle at the nominal pose."),
    ("T = (R, t)", "Camera pose in world frame. Here the nominal nadir pose "
                   "SE3.nominal_pose(h): R = I, t = [0, 0, h]."),
    ("xi", "A 6-vector pose perturbation [dx, dy, dz, d_roll, d_pitch, d_yaw] applied to T "
           "in the convention of pose_uncertainty_prop.SE3.perturb; translation in metres, "
           "rotation in radians."),
    ("d", "Depth d = h - z, the camera-to-ground distance along the optical axis (constant "
          "over the FOV at nadir)."),
    ("J_xi(X_w)", "The 2x6 Jacobian d[u, v] / d xi of the pixel projection of X_w with respect "
                  "to the pose perturbation, evaluated at the nominal pose. Computed as "
                  "J_pixel @ [I | -skew(xc)] with xc = R X_w + t. Units: px/m for columns 1-3, "
                  "px/rad for columns 4-6."),
    ("j_i(X_w)", "The i-th column of J_xi(X_w): the pixel displacement produced by a unit "
                 "perturbation of DOF i."),
    ("g_i(X_w)", "Gain field g_i = ||j_i(X_w)||_2^2, the squared pixel sensitivity to DOF i. "
                 "Units px^2/m^2 (i <= 3) or px^2/rad^2 (i >= 4)."),
    ("Sigma_0", "Pose covariance, assumed diagonal: Sigma_0 = diag(s)."),
    ("s", "Design vector s = [sigma_1^2 ... sigma_6^2], the per-DOF pose variances. Units m^2 "
          "(i <= 3), rad^2 (i >= 4). The unknown of every optimization here."),
    ("s_base", "The deployed pose-uncertainty budget, diag(CovarianceModel.realistic(std_devs)) "
               "= [1e-4, 1e-4, 3.6e-3, 4.873879e-7, 4.873879e-7, 5.148034e-6], i.e. 1-sigma "
               "[10, 10, 60] mm and [0.04, 0.04, 0.13] deg (datasheet). Note realistic converts "
               "DOF 4-6 from degrees to radians."),
    ("rho", "Normalized design rho = s / s_base, dimensionless multiples of the deployed "
            "variance. All solves are performed in these coordinates."),
    ("Sigma_u(X_w)", "Propagated pixel covariance Sigma_u = J_xi diag(s) J_xi^T, a 2x2 PSD "
                     "matrix, units px^2."),
    ("lambda_max(Sigma_u)", "Larger eigenvalue of Sigma_u: the variance along the worst pixel "
                            "direction."),
    ("tr(Sigma_u)", "Trace = sum_i s_i g_i(X_w) = s . g(X_w): total pixel variance, linear in s."),
    ("lambda_u,max", "The quantity being bounded: sup_{X_w in X} lambda_max(Sigma_u(X_w)), the "
                     "worst-case pixel variance anywhere in the field of view."),
    ("lambda_u,target", "The pixel-fidelity tolerance, in px^2. Related to a 3-sigma pixel radius "
                        "r by lambda_u,target = (r/3)^2. Default 100 px^2 (r = 30 px)."),
    ("r", "Desired pixel fidelity: 3-sigma positional radius in pixels of the projected point, "
          "r = 3 sqrt(lambda_u,target)."),
    ("certified", "A constraint functional f is certified if f(s) <= lambda_u,target implies "
                  "lambda_u,max <= lambda_u,target for all s >= 0; i.e. f is a true upper bound "
                  "on lambda_u,max. All three variants retained here are certified."),
    ("sup", "Supremum over the field of view, evaluated as a maximum over every sampled pixel "
            "(edges inclusive, so the frame corner where the suprema are attained is sampled "
            "exactly)."),
    ("kappa*", "Admissible uniform scaling kappa* = lambda_u,target / f(s_base): the largest "
               "factor by which the deployed budget can be scaled and still satisfy f. Exact by "
               "positive homogeneity, no solver needed. kappa* > 1 means the deployed budget has "
               "slack."),
    ("homogeneous of degree 1",
     "f(kappa s) = kappa f(s) for all kappa > 0. Every functional here has this property, which "
     "is what makes kappa* closed-form and every cut globally valid."),
    ("gap ratio", "sup_X tr(Sigma_u) / sup_X lambda_max(Sigma_u) at a fixed s. Bounded by 2 for "
                  "2x2 PSD matrices; measures how much the trace bound over-counts."),
    ("conservatism", "kappa*(exact_lmax) / kappa*(variant): how much design budget a variant "
                     "forfeits relative to the exact worst-case constraint. 1.0 = no loss."),
    ("cut (cutting plane)",
     "A halfspace a . s <= lambda_u,target that contains the true feasible set "
     "{f(s) <= lambda_u,target}. Here a is a subgradient of f at the query point and, by "
     "homogeneity, satisfies a . s = f(s) exactly."),
    ("subgradient", "A vector a with f(s') >= f(s) + a . (s' - s) for all s'. For a pointwise "
                    "maximum f(s) = max_n a_n . s, the row a_{n*} attaining the maximum is a "
                    "subgradient (Danskin)."),
    ("polyhedron", "The intersection of finitely many halfspaces; the accumulated cuts form an "
                   "outer approximation of the feasible set."),
    ("vertex solution", "The optimum of a linear objective over a polyhedron, attained at a "
                        "corner; it puts the whole budget into as few DOFs as the active "
                        "constraints permit."),
    ("balanced (geometric-mean) solution",
     "The maximizer of sum_i log rho_i, i.e. of the geometric mean of rho. Scale-invariant, "
     "strictly interior, unique; spends budget on every DOF instead of a corner."),
    ("violation fraction", "Fraction of sampled FOV pixels with lambda_max(Sigma_u) > "
                           "lambda_u,target(1 + 1e-9). Zero for every certified design by "
                           "construction; reported as an audit."),
    ("at_cap", "Per-DOF flag: the solution hit the artificial box bound rho <= rho_max rather "
               "than the constraint. Cap-limited results are not design advice."),
    ("N", "Number of sampled FOV points, (w/stride + 1) x (h/stride + 1); 2 307 121 at stride 1."),
]

REFERENCES = [
    ("Cutting-plane method (the solver loop itself).",
     "J. E. Kelley, \"The Cutting-Plane Method for Solving Convex Programs\", Journal of the "
     "Society for Industrial and Applied Mathematics 8(4), 703-712, 1960.",
     "the original algorithm: outer-approximate a convex feasible set by accumulated subgradient "
     "halfspaces and re-solve the LP."),
    ("Modern treatment, convergence and query-point choice.",
     "S. Boyd and L. Vandenberghe, \"Localization and Cutting-Plane Methods\", Stanford EE364b "
     "lecture notes, https://web.stanford.edu/class/ee364b/lectures/localization_methods_notes.pdf",
     "covers the basic cutting-plane/localization algorithm, measuring progress, "
     "Chebyshev/analytic-centre variants, and constraint dropping. Start here."),
    ("Why sup over a continuum is the hard part.",
     "R. Hettich and K. O. Kortanek, \"Semi-infinite programming: theory, methods, and "
     "applications\", SIAM Review 35(3), 380-429, 1993, doi:10.1137/1035089.",
     "our constraint sup_{X_w in X} (...) <= lambda is a semi-infinite program; the cutting-plane "
     "loop is its exchange/discretization method."),
    ("Subgradient of a pointwise maximum (why the maximizing row is a valid cut).",
     "J. M. Danskin, The Theory of Max-Min and its Application to Weapons Allocation Problems, "
     "Econometrics and Operations Research V, Springer, 1967.",
     "Danskin's theorem. Wikipedia's \"Danskin's theorem\" entry is an adequate summary; "
     "Bertsekas (1971) generalizes it."),
    ("Convex-analysis background: subgradients, support functions, positive homogeneity.",
     "R. T. Rockafellar, Convex Analysis, Princeton University Press, 1970 (see the chapters on "
     "support functions and on subgradients -- pinpoint theorem numbers unverified, confirm "
     "before citing them); J.-B. Hiriart-Urruty and C. Lemarechal, Convex Analysis and "
     "Minimization Algorithms I-II, Springer, 1993.",
     "the homogeneity/subgradient facts that make every cut globally valid."),
    ("Textbook framing of the design problem.",
     "S. Boyd and L. Vandenberghe, Convex Optimization, Cambridge University Press, 2004. "
     "Section numbers unverified -- confirm the experiment-design section number before citing it.",
     "linear programming and vertex optima, separating/supporting hyperplanes, and the "
     "experiment-design section whose D-optimal log det objective is the direct analogue of our "
     "max sum_i log rho_i balanced allocation."),
    ("Optimal experiment design (the geomean objective's pedigree).",
     "F. Pukelsheim, Optimal Design of Experiments, SIAM Classics in Applied Mathematics 50, "
     "2006 (orig. Wiley, 1993).",
     "the D-optimality lineage of the balanced log-allocation objective."),
    ("The modelling layer the inner solves go through.",
     "S. Diamond and S. Boyd, \"CVXPY: A Python-Embedded Modeling Language for Convex "
     "Optimization\", Journal of Machine Learning Research 17(83), 1-5, 2016.",
     "cvxpy builds both inner programs; it selects the backend and returns the solver "
     "statistics recorded as solver/inner_iters/inner_time_s in the JSON."),
    ("The LP solver actually called.",
     "Q. Huangfu and J. A. J. Hall, \"Parallelizing the dual revised simplex method\", "
     "Mathematical Programming Computation 10(1), 119-142, 2018.",
     "HiGHS, reached as cvxpy's SCIPY backend with method=\"highs\"; it solves the linear "
     "objective (vertex_lp_norm)."),
    ("The nonlinear solver actually called.",
     "P. J. Goulart and Y. Chen, \"Clarabel: An interior-point solver for conic programs with "
     "quadratic objectives\", arXiv:2405.12762, 2024.",
     "the homogeneous-embedding interior-point method behind cvxpy's CLARABEL backend; the "
     "geomean objective is solved as an exponential-cone program."),
    ("The lambda_max <= tr <= 2 lambda_max sandwich.",
     "R. A. Horn and C. R. Johnson, Matrix Analysis, 2nd ed., Cambridge University Press, 2012.",
     "for an n x n PSD matrix, lambda_max <= tr <= n lambda_max; n = 2 here."),
]


# ==========================================================
# Vectorized FOV Jacobian sampler
# ==========================================================

def batch_jacobian(camera, altitude, z=0.0, stride=1.0):
    """J_xi at every pixel of the FOV back-projected onto the plane Z=z.

    Returns (J, uv, Pw) with J (N,2,6), uv (N,2) pixel coords, Pw (N,3) world
    points.  Row-for-row identical to
    ``ProjectionModel.analytic_jacobian(SE3.nominal_pose(altitude), Pw)``
    for a nadir pose (R = I, t = [0,0,h]), where xc = Pw + t has
    xc = [x, y, altitude - z] for a ground point at (x, y, z).
    """
    d = altitude - z
    us = np.arange(0.0, camera.w + 1e-9, stride)
    vs = np.arange(0.0, camera.h + 1e-9, stride)
    U, V = np.meshgrid(us, vs, indexing="ij")
    x = ((U - camera.cx) / camera.fx * d).ravel()
    y = ((V - camera.cy) / camera.fy * d).ravel()
    n = x.size
    zc = np.full(n, d)

    Jp = np.zeros((n, 2, 3))
    Jp[:, 0, 0] = camera.fx / zc
    Jp[:, 0, 2] = -camera.fx * x / zc ** 2
    Jp[:, 1, 1] = camera.fy / zc
    Jp[:, 1, 2] = -camera.fy * y / zc ** 2

    Jx = np.zeros((n, 3, 6))                    # [I | -skew(xc)]
    Jx[:, 0, 0] = Jx[:, 1, 1] = Jx[:, 2, 2] = 1.0
    Jx[:, 0, 4] = zc
    Jx[:, 0, 5] = -y
    Jx[:, 1, 3] = -zc
    Jx[:, 1, 5] = x
    Jx[:, 2, 3] = y
    Jx[:, 2, 4] = -x

    J = np.einsum("nij,njk->nik", Jp, Jx)
    uv = np.stack([U.ravel(), V.ravel()], 1)
    Pw = np.stack([x, y, np.full(n, z)], 1)
    return J, uv, Pw, (us.size, vs.size)


def _sigma_entries(Jc, s):
    """Entries (a, b, c) of Sigma_u = J diag(s) J^T for a chunk of rows."""
    W = Jc * s                                  # (m,2,6)
    a = np.einsum("nk,nk->n", W[:, 0], Jc[:, 0])
    b = np.einsum("nk,nk->n", W[:, 0], Jc[:, 1])
    c = np.einsum("nk,nk->n", W[:, 1], Jc[:, 1])
    return a, b, c


def _lmax_2x2(a, b, c):
    """Closed-form largest eigenvalue of [[a,b],[b,c]] (symmetric PSD)."""
    half = 0.5 * (a + c)
    disc = np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0.0))
    return half + disc


class FovFields:
    """Full-resolution FOV fields for one altitude.

    Holds the per-pixel Jacobians and the per-DOF gain matrix
    ``g[n, i] = ||j_i(X_n)||_2^2`` so that every trace statistic is a matrix
    product and the exact eigenvalue statistics stream over ``J`` in chunks.
    """

    def __init__(self, camera, altitude, z=0.0, stride=1.0, chunk=262144):
        self.camera = camera
        self.altitude = float(altitude)
        self.z = float(z)
        self.stride = float(stride)
        self.chunk = int(chunk)

        t0 = time.time()
        self.J, self.uv, self.Pw, self.shape = batch_jacobian(
            camera, altitude, z=z, stride=stride)
        self.g = np.einsum("nik,nik->nk", self.J, self.J)
        self.N = self.g.shape[0]
        self.build_time = time.time() - t0

        # sup_X ||J_xi||_2^2 : the incumbent submultiplicative gain.
        self.supJ2, self.argsup_J2 = self._sup_lmax(np.ones(6))

        ext_x = self.Pw[:, 0].max() - self.Pw[:, 0].min()
        ext_y = self.Pw[:, 1].max() - self.Pw[:, 1].min()
        self.footprint_area = float(ext_x * ext_y)

    # -- trace statistics ---------------------------------------------------
    def trace(self, s):
        return self.g @ np.asarray(s, dtype=float)

    # -- exact eigenvalue statistics ---------------------------------------
    def lmax(self, s):
        """Pointwise lambda_max(J diag(s) J^T) over the whole FOV."""
        s = np.asarray(s, dtype=float)
        out = np.empty(self.N)
        for i in range(0, self.N, self.chunk):
            j = min(i + self.chunk, self.N)
            a, b, c = _sigma_entries(self.J[i:j], s)
            out[i:j] = _lmax_2x2(a, b, c)
        return out

    def _sup_lmax(self, s):
        """(sup lambda_max, argsup row index) without materializing the field."""
        s = np.asarray(s, dtype=float)
        best, best_n = -np.inf, 0
        for i in range(0, self.N, self.chunk):
            j = min(i + self.chunk, self.N)
            a, b, c = _sigma_entries(self.J[i:j], s)
            lam = _lmax_2x2(a, b, c)
            k = int(np.argmax(lam))
            if lam[k] > best:
                best, best_n = float(lam[k]), i + k
        return best, best_n

    def sup_lmax(self, s):
        return self._sup_lmax(s)[0]

    def top_eigvec(self, s, n):
        """Unit eigenvector of Sigma_u(X_n) for the largest eigenvalue."""
        a, b, c = _sigma_entries(self.J[n:n + 1], np.asarray(s, dtype=float))
        S = np.array([[a[0], b[0]], [b[0], c[0]]])
        return np.linalg.eigh(S)[1][:, -1]

    def reshape(self, field):
        """(N,) FOV field -> (nv, nu) image for imshow."""
        nu, nv = self.shape
        return field.reshape(nu, nv).T


# ==========================================================
# Constraint oracles (certified variants only)
# ==========================================================

class Oracle:
    """f(s) with a valid homogeneous cut: ``cut . s == f(s)``.

    Every f registered here is convex and positively homogeneous of degree 1,
    so the cut is a subgradient, ``{f <= lam}`` is contained in
    ``{cut . s' <= lam}``, and the cut is a valid global outer approximation.
    Every f is also *certified*: ``f(s) <= lam`` implies
    ``lambda_u,max(s) <= lam``.
    """

    def __init__(self, key, fn, certified, desc):
        self.key = key
        self.fn = fn
        self.certified = bool(certified)
        self.desc = desc

    def __call__(self, fields, s):
        return self.fn(fields, np.asarray(s, dtype=float))

    def __repr__(self):
        return f"Oracle({self.key})"


def _spectral(fields, s):
    i = int(np.argmax(s))
    cut = np.zeros(6)
    cut[i] = fields.supJ2
    return float(fields.supJ2 * s[i]), cut



def _sup_trace(fields, s):
    t = fields.trace(s)
    n = int(np.argmax(t))
    cut = fields.g[n].copy()
    return float(t[n]), cut


def _exact_lmax(fields, s):
    val, n = fields._sup_lmax(s)
    w = fields.top_eigvec(s, n)
    proj = fields.J[n].T @ w                    # (6,) = (w^T j_i)_i
    cut = proj ** 2
    return float(val), cut


def build_constraints():
    """Registry of certified upper bounds on lambda_u,max, loosest first."""
    reg = {}
    reg["spectral"] = Oracle(
        "spectral", _spectral, True,
        "sup_X ||J_xi||_2^2 * max_i s_i  (incumbent submultiplicative bound)")
    reg["sup_trace"] = Oracle(
        "sup_trace", _sup_trace, True,
        "max_n g_n . s  (worst-case trace bound)")
    reg["exact_lmax"] = Oracle(
        "exact_lmax", _exact_lmax, True,
        "max_n lambda_max(J_n diag(s) J_n^T)  (exact reference)")
    return reg


CONSTRAINTS = build_constraints()


# ==========================================================
# Objective policies
# ==========================================================

@dataclass(frozen=True)
class Objective:
    """One point-selection rule on the feasible polyhedron.

    kind == "linear":  max sum_i s_i / w_i        (LP vertex)
    kind == "log":     max sum_i log(s_i / w_i)   (balanced, strictly concave)
    kind == "ray":     max kappa s.t. s = kappa s_base  (closed form, no solver)
    """

    key: str
    kind: str
    desc: str


LINEAR_SOLVER = dict(solver=cp.SCIPY, scipy_options={"method": "highs"})
# Default Clarabel tolerances (1e-8) move the geomean optimum by ~5e-5 relative
# and can cost an extra cut; the published results are reproduced at 1e-6 only
# with the gap/feasibility tolerances tightened.  At 1e-12 Clarabel sometimes
# reports "optimal_inaccurate" while landing within 4e-9 of the optimum; that
# status is accepted because feasibility and optimality of the returned design
# are re-verified downstream (audit, and the golden test).
LOG_SOLVER = dict(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12,
                  tol_feas=1e-12)


def _inner_solve(objective, A_x, b, x_caps, coef):
    """max objective over {A_x x <= b, 0 <= x <= x_caps}, in solve coordinates x.

    ``coef[i] = D_i / w_i`` maps the solve variable to the objective's units:
    s = D * x and the objective is stated in units of w (see Objective.kind).
    Returns (x, info) with info = {"solver", "status", "inner_iters", "inner_time_s"}.
    """
    x = cp.Variable(6, nonneg=True)
    if objective.kind == "linear":
        expr, opts = cp.sum(cp.multiply(coef, x)), LINEAR_SOLVER
    elif objective.kind == "log":
        expr, opts = cp.sum(cp.log(cp.multiply(coef, x))), LOG_SOLVER
    else:
        raise ValueError(objective.kind)
    prob = cp.Problem(cp.Maximize(expr), [A_x @ x <= b, x <= x_caps])
    prob.solve(**opts)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"{objective.key}: cvxpy status {prob.status}")
    stats = prob.solver_stats
    info = {"solver": stats.solver_name, "status": prob.status,
            "inner_iters": int(stats.num_iters or 0),
            "inner_time_s": float(stats.solve_time or 0.0)}
    return np.asarray(x.value, dtype=float), info


# The design program is a *maximization* of admissible noise: the constraint is
# an upper bound, so with s >= 0 the minimization reading (min c^T s subject to
# f(s) <= lam) is degenerate -- its optimum is s = 0, a noiseless platform --
# and is not implemented.  Every objective below maximizes admissible variance;
# they differ only in how the budget is distributed across the six DOFs.
OBJECTIVES = {
    "vertex_lp_norm": Objective(
        "vertex_lp_norm", "linear",
        "max sum_i rho_i (largest total admissible noise budget, rho = s/s_base)"),
    "kappa_scaling": Objective(
        "kappa_scaling", "ray",
        "max kappa s.t. s = kappa s_base (uniform scaling of the deployed budget)"),
    "geomean": Objective(
        "geomean", "log",
        "max sum_i log rho_i (balanced allocation; scale-invariant, unique interior optimum)"),
}


def _project_feasible(oracle, fields, s, lam_target, caps):
    """Scale s onto the constraint surface. Exact: every f here is homogeneous
    of degree 1, so f(s * lam/f(s)) == lam, and clipping to caps only lowers f
    (all constraint functionals are monotone nondecreasing in s)."""
    val, _ = oracle(fields, s)
    if val > lam_target and val > 0:
        s = s * (lam_target / val)
    return np.minimum(s, caps)


# ==========================================================
# Cutting-plane solver engine
# ==========================================================

def _result(oracle, objective, fields, s, s_base, caps, lam_target, A, values,
            iters, converged, cycle, infos):
    A_arr = np.asarray(A, dtype=float).reshape(-1, 6)
    slack = (lam_target - A_arr @ s) if A_arr.size else np.zeros(0)
    final_val, final_cut = oracle(fields, s)
    return {
        "s": s,
        "rho": s / s_base,
        "cuts": A_arr,
        "slack": slack,
        "active": (np.abs(slack) <= 1e-6 * lam_target) if slack.size else slack,
        "values": values,
        "iterations": iters,
        # feasibility of the returned design is decided by the *recomputed*
        # constraint value, never by which break path the loop took
        "converged": bool(converged
                          and final_val <= lam_target * (1.0 + 1e-6)),
        "feasible": bool(final_val <= lam_target * (1.0 + 1e-6)),
        "cycle": bool(cycle),
        "solver": str(infos[-1]["solver"]) if infos else "none",
        "inner_iters": int(sum(i["inner_iters"] for i in infos)),
        "inner_time_s": float(sum(i["inner_time_s"] for i in infos)),
        "at_cap": (s >= caps * (1 - 1e-9)),
        "f_value": float(final_val),
        "final_cut": final_cut,
        "objective": objective.key,
        "constraint": oracle.key,
    }


def solve_design(oracle, lam_target, objective, caps=None, s_base=None,
                 fields=None, max_cuts=None, tol=1e-9, *,
                 coords="rho", weights="rho"):
    """Solve  max objective(s)  s.t.  f(s) <= lam_target,  0 <= s <= caps.

    Every oracle is convex, so the exact cutting-plane loop applies:
    homogeneity makes each subgradient cut ``a . s' <= lam`` a globally valid
    halfspace, the inner problem is an outer approximation, and termination at
    a point feasible for the true constraint certifies optimality.

    If the loop exits without the true constraint holding (cut set exhausted or
    max_cuts reached), the iterate is projected onto the constraint surface
    using homogeneity (``s <- s * lam / f(s)``), so the returned design is
    always feasible for its own constraint; the ``converged`` flag reports
    whether the loop actually certified it.

    coords:  "rho" solves in x = s / s_base (cut matrix column-scaled by
             s_base), "s" solves in x = s directly.  Mathematically identical
             feasible set; the switch exists to measure conditioning (see
             design_bound_ablation.py).
    weights: "rho" states the objective in multiples of s_base, "s" in raw
             m^2/rad^2.  Invariant for kind == "log" (a subtracted constant);
             NOT invariant for kind == "linear", which is the whole point of
             the ablation.  May also be a length-6 array of explicit weights.
    """
    if fields is None:
        raise ValueError("fields is required")
    s_base = np.ones(6) if s_base is None else np.asarray(s_base, dtype=float)
    caps = 1e3 * s_base if caps is None else np.asarray(caps, dtype=float)
    if max_cuts is None:
        max_cuts = 200

    if coords == "rho":
        D = s_base
    elif coords == "s":
        D = np.ones(6)
    else:
        raise ValueError(f"coords={coords!r}")
    if isinstance(weights, str):
        if weights == "rho":
            w = s_base
        elif weights == "s":
            w = np.ones(6)
        else:
            raise ValueError(f"weights={weights!r}")
    else:
        w = np.asarray(weights, dtype=float)
    coef = D / w
    x_caps = caps / D
    ray = s_base / D                      # the scaling ray in solve coordinates

    A, values, seen, infos = [], [], [], []
    cycle = False

    s = caps.copy()
    converged = False
    for iters in range(1, max_cuts + 1):
        val, cut = oracle(fields, s)
        values.append(float(val))
        if val <= lam_target * (1.0 + tol):
            converged = True
            break
        key = cut / max(float(np.max(np.abs(cut))), 1e-300)
        if any(np.allclose(key, k, rtol=1e-12, atol=1e-14) for k in seen):
            cycle = True                     # cut set exhausted, no progress
            break
        seen.append(key)
        A.append(cut)
        A_x = np.asarray(A, dtype=float) * D            # rescale columns
        b = np.full(len(A), lam_target)
        if objective.kind == "ray":
            # max kappa s.t. kappa (a_k . s_base) <= lam for every cut k
            kappa = float(np.min(b / (A_x @ ray)))
            x = np.minimum(kappa * ray, x_caps)
            info = {"solver": "closed-form", "status": "optimal",
                    "inner_iters": 0, "inner_time_s": 0.0}
        else:
            x, info = _inner_solve(objective, A_x, b, x_caps, coef)
        infos.append(info)
        s = x * D
    if not converged:
        s = _project_feasible(oracle, fields, s, lam_target, caps)
    return _result(oracle, objective, fields, s, s_base, caps, lam_target,
                   A, values, iters, converged, cycle, infos)


def kappa_star(oracle, fields, s_base, lam_target):
    """Closed-form largest uniform scaling: f(kappa s_base) = lam_target."""
    v, _ = oracle(fields, s_base)
    return lam_target / v, v


# ==========================================================
# Post-hoc audit
# ==========================================================

def audit(fields, s_star, lam_target, oracle=None, caps=None):
    """Statistics achieved by a design and the guarantee actually delivered."""
    s_star = np.asarray(s_star, dtype=float)
    t = fields.trace(s_star)
    lam = fields.lmax(s_star)

    sup_trace = float(t.max())
    sup_lmax = float(lam.max())
    # relative tolerance: an exactly-binding design puts the argsup pixel at
    # lambda_target up to float rounding, which must not count as a violation
    viol = float(np.mean(lam > lam_target * (1.0 + 1e-9)))
    rec = {
        "s": s_star.tolist(),
        "sup_trace": sup_trace,
        "sup_lmax": sup_lmax,
        "gap_ratio": sup_trace / sup_lmax if sup_lmax > 0 else float("nan"),
        "mean_trace": float(t.mean()),
        "median_trace": float(np.median(t)),
        "p95_trace": float(np.quantile(t, 0.95)),
        "violation_frac": viol,
        "violation_area_m2": viol * fields.footprint_area,
        "p99_lmax": float(np.quantile(lam, 0.99)),
        "max_overshoot": sup_lmax / lam_target,
    }
    if oracle is not None:
        val, cut = oracle(fields, s_star)
        share = cut * s_star
        rec["f_value"] = float(val)
        rec["binding_dofs"] = np.flatnonzero(share > 0.01 * lam_target).tolist()
        rec["budget_share"] = (share / val if val > 0 else share * 0.0).tolist()
        rec["certified"] = oracle.certified
    if caps is not None:
        caps = np.asarray(caps, dtype=float)
        rec["at_cap"] = (s_star >= caps * (1 - 1e-9)).tolist()
    return rec


# ==========================================================
# Physical-units design heuristic
# ==========================================================

def admissible_sigma(s):
    """Design vector -> per-DOF 1-sigma in report units.

    Returns (sigma_mm, sigma_deg): translation DOFs 1-3 as mm (sqrt(s)*1e3),
    rotation DOFs 4-6 as degrees (degrees(sqrt(s))). Entries not belonging to a
    group are nan, so the two arrays can be printed side by side.
    """
    s = np.asarray(s, dtype=float)
    sig = np.sqrt(np.maximum(s, 0.0))
    sigma_mm = np.full(6, np.nan)
    sigma_deg = np.full(6, np.nan)
    sigma_mm[:3] = sig[:3] * 1e3
    sigma_deg[3:] = np.degrees(sig[3:])
    return sigma_mm, sigma_deg


def design_heuristic(fields, s_base, radii, rho_max, oracle, objective):
    """Admissible per-DOF 1-sigma pose budget vs desired pixel fidelity.

    Solves once at lam = (radii[0]/3)**2 and rescales: the feasible set
    {A rho <= lam} is positively homogeneous in lam, so rho*(lam) = (lam/lam0)
    rho*(lam0) and sigma* scales linearly in r. Returns one row per radius with
    r, lam, rho, s, sigma_mm, sigma_deg, plus the solve's cut count and
    convergence flag.
    """
    s_base = np.asarray(s_base, dtype=float)
    radii = [float(r) for r in radii]
    caps = float(rho_max) * s_base
    lam0 = (radii[0] / 3.0) ** 2
    base = solve_design(oracle, lam0, objective, caps=caps, s_base=s_base,
                        fields=fields)
    rho0 = base["rho"]

    rows = []
    for r in radii:
        lam = (r / 3.0) ** 2
        rho = rho0 * (lam / lam0)
        s = rho * s_base
        sigma_mm, sigma_deg = admissible_sigma(s)
        rows.append({
            "r": r,
            "lam": lam,
            "rho": rho.tolist(),
            "s": s.tolist(),
            "sigma_mm": sigma_mm.tolist(),
            "sigma_deg": sigma_deg.tolist(),
            "at_cap": bool(np.any(rho >= float(rho_max) * (1 - 1e-9))),
            "at_cap_dofs": (rho >= float(rho_max) * (1 - 1e-9)).tolist(),
            "n_cuts": int(base["cuts"].shape[0]),
            "converged": bool(base["converged"]),
            "constraint": oracle.key,
            "objective": objective.key,
        })
    return rows


# ==========================================================
# Self-tests: scripts/test_design_bound_lp.py (pytest), reachable through
# --self-test.  No in-module harness.
# ==========================================================


@dataclass
class Study:
    """Everything a run produces: JSON-serializable records plus the cached
    reference-altitude FOV fields the report pages need."""

    meta: dict
    per_alt: dict           # {float altitude: {"stats": ..., "variants": ...}}
    heuristic: dict         # {float altitude: [row, ...]}
    ref_fields: object = None   # FovFields at meta["reference_altitude"], or None

    def payload(self):
        return {"meta": self.meta,
                "altitudes": {f"{a:g}": self.per_alt[a] for a in self.per_alt},
                "heuristic": {f"{a:g}": self.heuristic[a] for a in self.heuristic}}

    def save_json(self, path):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.payload(), fh, indent=1)
        return path

    @classmethod
    def from_payload(cls, payload):
        return cls(meta=payload["meta"],
                   per_alt={float(k): v for k, v in payload["altitudes"].items()},
                   heuristic={float(k): v for k, v in payload["heuristic"].items()})

    @classmethod
    def load_json(cls, path):
        with open(path) as fh:
            return cls.from_payload(json.load(fh))

    def fields(self):
        """Reference-altitude FovFields, rebuilt from meta when absent (the
        JSON record cannot carry them)."""
        if self.ref_fields is None:
            self.ref_fields = FovFields(Camera(**self.meta["camera"]),
                                        self.meta["reference_altitude"],
                                        z=self.meta["z"],
                                        stride=self.meta["stride"])
        return self.ref_fields


# ==========================================================
# Experiment driver
# ==========================================================

def run_altitude(camera, altitude, lam_target, s_base, z, stride, rho_max,
                 constraints, objectives, log=print):
    F = FovFields(camera, altitude, z=z, stride=stride)
    t0 = F.trace(s_base)
    lm0 = F.lmax(s_base)
    caps = rho_max * s_base

    stats = {
        "altitude": float(altitude),
        "N": int(F.N),
        "shape": list(F.shape),
        "stride": float(stride),
        "build_time_s": F.build_time,
        "supJ2": float(F.supJ2),
        "argsup_J2_uv": F.uv[F.argsup_J2].tolist(),
        "footprint_area_m2": F.footprint_area,
        "sup_trace": float(t0.max()),
        "sup_lmax": float(lm0.max()),
        "gap_ratio": float(t0.max() / lm0.max()),
        "mean_trace": float(t0.mean()),
        "median_trace": float(np.median(t0)),
        "p95_trace": float(np.quantile(t0, 0.95)),
        "g_min": F.g.min(0).tolist(),
        "g_max": F.g.max(0).tolist(),
    }
    log(f"[ALT] h={altitude:g} m  N={F.N}  sup_trace={stats['sup_trace']:.1f} "
        f"sup_lmax={stats['sup_lmax']:.1f} gap={stats['gap_ratio']:.3f} "
        f"supJ2={stats['supJ2']:.4e}  ({F.build_time:.1f} s build)")

    kap_exact, _ = kappa_star(constraints["exact_lmax"], F, s_base, lam_target)
    variants = {}
    for key, orc in constraints.items():
        kap, f_base = kappa_star(orc, F, s_base, lam_target)
        rec = {
            "certified": orc.certified,
            "desc": orc.desc,
            "f_base": float(f_base),
            "kappa": float(kap),
            "conservatism_vs_exact": float(kap_exact / kap),
            "audit_kappa": audit(F, kap * s_base, lam_target, oracle=orc,
                                 caps=caps),
            "solutions": {},
        }
        for oname, obj in objectives.items():
            r = solve_design(orc, lam_target, obj, caps=caps, s_base=s_base,
                             fields=F)
            a = audit(F, r["s"], lam_target, oracle=orc, caps=caps)
            rec["solutions"][oname] = {
                "rho": r["rho"].tolist(),
                "s": r["s"].tolist(),
                "iterations": r["iterations"],
                "converged": r["converged"],
                "cycle": r["cycle"],
                "solver": r["solver"],
                "inner_iters": r["inner_iters"],
                "inner_time_s": r["inner_time_s"],
                "at_cap": r["at_cap"].tolist(),
                "f_value": r["f_value"],
                "n_cuts": int(r["cuts"].shape[0]),
                "audit": a,
            }
        log(f"[VAR] {key:13s} f(s_base)={f_base:11.2f} kappa*={kap:9.3e} "
            f"cons={rec['conservatism_vs_exact']:8.2f}x "
            f"sup_lmax(kappa)={rec['audit_kappa']['sup_lmax']:9.1f} "
            f"viol={rec['audit_kappa']['violation_frac']:.4f} certified")
        variants[key] = rec

    return F, stats, variants


def run_study(camera, altitudes, lam_target, s_base, z=0.0, stride=1.0,
              rho_max=1e3, radii=None, std_devs=None, constraints=None,
              objectives=None, keep_fields=True, log=print):
    """Run the altitude sweep and the design heuristic.  Pure computation: no
    plotting, no file I/O."""
    radii = DEFAULT_RADII if radii is None else radii
    constraints = build_constraints() if constraints is None else constraints
    objectives = OBJECTIVES if objectives is None else objectives

    log(f"[CFG] s_base = {np.array2string(np.asarray(s_base), precision=8)}")
    log(f"[CFG] lam_target = {lam_target:g} px^2, stride = {stride:g}, "
        f"rho_max = {rho_max:g}")
    log(f"[CFG] heuristic radii = {list(radii)} px (3-sigma)")

    ref_alt = 10.0 if 10.0 in altitudes else float(altitudes[0])
    per_alt, ref_fields, heuristic = {}, None, {}
    t_start = time.time()

    for alt in altitudes:
        F, stats, variants = run_altitude(
            camera, alt, lam_target, s_base, z, stride,
            rho_max, constraints, objectives, log=log)
        per_alt[float(alt)] = {"stats": stats, "variants": variants}
        rows = design_heuristic(F, s_base, radii, rho_max,
                                constraints["exact_lmax"],
                                objectives["geomean"])
        heuristic[float(alt)] = rows
        for row in rows:
            mm = np.asarray(row["sigma_mm"], dtype=float)
            dg = np.asarray(row["sigma_deg"], dtype=float)
            log(f"[HEUR] h={alt:g} m r={row['r']:5.1f} px "
                f"lam={row['lam']:8.2f} "
                f"sigma_xyz=[{mm[0]:7.2f},{mm[1]:7.2f},{mm[2]:8.2f}] mm "
                f"sigma_rpy=[{dg[3]:7.4f},{dg[4]:7.4f},{dg[5]:7.4f}] deg "
                f"at_cap={row['at_cap']}")
        if float(alt) == ref_alt and keep_fields:
            ref_fields = F
        else:
            del F

    meta = {
        "camera": DEFAULT_CAMERA,
        "altitudes": [float(a) for a in altitudes],
        "reference_altitude": ref_alt,
        "lam_target": float(lam_target),
        "stride": float(stride),
        "std_devs": [float(v) for v in
                     (DEFAULT_STD_DEVS if std_devs is None else std_devs)],
        "s_base": np.asarray(s_base, dtype=float).tolist(),
        "rho_max": float(rho_max),
        "radii": [float(r) for r in radii],
        "z": float(z),
        "constraints": {k: o.desc for k, o in constraints.items()},
        "objectives": {k: o.desc for k, o in objectives.items()},
        "heuristic_constraint": "exact_lmax",
        "heuristic_objective": "geomean",
        "runtime_s": time.time() - t_start,
    }
    return Study(meta=meta, per_alt=per_alt, heuristic=heuristic,
                 ref_fields=ref_fields)



# ==========================================================
# Report
# ==========================================================

def _wrap(lines, width=155):
    out = []
    for ln in lines:
        if len(ln) <= width:
            out.append(ln)
            continue
        indent = " " * (len(ln) - len(ln.lstrip()) + 2)
        out.extend(textwrap.wrap(ln, width, subsequent_indent=indent,
                                 break_long_words=False))
    return out


def _text_pages(pdf, lines, per_page=78, fontsize=5.9):
    """Emit monospace text pages, chunked so nothing runs off the sheet."""
    wrapped = _wrap(lines)
    for i in range(0, max(len(wrapped), 1), per_page):
        chunk = wrapped[i:i + per_page]
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.035, 0.97, "\n".join(chunk), va="top", ha="left",
                 family="monospace", fontsize=fontsize)
        pdf.savefig(fig)
        plt.close(fig)


def _page_definitions(pdf, meta):
    lines = [
        "DEFINITIONS & NOTATION",
        "",
        "Every symbol appearing in a figure, table or JSON key of this report is defined here.",
        "The same glossary, verbatim, is section 2 of scripts/design_bound_lp.md.",
        "",
    ]
    for term, desc in DEFINITIONS:
        head = f"  {term:34s}"
        body = textwrap.wrap(desc, 155 - len(head), break_long_words=False)
        lines.append(head + body[0])
        for cont in body[1:]:
            lines.append(" " * len(head) + cont)
    lines += [
        "",
        f"Numeric context for this run: lambda_u,target = {meta['lam_target']:g} px^2 "
        f"(r = 3 sqrt(lambda) = {3.0 * np.sqrt(meta['lam_target']):.0f} px), "
        f"stride = {meta['stride']:g}, "
        f"rho_max = {meta['rho_max']:g}, altitudes {meta['altitudes']} m.",
        "s_base = " + np.array2string(np.asarray(meta["s_base"]), precision=8,
                                      max_line_width=250) + " m^2 / rad^2.",
        "",
        "MANUSCRIPT CONVENTION: lambda_u,target = (r/3)^2 with r the 3-sigma pixel radius, so the "
        "default 100 px^2 is r = 30 px, equivalently a",
        "1-sigma pixel error of 10 px.  This is the deployed soft-label parameter of the BirdsEye "
        "annotation pipeline (Gaussian humps drawn to a 3-sigma",
        "radius with sigma = 10 px, i.e. 60 px diameter patches), so the admissible pose budget "
        "reported here is the budget required to honour the label",
        "radius actually used in training.  The heuristic radii are converted with the same formula.",
    ]
    _text_pages(pdf, lines)


def _page_fields(pdf, F, lam_target, s_base):
    t = F.reshape(F.trace(s_base))
    lm = F.reshape(F.lmax(s_base))
    ext = [0, F.camera.w, F.camera.h, 0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    panels = [
        (t, r"$\mathrm{tr}(J\Sigma_0 J^T)$  [px$^2$]", "viridis"),
        (lm, r"exact $\lambda_{\max}(\Sigma_u)$  [px$^2$]", "viridis"),
        (t / lm, r"ratio $\mathrm{tr}/\lambda_{\max}$  ($\leq 2$)", "magma"),
        ((lm > lam_target).astype(float),
         rf"violation mask $\lambda_{{\max}} > {lam_target:g}$", "RdYlGn_r"),
    ]
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(data, extent=ext, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"FOV fields at deployed $s_{{base}}$, h = {F.altitude:g} m, "
                 f"N = {F.N} pixels (stride {F.stride:g})")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_gains(pdf, F):
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    ext = [0, F.camera.w, F.camera.h, 0]
    for i, ax in enumerate(axes.ravel()):
        gi = F.reshape(F.g[:, i])
        im = ax.imshow(np.log10(gi + 1e-12), extent=ext, cmap="cividis",
                       aspect="auto")
        ax.set_title(f"$g_{{{i+1}}}$ ({DOF_LABELS[i]})\n"
                     f"[{F.g[:, i].min():.3e}, {F.g[:, i].max():.3e}]",
                     fontsize=9)
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\log_{10} g_i$")
    fig.suptitle(rf"Per-DOF gain maps $g_i(X_w) = \|j_i\|_2^2$, "
                 rf"h = {F.altitude:g} m")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_hist(pdf, F, lam_target, s_base):
    t = F.trace(s_base)
    lm = F.lmax(s_base)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    for ax, data, name in ((axes[0], t, r"$\mathrm{tr}(J\Sigma_0J^T)$"),
                           (axes[1], lm, r"$\lambda_{\max}(\Sigma_u)$")):
        pos = data[data > 0]
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 200)
        ax.hist(pos, bins=bins, color="steelblue", alpha=0.8)
        marks = [("mean", data.mean(), "tab:green"),
                 ("median", np.median(data), "tab:olive"),
                 ("sup", data.max(), "tab:red")]
        for label, val, col in marks:
            ax.axvline(val, color=col, ls="--", lw=1.2,
                       label=f"{label} = {val:.1f} (descriptive only "
                             "- not a constraint)")
        ax.axvline(lam_target, color="k", lw=2,
                   label=rf"$\lambda_{{u,target}}$ = {lam_target:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(name + r"  [px$^2$]")
        ax.set_ylabel("pixel count")
        ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"FOV distribution of {name} at $s_{{base}}$", fontsize=10)
    fig.suptitle(f"Statistic distributions, h = {F.altitude:g} m "
                 "(only the sup enters a certified constraint)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_table(pdf, variants, altitude, lam_target):
    cols = ["variant", "f(s_base)", "kappa*", "conserv.", "sup_lmax",
            "gap tr/lmax", "viol frac", "overshoot", "certified"]
    rows = []
    for key, rec in variants.items():
        a = rec["audit_kappa"]
        rows.append([key, f"{rec['f_base']:.2f}", f"{rec['kappa']:.4g}",
                     f"{rec['conservatism_vs_exact']:.3g}x",
                     f"{a['sup_lmax']:.1f}", f"{a['gap_ratio']:.3f}",
                     f"{a['violation_frac']:.4f}",
                     f"{a['max_overshoot']:.2f}x",
                     "yes" if rec["certified"] else "NO"])

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, loc="center",
                   cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.7)
    ax.set_title(
        f"Certified constraint variants at h = {altitude:g} m, "
        rf"$\lambda_{{u,target}}$ = {lam_target:g} px$^2$ "
        "(uniform-scaling design $s = \\kappa^* s_{base}$)\n"
        "all three variants are upper bounds on $\\lambda_{u,max}$; "
        "conservatism = $\\kappa^*$(exact_lmax) / $\\kappa^*$(variant)",
        fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_objectives(pdf, meta, per_alt):
    """Where the objectives come from, what each one decides, what it invokes."""
    a10 = meta["reference_altitude"]
    v = per_alt[a10]["variants"]
    st = v["sup_trace"]["solutions"]
    ex = v["exact_lmax"]["solutions"]
    fmt = lambda x: np.array2string(np.asarray(x, dtype=float), precision=3,
                                    max_line_width=250)
    lines = [
        "THE OBJECTIVE FAMILY: development, the decisions it forces, and the structure each choice invokes",
        "",
        "STEP 0 -- why an objective is needed at all",
        "  The certified constraint defines a feasible set  F = {s : f(s) <= lambda, 0 <= s <= caps}.  Every point of F is a design that",
        "  provably meets the pixel-fidelity target.  F is 6-dimensional and has nonempty interior, so the constraint alone does not",
        "  produce a design: it rules out, it does not choose.  The objective is exactly the extra engineering preference that selects one",
        "  point of F, and it must be stated explicitly because different preferences give materially different hardware specs.",
        "  STRUCTURE: F is convex and compact.  For sup_trace it is a polytope with an explicit row; for exact_lmax it is the intersection",
        "  of infinitely many halfspaces (a spectrahedron), approached from outside by the accumulated cuts.",
        "",
        "STEP 1 -- decision: maximize or minimize?  (sign of the program)",
        "  The constraint is an UPPER bound and every f is monotone nondecreasing in s, so 'min c^T s subject to f(s) <= lambda, s >= 0'",
        "  attains its optimum at s = 0: a noiseless platform.  Degenerate, no information; not implemented.  The design question is",
        "  therefore necessarily a MAXIMIZATION of admissible noise: how bad may the pose estimate be and still meet the pixel target.",
        "  CONSEQUENCE: the objective must be strictly increasing in each s_i, which forces the optimum onto the constraint surface",
        "  f(s) = lambda.  Every reported design is active: f(s*) / lambda = 1.000000 for all (variant, objective) pairs.",
        "  STRUCTURE INVOKED: positive homogeneity of f.  The active surface is a scaled copy of itself, so solving at one lambda solves",
        "  every lambda:  rho*(lambda) = (lambda/lambda0) rho*(lambda0).  This is what makes the page-9 heuristic a single solve.",
        "",
        "STEP 2 -- decision: in which coordinates?  (commensurability of the six DOFs)",
        "  Summing variances across DOFs requires a common unit, and s mixes m^2 with rad^2.  'max sum_i s_i' in raw units is therefore",
        "  unit-degenerate: its answer changes if angles are expressed in degrees instead of radians, which is not a property a design",
        "  rule may have.  Two legitimate repairs exist: (a) normalize by the deployed budget, rho = s/s_base, giving dimensionless",
        "  multiples of hardware we actually own; (b) weight by a procurement cost vector c (currency per unit variance).  (b) needs cost",
        "  data that does not exist for this platform, so (a) is used and the raw-unit objective is deleted rather than kept as a variant.",
        "  STRUCTURE INVOKED: the cut matrix is rescaled columnwise, A_x = A * s_base.  This is a SEMANTIC choice only.  Measured by",
        "  design_bound_ablation.py (stride 4, h = 10/20 m): solving in raw s instead of rho moves the geomean optimum by <= 2.2e-12",
        "  (Clarabel) and never stalls -- the earlier 'unscaled solves stall at the initial point' claim does not reproduce.  Cost of",
        "  raw coordinates: Clarabel unchanged (34 ms vs 34 ms, 225 vs 192 interior iterations on spectral); the old trust-constr",
        "  backend degrades 2.8x in iterations (664 vs 237) and 2.1x in wall time (343 ms vs 161 ms).  Unit degeneracy is real and is",
        "  measured on the LP in STEP 3.",
        "",
        "STEP 3 -- decision: an extreme design or a balanced one?  (linear vs strictly concave objective)",
        "  A LINEAR objective over a polytope attains its optimum at a vertex (fundamental theorem of LP).  With one active row, a vertex",
        "  of F puts the entire budget into the single cheapest DOF and zero into the other five.  Measured, h = 10 m, sup_trace:",
        f"    vertex_lp_norm   rho* = {fmt(st['vertex_lp_norm']['rho'])}   budget share = {fmt(st['vertex_lp_norm']['audit']['budget_share'])}",
        "  That is the honest maximizer of sum_i rho_i and it is useless as a specification: it says the x, y, z and roll/pitch estimates",
        "  may be arbitrarily bad provided the rest is perfect.  It is also non-unique when rows tie.  Reported because it is the correct",
        "  answer to the question as literally posed -- which is the argument for not posing it that way.",
        "  MEASURED unit degeneracy of the raw-unit LP (ablation, sup_trace, h = 10 m): summing raw variances puts the whole budget on z",
        "  (rho* = [0, 0, 2.147, 0, 0, 0]); relabelling the rotations from radians to degrees moves the same objective's answer to yaw",
        "  (rho* = [0, 0, 0, 0, 0, 15.021]).  The normalized objective is invariant to that relabelling by construction.",
        "  If a design that spends on every DOF is wanted, the objective must be STRICTLY CONCAVE; then the optimum is interior and unique.",
        "  STRUCTURE INVOKED: strict concavity + a strictly feasible point (Slater) => unique maximizer, and KKT stationarity on a single",
        "  active row gives the closed form used to validate the solver (test_geomean_single_cut_closed_form).",
        "",
        "STEP 4 -- decision: which strictly concave objective?  (what 'balanced' means)",
        "  Candidates: sum_i log rho_i (geometric mean, D-optimality), sum_i rho_i^p with p < 1 (power means), min_i rho_i (max-min fair).",
        "  Each encodes a different fairness notion; none is neutral.  sum_i log rho_i is chosen because it is the only one of the three",
        "  that is scale-invariant in the per-DOF reference (rescaling s_base,i shifts the objective by a constant and so cannot change",
        "  the ranking of allocations), it is separable with analytic gradient and Hessian, and it is the diagonal case of the D-optimal",
        "  log det criterion of optimal experiment design -- references [6], [7].",
        "  STRUCTURE INVOKED, and worth reading as the design rule: on a single active row a . rho <= lambda, stationarity gives",
        "  rho_i = lambda / (6 a_i), i.e. a_i rho_i = lambda / 6 for every i -- EQUAL BUDGET SHARE.  Each DOF is allotted exactly one",
        "  sixth of the pixel-variance budget.  Measured, h = 10 m:",
        f"    sup_trace  + geomean  rho* = {fmt(st['geomean']['rho'])}   budget share = {fmt(st['geomean']['audit']['budget_share'])}",
        f"    exact_lmax + geomean  rho* = {fmt(ex['geomean']['rho'])}   budget share = {fmt(ex['geomean']['audit']['budget_share'])}",
        "  The shares are exactly 1/6 for sup_trace, whose feasible set has one active row.  For exact_lmax the optimum sits where several",
        "  eigenvalue linearizations intersect, so the equal-share identity holds per active row, not against the final cut alone; the",
        "  shares spread but no DOF is starved.  This is the allocation the page-9 heuristic reports.",
        "",
        "STEP 5 -- decision: six numbers or one?  (do we redesign the platform, or grade the one we have)",
        "  Restricting to the ray s = kappa s_base asks a different and often more useful question: may the EXISTING budget, with its",
        "  existing shape, be scaled up, and by how much.  It is a 1-D LP, always feasible, always unique, and needs no solver at all --",
        "  kappa* = lambda / f(s_base) in closed form by homogeneity.  It cannot rebalance DOFs, which is the point: procurement is",
        "  usually a single scalar margin on a fixed sensor suite.  Measured at h = 10 m:",
        f"    kappa*(spectral) = {v['spectral']['kappa']:.4e}   kappa*(sup_trace) = {v['sup_trace']['kappa']:.4f}   "
        f"kappa*(exact_lmax) = {v['exact_lmax']['kappa']:.4f}",
        "  STRUCTURE INVOKED: kappa* is the support-function value of F along the direction s_base; its reciprocal is f evaluated at the",
        "  deployed point, which is why no iteration is needed and why the number is exact to machine precision (self-test check 5).",
        "",
        "INTERACTION WITH THE CONSTRAINT -- the objective changes the solver's cost, not only its answer",
        f"  Cuts to certify optimality at h = {a10:g} m:  sup_trace needs "
        f"{st['vertex_lp_norm']['n_cuts']} (LP) and {st['geomean']['n_cuts']} (geomean); exact_lmax needs "
        f"{ex['vertex_lp_norm']['n_cuts']} (LP) and {ex['geomean']['n_cuts']} (geomean).",
        "  A vertex chases the argsup pixel: every reallocation to a corner moves the worst pixel, which demands a new linearization.",
        "  The interior point is stable under the same migration, so the balanced objective is both better advice and cheaper to certify.",
        "",
        "WHAT IS NOT AN OBJECTIVE HERE",
        "  Risk-weighted objectives (mean, quantile or CVaR of the pixel-error field) would change the CONSTRAINT, not the objective, and",
        "  none of them is an upper bound on lambda_u,max.  They are excluded by the certification requirement, not by preference; page 4",
        "  shows them as descriptive markers so the reader can see how much would be given away by accepting them.",
    ]
    _text_pages(pdf, lines, per_page=110, fontsize=5.0)


def _page_alloc(pdf, variants, altitude, objectives):
    panels = [o for o in ("vertex_lp_norm", "geomean") if o in objectives]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 8.5), squeeze=False)
    keys = list(variants.keys())
    width = 0.8 / max(len(keys), 1)
    for ax, oname in zip(axes.ravel(), panels):
        for i, key in enumerate(keys):
            sol = variants[key]["solutions"][oname]
            rho = np.maximum(np.asarray(sol["rho"]), 1e-12)
            x = np.arange(6) + i * width
            hatch = "//" if any(sol["at_cap"]) else None
            ax.bar(x, rho, width=width, label=key, hatch=hatch)
        ax.set_xticks(np.arange(6) + 0.4 - width / 2)
        ax.set_xticklabels(DOF_LABELS)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\rho^* = s^*/s_{base}$")
        ax.set_title(f"objective = {oname}  "
                     f"({OBJECTIVES[oname].desc})", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    axes.ravel()[0].legend(fontsize=6, ncol=4, loc="upper center")
    fig.suptitle(f"Budget allocation per DOF, h = {altitude:g} m "
                 "(hatched = cap-limited)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_altitude(pdf, per_alt):
    alts = sorted(per_alt.keys())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    ax = axes[0]
    keys = list(per_alt[alts[0]]["variants"].keys())
    for key in keys:
        y = [per_alt[a]["variants"][key]["kappa"] for a in alts]
        ax.plot(alts, y, "o-", label=key)
    ax.set_yscale("log")
    ax.set_xlabel("altitude [m]")
    ax.set_ylabel(r"$\kappa^* = \lambda_{u,target}/f(s_{base})$")
    ax.set_title("Admissible uniform scaling vs altitude "
                 "(all three variants are certified upper bounds)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=4)

    ax2 = axes[1]
    gap = [per_alt[a]["stats"]["gap_ratio"] for a in alts]
    ax2.plot(alts, gap, "ko-", label=r"$\sup\,\mathrm{tr}/\sup\lambda_{\max}$")
    ax2.axhline(2.0, color="r", ls=":", label="theoretical ceiling 2")
    ax2.set_xlabel("altitude [m]")
    ax2.set_ylabel("gap ratio")
    ax2.set_ylim(1.0, 2.1)
    ax2.grid(True, alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(alts, [per_alt[a]["stats"]["sup_trace"] for a in alts],
             "b^--", label=r"$\sup\,\mathrm{tr}$")
    ax3.plot(alts, [per_alt[a]["stats"]["sup_lmax"] for a in alts],
             "gv--", label=r"$\sup\lambda_{\max}$")
    ax3.set_ylabel(r"px$^2$ at $s_{base}$")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    ax2.set_title("Trace-vs-eigenvalue conservatism shrinks with altitude",
                  fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_heuristic(pdf, heuristic, meta):
    alts = sorted(heuristic.keys())
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, max(len(alts), 1), height_ratios=[1.35, 1.0],
                          hspace=0.35, wspace=0.45)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    cols = ["h [m]", "r [px]", "lam [px^2]",
            "sig_x [mm]", "sig_y [mm]", "sig_z [mm]",
            "sig_roll [deg]", "sig_pitch [deg]", "sig_yaw [deg]"]
    rows = []
    capped = False
    for a in alts:
        for row in heuristic[a]:
            mm = np.asarray(row["sigma_mm"], dtype=float)
            dg = np.asarray(row["sigma_deg"], dtype=float)
            star = "*" if row["at_cap"] else ""
            capped = capped or row["at_cap"]
            rows.append([f"{a:g}", f"{row['r']:.0f}", f"{row['lam']:.1f}"]
                        + [f"{mm[i]:.2f}{star}" for i in range(3)]
                        + [f"{dg[i]:.4f}{star}" for i in range(3, 6)])
    tab = ax.table(cellText=rows, colLabels=cols, loc="center",
                   cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.4)
    tab.scale(1.0, 1.06)

    for j, a in enumerate(alts):
        axp = fig.add_subplot(gs[1, j])
        rr = np.array([row["r"] for row in heuristic[a]], dtype=float)
        mm = np.array([row["sigma_mm"] for row in heuristic[a]], dtype=float)
        dg = np.array([row["sigma_deg"] for row in heuristic[a]], dtype=float)
        for i, col in zip(range(3), ("tab:blue", "tab:cyan", "tab:green")):
            axp.loglog(rr, mm[:, i], "o-", color=col, ms=3,
                       label=f"{DOF_LABELS[i]} [mm]")
        axt = axp.twinx()
        for i, col in zip(range(3, 6), ("tab:red", "tab:orange", "tab:purple")):
            axt.loglog(rr, dg[:, i], "s--", color=col, ms=3,
                       label=f"{DOF_LABELS[i]} [deg]")
        axt.set_yscale("log")
        axp.set_title(f"h = {a:g} m", fontsize=8)
        axp.set_xlabel(r"desired $3\sigma$ pixel radius $r$ [px]", fontsize=7)
        if j == 0:
            axp.set_ylabel(r"admissible $\sigma$ [mm]", fontsize=7)
        if j == len(alts) - 1:
            axt.set_ylabel(r"admissible $\sigma$ [deg]", fontsize=7)
        axp.grid(True, which="both", alpha=0.25)
        axp.tick_params(labelsize=6)
        axt.tick_params(labelsize=6)
        if j == 0:
            h1, l1 = axp.get_legend_handles_labels()
            h2, l2 = axt.get_legend_handles_labels()
            axp.legend(h1 + h2, l1 + l2, fontsize=5.5, ncol=2,
                       loc="upper left")

    fig.suptitle(
        "DESIGN HEURISTIC: admissible per-DOF 1$\\sigma$ pose accuracy vs "
        "altitude and desired pixel fidelity\n"
        "constraint = exact_lmax (tightest certified bound), allocation = "
        "geomean (balanced, all six DOFs), nadir pose; "
        r"$\lambda = (r/3)^2$, so $\sigma \propto r$ exactly (log-log slope 1)"
        + ("\n* = cap-limited (rho_max), not a design tolerance"
           if capped else "")
        + "\nRead this table for platform design: pick altitude and desired r, "
          "read the six 1$\\sigma$ tolerances.", fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def _page_methods(pdf, meta, per_alt, heuristic):
    a10 = meta["reference_altitude"]
    v = per_alt[a10]["variants"]
    cutinfo = ", ".join(
        f"{k}:{v[k]['solutions']['vertex_lp_norm']['n_cuts']}"
        f"{'' if v[k]['solutions']['vertex_lp_norm']['converged'] else '!'}"
        for k in v)
    hrow = heuristic[a10][0] if heuristic.get(a10) else None
    lines = [
        "METHODS",
        "",
        f"Camera: {meta['camera']['w']}x{meta['camera']['h']} px, "
        f"fx={meta['camera']['fx']:.6f}, fy={meta['camera']['fy']:.6f}, "
        f"cx={meta['camera']['cx']:.6f}, cy={meta['camera']['cy']:.6f}",
        f"Pose: nadir SE3.nominal_pose(h), ground plane z = {meta['z']:g} m; "
        f"altitudes {meta['altitudes']} m",
        f"Pixel stride {meta['stride']:g} (edges inclusive, so the frame corner "
        f"argsup is sampled exactly); N = {per_alt[a10]['stats']['N']} samples",
        f"lambda_u,target = {meta['lam_target']:g} px^2 "
        "(pixel-variance tolerance; sqrt(lambda) = 1-sigma pixel error, "
        f"r = 3 sqrt(lambda) = {3.0 * np.sqrt(meta['lam_target']):.0f} px 3-sigma radius)",
        f"std_devs = {meta['std_devs']} (m, m, m, deg, deg, deg)",
        "s_base = diag(CovarianceModel.realistic(std_devs)) = "
        + np.array2string(np.asarray(meta["s_base"]), precision=6,
                          max_line_width=250) + " m^2/rad^2",
        f"Heuristic radii r = {meta['radii']} px (3-sigma); lambda = (r/3)^2",
        "",
        "PROGRAM (all functionals are homogeneous of degree 1 in s)",
        "  g_i(X_w) = ||j_i(X_w)||_2^2,  tr(J diag(s) J^T) = s . g(X_w)",
        "  The design program MAXIMIZES admissible noise subject to a certified "
        "upper bound: max objective(s) s.t. f(s) <= lambda_u,target, 0 <= s <= caps.",
        "  The minimization reading (min c^T s under an upper bound, s >= 0) is "
        "degenerate -- its optimum is s = 0, a noiseless platform -- and is not implemented.",
    ]
    for k, rec in v.items():
        lines.append(f"  {k:13s} f(s) = {rec['desc']}")
    for k, obj in OBJECTIVES.items():
        lines.append(f"  {k:15s} objective: {obj.desc}")
    lines += [
        "",
        "SOLVER",
        "  Cutting-plane loop in rho = s/s_base; the rescaling is semantic, "
        "not numerical (ablation: raw-s identical to 2.2e-12, same cuts).",
        "  Homogeneity => each subgradient a satisfies f(s) = a.s, so every cut "
        "a.s' <= lambda is a globally valid halfspace (all f here are convex).",
        "  Inner solves via cvxpy 1.7: linear -> SCIPY/HiGHS simplex, log -> "
        "Clarabel (exp cone, tol 1e-12), kappa_scaling -> closed form.",
        f"  Caps: rho <= {meta['rho_max']:g}; cap-limited DOFs are flagged "
        "at_cap in the JSON, hatched on page 7 and starred on page 9.",
        f"  Cut counts (vertex_lp_norm, h={a10:g} m): {cutinfo}   "
        "('!' = not converged)",
        "",
        "CERTIFICATION",
        "  The registry admits certified variants only: f(s) <= lambda implies "
        "lambda_u,max = sup_X lambda_max(Sigma_u) <= lambda, for spectral, "
        "sup_trace and exact_lmax.  Descriptive trace statistics "
        "(mean, median, quantiles, CVaR) are NOT upper bounds and are not",
        "  registered as constraints; page 4 shows them as descriptive markers only.",
        "  Pointwise sandwich lambda_max <= tr <= 2 lambda_max and the full-FOV "
        "re-audit (sup_lmax <= lambda, viol = 0) run in test_design_bound_lp.py.",
        "",
        "HEADLINE RESULT (h = {:g} m)".format(a10),
        f"  incumbent spectral kappa* = {v['spectral']['kappa']:.4e}  "
        f"({v['spectral']['conservatism_vs_exact']:.0f}x conservative)",
        f"  sup_trace kappa*          = {v['sup_trace']['kappa']:.4f}  "
        f"({v['sup_trace']['conservatism_vs_exact']:.3f}x)",
        f"  exact_lmax kappa*         = {v['exact_lmax']['kappa']:.4f}  (1.000x)",
        f"  => the trace step recovers "
        f"{v['sup_trace']['kappa'] / v['spectral']['kappa']:.3g}x of design "
        "budget and leaves only the <= 2x (here "
        f"{per_alt[a10]['stats']['gap_ratio']:.3f}x) trace-vs-eigenvalue slack.",
    ]
    if hrow is not None:
        mm = np.asarray(hrow["sigma_mm"], dtype=float)
        dg = np.asarray(hrow["sigma_deg"], dtype=float)
        lines.append(
            f"  heuristic anchor (h = {a10:g} m, r = {hrow['r']:.0f} px, "
            f"exact_lmax + geomean): sigma = [{mm[0]:.2f}, {mm[1]:.2f}, "
            f"{mm[2]:.2f}] mm, [{dg[3]:.4f}, {dg[4]:.4f}, {dg[5]:.4f}] deg "
            f"({hrow['n_cuts']} cuts, converged={hrow['converged']}); "
            "sigma scales linearly in r.")
    lines += ["", "REFERENCES"]
    for n, (what, cite, note) in enumerate(REFERENCES, start=1):
        lines.append(f"  [{n}] {cite}")
        lines.append(f"      {what}  {note}")
    _text_pages(pdf, lines)


def build_report(out_pdf, study, objectives=None):
    """Render the 10-page PDF for a completed Study.  Rebuilds the reference
    FOV fields if the study was loaded from JSON."""
    objectives = OBJECTIVES if objectives is None else objectives
    meta, per_alt, heuristic = study.meta, study.per_alt, study.heuristic
    ref_fields = study.fields()
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    a_ref = meta["reference_altitude"]
    with PdfPages(out_pdf) as pdf:
        _page_definitions(pdf, meta)
        _page_fields(pdf, ref_fields, meta["lam_target"],
                     np.asarray(meta["s_base"]))
        _page_gains(pdf, ref_fields)
        _page_hist(pdf, ref_fields, meta["lam_target"],
                   np.asarray(meta["s_base"]))
        _page_table(pdf, per_alt[a_ref]["variants"], a_ref, meta["lam_target"])
        _page_objectives(pdf, meta, per_alt)
        _page_alloc(pdf, per_alt[a_ref]["variants"], a_ref, objectives)
        _page_altitude(pdf, per_alt)
        _page_heuristic(pdf, heuristic, meta)
        _page_methods(pdf, meta, per_alt, heuristic)
    return out_pdf


# ==========================================================
# CLI
# ==========================================================

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--altitudes", type=float, nargs="+", default=DEFAULT_ALTITUDES)
    p.add_argument("--lam-target", type=float, default=DEFAULT_LAM_TARGET)
    p.add_argument("--stride", type=float, default=1.0)
    p.add_argument("--std-devs", type=float, nargs=6, default=DEFAULT_STD_DEVS)
    p.add_argument("--rho-max", type=float, default=1e3)
    p.add_argument("--radii", type=float, nargs="+", default=DEFAULT_RADII,
                   help="desired 3-sigma pixel radii for the design heuristic")
    p.add_argument("--z", type=float, default=0.0)
    p.add_argument("--out", default=os.path.join(os.path.expanduser("~"),
                                                 "catch", "design_bound_lp.pdf"))
    p.add_argument("--json", default=None)
    p.add_argument("--pdf", action="store_true",
                   help="render the PDF report after the study "
                        "(default: study + JSON only)")
    p.add_argument("--render", metavar="STUDY_JSON", default=None,
                   help="skip the study: load a saved study JSON and render "
                        "its PDF to --out")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    camera = Camera(**DEFAULT_CAMERA)

    if args.self_test:
        # the harness is scripts/test_design_bound_lp.py; --self-test is the
        # documented alias for running it.
        import pytest
        here = os.path.dirname(os.path.abspath(__file__))
        return pytest.main(["-q", os.path.join(here, "test_design_bound_lp.py")])

    if args.render:
        if not os.path.isfile(args.render):
            print(f"[ERR] no study JSON at {args.render}")
            return 2
        study = Study.load_json(args.render)
        print(f"[CFG] rebuilding FOV fields at "
              f"h={study.meta['reference_altitude']:g} m, "
              f"stride={study.meta['stride']:g}")
        print(f"[OUT] report  -> {build_report(args.out, study)}")
        return 0

    s_base = np.diag(CovarianceModel.realistic(args.std_devs)).copy()
    study = run_study(camera, args.altitudes, args.lam_target, s_base,
                      z=args.z, stride=args.stride, rho_max=args.rho_max,
                      radii=args.radii, std_devs=args.std_devs)

    out_json = study.save_json(args.json or
                               os.path.splitext(args.out)[0] + ".json")
    print(f"[OUT] records -> {out_json}")
    if args.pdf:
        print(f"[OUT] report  -> {build_report(args.out, study)}")
    else:
        print("[OUT] report  -> skipped (pass --pdf to render, "
              f"or --render {out_json})")
    print(f"[OUT] total runtime {study.meta['runtime_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
