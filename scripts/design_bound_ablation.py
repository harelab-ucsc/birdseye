#!/usr/bin/env python3
"""
Normalization ablation for the certified design bound (scripts/design_bound_lp.py).

Two axes are varied independently, for each (altitude, constraint, objective):

  coords   -- the SOLVE COORDINATES.  "rho" solves in x = s / s_base (the cut
              matrix is column-scaled by s_base), "s" solves in x = s directly.
              The feasible set is the same set; only the conditioning of the
              program handed to the solver changes.
  weights  -- the OBJECTIVE UNITS.  "rho" states the objective in multiples of
              the deployed budget s_base, "s" in raw m^2/rad^2, "s_deg" in raw
              units with the rotational DOFs relabelled to degrees.  For the
              log objective this is a subtracted constant (invariant); for the
              linear objective it re-points the LP vertex.

Backends: cvxpy/HiGHS (linear), cvxpy/Clarabel (log), and -- as the migration's
control arm, living only in this file -- scipy trust-constr on the log
objective, the solver the production module used before the cvxpy cutover.

Standalone CLI; imports design_bound_lp, writes nothing into the study JSON and
renders no PDF pages.
"""

import argparse
import json
import os
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize, nnls

import design_bound_lp as dbl

DEG2 = (180.0 / np.pi) ** 2      # 1 rad^2 = DEG2 deg^2

WEIGHT_VECTORS = {
    # objective = sum_i s_i / w_i, so w_i is the unit the DOF is counted in.
    "rho": lambda s_base: np.asarray(s_base, dtype=float),
    "s": lambda s_base: np.ones(6),
    # the same raw-unit objective with the rotations relabelled to degrees:
    # a rotational variance of s rad^2 is s * DEG2 deg^2, i.e. w = 1 / DEG2.
    "s_deg": lambda s_base: np.array([1.0, 1.0, 1.0, 1.0 / DEG2, 1.0 / DEG2,
                                      1.0 / DEG2]),
}


# ==========================================================
# Backends
# ==========================================================

def _cvxpy_solve(objective, A_x, b, x_caps, coef):
    """Production inner solve (design_bound_lp._inner_solve), timed."""
    t0 = perf_counter()
    x, info = dbl._inner_solve(objective, A_x, b, x_caps, coef)
    info = dict(info)
    info["wall_time_s"] = perf_counter() - t0
    return x, info


def _scipy_geomean(A_x, b, x_caps, coef=None):
    """Control arm: max sum_i log x_i by scipy trust-constr.

    coef only shifts the objective by a constant (sum log coef_i) and is
    ignored; the reference arm is only ever run with weights == "rho".
    x0 is the analytic single-row optimum on the tightest cut, shrunk where a
    row is still violated.
    """
    A_x = np.atleast_2d(np.asarray(A_x, dtype=float))
    b = np.asarray(b, dtype=float)
    x_caps = np.asarray(x_caps, dtype=float)
    tiny = 1e-300
    k = int(np.argmin(b - A_x @ x_caps))
    x0 = np.minimum(b[k] / (6.0 * np.maximum(A_x[k], tiny)), x_caps)
    row = A_x @ x0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.min(np.where(row > 0, b / np.maximum(row, tiny), np.inf))
    if np.isfinite(t) and t < 1.0:
        x0 = x0 * 0.99 * t
    x0 = np.maximum(x0, 1e-300)

    t0 = perf_counter()
    res = minimize(lambda x: -np.sum(np.log(x)), x0,
                   jac=lambda x: -1.0 / x,
                   hess=lambda x: np.diag(1.0 / x ** 2),
                   method="trust-constr",
                   constraints=[LinearConstraint(A_x, -np.inf, b)],
                   bounds=Bounds(np.full(6, 1e-12), x_caps, keep_feasible=True),
                   options={"maxiter": 3000, "gtol": 1e-12, "xtol": 1e-14,
                            "verbose": 0})
    wall = perf_counter() - t0
    info = {"solver": "trust-constr", "status": str(res.status),
            "inner_iters": int(res.niter),
            "inner_time_s": float(res.execution_time),
            "wall_time_s": wall}
    return np.asarray(res.x, dtype=float), info


# ==========================================================
# Cut loop (backend-parametrized copy of design_bound_lp.solve_design)
# ==========================================================

def _cut_loop(oracle, lam, fields, s_base, caps, coords, w, solve_fn,
              max_cuts=200, tol=1e-9):
    """Cutting-plane loop with a pluggable inner solver.

    Structurally identical to design_bound_lp.solve_design; it exists so the
    production module carries no backend switch.  Returns the final design and
    the per-iteration solver records plus the final cut matrix in solve
    coordinates.
    """
    D = np.asarray(s_base, dtype=float) if coords == "rho" else np.ones(6)
    coef = D / w
    x_caps = caps / D

    A, seen, infos = [], [], []
    s = caps.copy()
    converged, cycle = False, False
    oracle_time = 0.0
    t_loop = perf_counter()
    for iters in range(1, max_cuts + 1):
        t0 = perf_counter()
        val, cut = oracle(fields, s)
        oracle_time += perf_counter() - t0
        if val <= lam * (1.0 + tol):
            converged = True
            break
        key = cut / max(float(np.max(np.abs(cut))), 1e-300)
        if any(np.allclose(key, k, rtol=1e-12, atol=1e-14) for k in seen):
            cycle = True
            break
        seen.append(key)
        A.append(cut)
        A_x = np.asarray(A, dtype=float) * D
        b = np.full(len(A), lam)
        x, info = solve_fn(A_x, b, x_caps, coef)
        infos.append(info)
        s = x * D
    loop_time = perf_counter() - t_loop
    A_x = np.asarray(A, dtype=float).reshape(-1, 6) * D
    return {"s": s, "x": s / D, "D": D, "A_x": A_x, "infos": infos,
            "n_cuts": int(A_x.shape[0]), "converged": bool(converged),
            "cycle": bool(cycle), "oracle_time_s": oracle_time,
            "loop_time_s": loop_time, "iterations": iters}


# ==========================================================
# Per-cell diagnostics
# ==========================================================

def _kkt_residual(A_x, b, x, x_caps):
    """Stationarity residual of max sum log x_i s.t. A_x x <= b.

    With multiplier mu >= 0 on the cut block, stationarity is
    1/x_i = (A_x^T mu)_i.  mu is recovered by nonnegative least squares on the
    active rows (the backends' dual vectors are not uniformly available).
    Returns NaN when a DOF sits on its box cap: the box multiplier is then
    nonzero and the cut block alone cannot balance the gradient.
    """
    if not A_x.size:
        return float("nan")
    if np.any(x >= x_caps * (1 - 1e-9)):
        return float("nan")
    act = (b - A_x @ x) <= 1e-6 * np.maximum(np.abs(b), 1.0)
    if not np.any(act):
        return float("nan")
    g = 1.0 / x
    mu, _ = nnls(A_x[act].T, g)
    resid = np.abs(g - A_x[act].T @ mu) / g
    return float(np.max(resid))


def _closed_form_rel_err(fields, s, lam):
    """sup_trace + geomean only: s_i vs lam / (6 a_i) on the argsup gain row."""
    _, a = dbl._sup_trace(fields, s)
    ref = lam / (6.0 * a)
    return float(np.max(np.abs(s - ref) / ref))


def _col_norm_spread(A_x):
    if not A_x.size:
        return float("nan")
    n = np.linalg.norm(A_x, axis=0)
    lo = float(np.min(n))
    if lo == 0.0:
        return float("inf")
    return float(np.max(n) / lo)


# ==========================================================
# Cell grid
# ==========================================================

def _cells():
    """(objective, backend, coords, weights) grid, per constraint/altitude."""
    out = []
    for coords in ("rho", "s"):
        for weights in ("rho", "s"):
            out.append(("vertex_lp_norm", "cvxpy", coords, weights))
            out.append(("geomean", "cvxpy", coords, weights))
    for coords in ("rho", "s"):
        out.append(("geomean", "trust-constr", coords, "rho"))
    # unit-relabelling probe: the raw-unit LP answer under degrees
    out.append(("vertex_lp_norm", "cvxpy", "rho", "s_deg"))
    return out


def run_cell(fields, oracle, objective, backend, coords, weights, s_base,
             caps, lam, repeats):
    w = WEIGHT_VECTORS[weights](s_base)
    solve_fn = (_scipy_geomean if backend == "trust-constr"
                else (lambda A_x, b, x_caps, coef:
                      _cvxpy_solve(objective, A_x, b, x_caps, coef)))

    runs = [_cut_loop(oracle, lam, fields, s_base, caps, coords, w, solve_fn)
            for _ in range(repeats)]
    r = runs[-1]
    inner_times = [sum(i["wall_time_s"] for i in run["infos"]) for run in runs]
    s = r["s"]
    rho = s / s_base
    x, A_x = r["x"], r["A_x"]
    aud = dbl.audit(fields, s, lam, oracle=oracle, caps=caps)

    cell = {
        "altitude": float(fields.altitude),
        "constraint": oracle.key,
        "objective": objective.key,
        "backend": backend,
        "coords": coords,
        "weights": weights,
        "s": s.tolist(),
        "rho": rho.tolist(),
        "budget_share": aud["budget_share"],
        # LP vertices have zero components; -inf is the honest value.
        "sum_log_s": _sum_log(s),
        "sum_log_rho": _sum_log(rho),
        "sum_rho": float(np.sum(rho)),
        "sum_s": float(np.sum(s)),
        "f_over_lam": float(aud["f_value"] / lam),
        "sup_lmax_over_lam": float(aud["sup_lmax"] / lam),
        "violation_frac": float(aud["violation_frac"]),
        "n_cuts": r["n_cuts"],
        "converged": r["converged"],
        "inner_iters_total": int(sum(i["inner_iters"] for i in r["infos"])),
        "inner_time_s_median": float(np.median(inner_times)),
        "oracle_time_s": float(np.median([run["oracle_time_s"] for run in runs])),
        "loop_time_s": float(np.median([run["loop_time_s"] for run in runs])),
        "cond_A_x": float(np.linalg.cond(A_x)) if A_x.size else float("nan"),
        "col_norm_spread": _col_norm_spread(A_x),
        "kkt_residual": (_kkt_residual(A_x, np.full(r["n_cuts"], lam), x,
                                       caps / r["D"])
                         if objective.kind == "log" else float("nan")),
        "closed_form_rel_err": (_closed_form_rel_err(fields, s, lam)
                                if (oracle.key == "sup_trace"
                                    and objective.kind == "log")
                                else float("nan")),
        "argmax_dof": int(np.argmax(rho)),
        "status": str(r["infos"][-1]["status"]) if r["infos"] else "none",
        "solver": str(r["infos"][-1]["solver"]) if r["infos"] else "none",
    }
    return cell


def run_ablation(camera, altitudes, stride, lam, std_devs, rho_max, repeats,
                 z=0.0, log=print):
    s_base = np.diag(dbl.CovarianceModel.realistic(std_devs)).copy()
    caps = float(rho_max) * s_base
    constraints = dbl.build_constraints()
    cells = []
    for h in altitudes:
        F = dbl.FovFields(camera, h, z=z, stride=stride)
        log(f"[ALT] h={h:g} m  N={F.N}  ({F.build_time:.1f} s build)")
        for ckey, oracle in constraints.items():
            for okey, backend, coords, weights in _cells():
                cell = run_cell(F, oracle, dbl.OBJECTIVES[okey], backend,
                                coords, weights, s_base, caps, lam, repeats)
                cells.append(cell)
                log("[CELL] " + _row(cell))
    _attach_ref_deviation(cells)
    return {"meta": {"altitudes": [float(h) for h in altitudes],
                     "stride": float(stride), "lam_target": float(lam),
                     "std_devs": list(map(float, std_devs)),
                     "rho_max": float(rho_max), "repeats": int(repeats),
                     "z": float(z),
                     "s_base": s_base.tolist(),
                     "camera": {k: float(v) for k, v in
                                dict(w=camera.w, h=camera.h, fx=camera.fx,
                                     fy=camera.fy, cx=camera.cx,
                                     cy=camera.cy).items()},
                     "solvers": {"linear": "cvxpy/SCIPY-HiGHS",
                                 "log": "cvxpy/Clarabel",
                                 "reference": "scipy trust-constr"}},
            "cells": cells}


def _attach_ref_deviation(cells):
    """Two deviation measures, both against cvxpy cells:

    rel_dev_vs_ref   -- vs coords == "rho" at the SAME weights: isolates the
                        solve-coordinate axis.
    rel_dev_vs_norm  -- vs (coords, weights) == ("rho", "rho"): isolates the
                        objective-units axis (the invariance claim for the log
                        objective, the non-invariance for the linear one).
    """
    ref = {(c["altitude"], c["constraint"], c["objective"], c["weights"]): c
           for c in cells if c["coords"] == "rho" and c["backend"] == "cvxpy"}
    norm = {(c["altitude"], c["constraint"], c["objective"]): c
            for c in cells if c["coords"] == "rho" and c["weights"] == "rho"
            and c["backend"] == "cvxpy"}
    for c in cells:
        c["rel_dev_vs_ref"] = _rel_dev(
            c, ref.get((c["altitude"], c["constraint"], c["objective"],
                        c["weights"])))
        c["rel_dev_vs_norm"] = _rel_dev(
            c, norm.get((c["altitude"], c["constraint"], c["objective"])))


def _rel_dev(cell, ref):
    if ref is None:
        return float("nan")
    s, sr = np.asarray(cell["s"]), np.asarray(ref["s"])
    if not np.any(sr > 0):
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(s - sr) / np.where(sr > 0, sr, np.nan)
    # a DOF the reference sets to exactly zero: any nonzero value there is a
    # different point, so report inf rather than dropping the column.
    if np.any((sr <= 0) & (s > 0)):
        return float("inf")
    return float(np.nanmax(d))


def _sum_log(v):
    with np.errstate(divide="ignore"):
        return float(np.sum(np.log(v)))


# ==========================================================
# Console table
# ==========================================================

HEADER = (f"{'h':>4} {'constraint':<11} {'obj':<14} {'backend':<12} "
          f"{'coords':<6} {'w':<5} {'amax':>4} {'cuts':>4} {'iters':>5} "
          f"{'inner_ms':>9} {'loop_s':>7} {'dev_c':>9} {'dev_w':>9} "
          f"{'cond':>10} {'spread':>10} {'kkt':>9} {'cf_err':>9}")


def _row(c):
    return (f"{c['altitude']:>4.0f} {c['constraint']:<11} {c['objective']:<14} "
            f"{c['backend']:<12} {c['coords']:<6} {c['weights']:<5} "
            f"{c['argmax_dof']:>4} {c['n_cuts']:>4} {c['inner_iters_total']:>5} "
            f"{1e3 * c['inner_time_s_median']:>9.3f} {c['loop_time_s']:>7.3f} "
            f"{c.get('rel_dev_vs_ref', float('nan')):>9.2e} "
            f"{c.get('rel_dev_vs_norm', float('nan')):>9.2e} "
            f"{c['cond_A_x']:>10.3e} {c['col_norm_spread']:>10.3e} "
            f"{c['kkt_residual']:>9.2e} {c['closed_form_rel_err']:>9.2e}")


def print_table(cells, log=print):
    log(HEADER)
    for c in cells:
        log(_row(c))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--altitudes", type=float, nargs="+", default=[10.0, 20.0])
    p.add_argument("--stride", type=float, default=4.0)
    p.add_argument("--lam-target", type=float, default=dbl.DEFAULT_LAM_TARGET)
    p.add_argument("--std-devs", type=float, nargs=6,
                   default=dbl.DEFAULT_STD_DEVS)
    p.add_argument("--rho-max", type=float, default=1e3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--z", type=float, default=0.0)
    p.add_argument("--json", default=os.path.expanduser(
        "~/catch/design_bound_ablation.json"))
    args = p.parse_args(argv)

    camera = dbl.Camera(**dbl.DEFAULT_CAMERA)
    t0 = perf_counter()
    out = run_ablation(camera, args.altitudes, args.stride, args.lam_target,
                       args.std_devs, args.rho_max, args.repeats, z=args.z)
    print()
    print_table(out["cells"])
    path = os.path.abspath(args.json)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[OUT] cells -> {path}  ({len(out['cells'])} cells, "
          f"{perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
