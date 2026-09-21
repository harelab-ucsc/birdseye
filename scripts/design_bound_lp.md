# A certified design bound on pose noise, and the budget it implies

Companion to `scripts/design_bound_lp.py`. Artifacts: `~/catch/design_bound_lp.pdf` (10 pages) and
`~/catch/design_bound_lp.json`.

## 1. What this note is

This is the working record behind Sections `sec:bound` and `sec:results_bound` of the BirdsEye manuscript. The object
of study is the sufficient condition that converts an annotation tolerance into a convex set of admissible pose-noise
budgets, and the point-selection rule that turns that set into a per-axis hardware specification. Every number below
is read from `~/catch/design_bound_lp.json` and, for Section 8, from `~/catch/design_bound_ablation.json`; the
machinery of Sections 7 and 8 is newer than the manuscript and seeds a follow-on paper.

One restriction keeps the note short. We study only **certified** constraints: a functional `f` qualifies only if
`f(s) <= lambda_u,target` implies `lambda_u,max <= lambda_u,target` for every `s >= 0`. Descriptive statistics of the
pixel-error field — mean, median, quantile, CVaR — are not upper bounds on `lambda_u,max`, and are excluded by the
definition rather than by preference.

## 2. Notation

|symbol|definition|units|
|---|---|---|
|`X_w`|A point on the ground plane at height `z` in world coordinates.|m|
|`X`|The imaged set: back-projection of the image rectangle onto that plane at the nominal pose, which is nadir throughout — camera rotation `R = I`, translation `t = [0, 0, h]`.|—|
|`h`|Above-ground-level altitude of the camera.|m|
|`z`|Height of the ground plane; zero throughout.|m|
|`d`|Depth `d = h - z`, camera-to-ground distance along the optical axis, constant over the frame at nadir.|m|
|`xi`|Pose perturbation, a 6-vector `[dx, dy, dz, d_roll, d_pitch, d_yaw]` in the `se(3)` convention of `pose_uncertainty_prop.SE3.perturb`.|m, rad|
|`J_xi`|Jacobian of the pixel projection of `X_w` with respect to `xi` at the nominal pose; 2×6, computed as `J_pixel @ [I \| -skew(xc)]`.|px/m, px/rad|
|`j_i`|The `i`-th column of `J_xi`: pixel displacement from a unit perturbation of axis `i`.|px/m, px/rad|
|`g_i`|Gain field, the squared pixel sensitivity to axis `i`, `g_i = \|\|j_i\|\|_2^2`.|px²/m², px²/rad²|
|`g`|Gain vector at the imaged point, `g = [g_1 ... g_6]`.|px²/m², px²/rad²|
|`g*`|The gain vector on the binding row, i.e. at the worst-imaged point of the frame.|px²/m², px²/rad²|
|`Sigma_0`|Pose covariance, assumed diagonal, `Sigma_0 = diag(s)`.|m², rad²|
|`s`|Design vector `s = [sigma_1^2 ... sigma_6^2]`, the per-axis pose variances; the unknown of every program here.|m², rad²|
|`s_base`|Deployed budget, `[1e-4, 1e-4, 3.6e-3, 4.873879e-7, 4.873879e-7, 5.148034e-6]`, i.e. datasheet 1σ of `[10, 10, 60] mm` and `[0.04, 0.04, 0.13] deg`.|m², rad²|
|`sigma_i`|Per-axis standard deviation `sqrt(s_i)`, the form in which a tolerance is specified.|mm, deg|
|`rho`|Normalized design `rho = s / s_base`, dimensionless multiples of the deployed variance.|—|
|`rho*`|The maximizer of a stated objective over the feasible set, in normalized coordinates.|—|
|`caps`|Artificial box on the design, `rho_max = 1e3` in normalized coordinates, imposed to keep the program bounded. An axis that hits it is flagged `at_cap`, and a cap-limited row is not a tolerance.|m², rad²|
|`Sigma_u`|Propagated pixel covariance `Sigma_u = J_xi diag(s) J_xi^T`, a 2×2 PSD matrix.|px²|
|`lambda_max`|Larger eigenvalue of `Sigma_u`: variance along the worst pixel direction.|px²|
|`tr`|Trace of `Sigma_u`; by the identity of Section 3, the inner product of `s` and `g`.|px²|
|`w`|Unit pixel direction; at the argsup pixel, the top eigenvector of `Sigma_u`.|—|
|`lambda_u,max`|The quantity being bounded: the supremum of `lambda_max` over `X`.|px²|
|`lambda_u,target`|Annotation tolerance, `(r/3)^2`; default 100.|px²|
|`r`|Desired pixel fidelity: 3σ positional radius of the projected point.|px|
|`f`|A certified constraint functional; the three instances are tabulated in Section 4.|px²|
|`a`|A cut: the subgradient of `f` at the query point, defining the half-space appended to the polyhedron. By homogeneity it is exact there, `a . s = f(s)`.|px²/m², px²/rad²|
|`F`|The feasible set: the admissible budgets cut out by a constraint and the box.|m², rad²|
|`kappa*`|Admissible uniform scaling, `lambda_u,target / f(s_base)`; above 1 the deployed budget has slack.|—|
|`gap_ratio`|`sup_X tr` divided by `sup_X lambda_max` at fixed `s`: how much the trace bound over-counts.|—|
|`conservatism`|`kappa*` of `exact_lmax` divided by `kappa*` of the row: design budget forfeited relative to the exact constraint.|—|
|`sup_X`|Supremum over `X`, evaluated as a maximum over every sampled pixel, edges inclusive.|—|
|`N`|Number of sampled points, `(w/stride + 1)(h/stride + 1)`; 2 307 121 at stride 1.|—|
|`budget_share`|Axis `i`'s share of the constraint value on the binding row, `g*_i s_i / f(s)`.|—|
|`violation_frac`|Fraction of sampled pixels with `lambda_max` above `lambda_u,target (1 + 1e-9)`; zero for every certified design, reported as an audit.|—|
|`coords`|Ablation axis: coordinates handed to the solver, normalized or raw. Same feasible set, different conditioning.|—|
|`weights`|Ablation axis: units the objective is stated in — `rho`, raw `s`, or `s_deg` (raw with rotations relabelled to degrees).|—|
|`dev_c`|Maximum relative deviation of a solution from the normalized-coordinate cell at the same `weights`.|—|
|`dev_w`|Maximum relative deviation from the reference cell, normalized in both axes.|—|
|`cond_A_x`|2-norm condition number of the final cut matrix in solve coordinates.|—|
|`col_norm_spread`|Ratio of largest to smallest column norm of that matrix.|—|
|`kkt_residual`|Maximum violation of stationarity at the returned point.|—|

**Manuscript convention.** `lambda_u,target = (r/3)^2`, so the default of 100 px² is `r` = 30 px, a 1σ pixel error of
10 px. That is the deployed soft-label parameter of the annotation pipeline — Gaussian humps drawn to a 3σ radius
with σ = 10 px — so the budget reported here honours the label radius actually used in training.

## 3. From pose uncertainty to a linear design constraint

Appendix `sec:methods_sensitivity` derives the error model: first-order propagation of an `se(3)` pose perturbation
`xi` through the projective function gives pixel error distributed as `N(0, Sigma_u)`, with `J_xi` as given there, so
the design question is a statement about `Sigma_u` over the frame; the validity regime of the linearization is
inherited from that appendix and from `sec:results_sensitivity`. The deployed bound in
`pose_uncertainty_prop.design_bound` takes the submultiplicative step

```
lambda_u,max  <=  sup_X ||J_xi||_2^2 * lambda_max(Sigma_0),
```

which pairs the worst gain with an unrelated worst variance. At `h` = 10 m the worst gain is rotational at
`2.1056e7 px²/rad²`, while `lambda_max(Sigma_0) = 3.6e-3 m²` is the altitude variance, whose gain never exceeds
`1.2936e4 px²/m²`. The product gives 75 801.84 px² against a true `lambda_u,max` of 74.92 px², 1012× conservative, so
at `lambda_u,target` = 100 px² it rejects a platform that conforms with headroom; being a bound on one scalar, it
also cannot say which axis to spend on. It is not studied further, appearing only as the baseline row of Section 4.

The replacement is the trace identity over the columns `j_i`, exactly linear in `s`, with the eigenvalue sandwich for
2×2 PSD matrices [10], asserted pointwise over all sampled pixels at a measured worst ratio of 1.999969:

```
tr(J_xi Sigma_0 J_xi^T) = sum_i sigma_i^2 ||j_i(X_w)||_2^2 = s . g(X_w),
lambda_max(Sigma_u)  <=  tr(Sigma_u)  <=  2 lambda_max(Sigma_u).
```

The sufficient condition for the tolerance to hold everywhere in frame is then

```
sup_{X_w in X}  s . g(X_w)  <=  lambda_u,target,
```

a half-space in `s`: the admissible budgets form a convex polyhedron whose bounding hyperplane is `g*`, the gain
vector `g` at the worst-imaged point `X_w`. At nadir `sup_X` is attained at the frame corner (1920, 1200), so
edge-inclusive sampling is mandatory. Formally the condition is semi-infinite [3], one row per point of `X`, which a
cutting-plane exchange handles in general (Section 7) and which here collapses to a single row. Positive homogeneity,
`f(kappa s) = kappa f(s)`, makes the largest uniform scaling of an existing suite closed-form,
`kappa* = lambda_u,target / f(s_base)`, measured exact to 1.4e-16.

## 4. The two operative constraints, and the price of the cheap one

Values at `h` = 10 m, `lambda_u,target` = 100 px², stride 1.

|key|`f(s)`|cut `a` at the query point|`f(s_base)` [px²]|`kappa*`|`conservatism`|role|
|---|---|---|---|---|---|---|
|`spectral`|`sup_X \|\|J_xi\|\|_2^2 * max_i s_i`|`sup_X \|\|J_xi\|\|_2^2` on the argmax axis|75 801.84|1.319e-3|1012×|baseline only: the deployed bound of Section 3|
|`sup_trace`|`max_n g_n . s`|`g*`|108.61|0.9207|1.450×|the operative condition: one linear row, the whole design polyhedron|
|`exact_lmax`|`max_n lambda_max(J_n diag(s) J_n^T)`|`a_i = (w^T j_i)^2` at the argsup pixel|74.92|1.335|1.000×|the exact worst-case reference|

The trace step recovers a factor `0.9207 / 1.319e-3 = 698` of design budget over the spectral form and leaves only
the ≤ 2× trace-versus-eigenvalue slack, here 1.450×. `exact_lmax` closes that remainder: because `lambda_max` is
itself a supremum of linear functionals of `s`, namely `max_{||w|| = 1} w^T Sigma_u w`, the exact constraint is
*still* a polyhedron, cut at the argsup pixel with `w` the top eigenvector there — at the price of one
eigendecomposition per query, 125.5 ms against 9.7 ms per oracle call at stride 1.

At this operating point the residual slack is decision-relevant, and that is the finding the manuscript reports.
`kappa*` is 0.9207 under `sup_trace` — a rejection by 8 % — and 1.335 under `exact_lmax`, an acceptance with 33 %
headroom; at `h` = 20 m (2.154 against 3.794) either form answers the question. Allocate and screen with the trace
condition, where linearity in `s` is the point, and confirm a near-boundary platform against `sup_X lambda_max`.

The binding row also resolves the budget by axis: at `s_base`, `h` = 10 m,
`g* = [1.8186e5, 1.8172e5, 1.2936e4, 1.8937e7, 2.0101e7, 1.2932e6]`, so the contributions `g*_i s_i` are
`[18.19, 18.17, 46.57, 9.23, 9.80, 6.66] px²` summing to 108.61 px², i.e. `budget_share`
`[16.7, 16.7, 42.9, 8.5, 9.0, 6.1] %`. The manuscript's Table 9 reports 44.7 % for the altitude axis; that column is
hand-entered rather than produced by `numbers.json`, and the code gives 42.9 %. *Unverified against the
manuscript's intent — confirm with the manuscript owner before the follow-on paper reuses either figure.*

## 5. Choosing a point in the feasible set

The feasible set `F = { s >= 0 : sup_X s . g(X_w) <= lambda_u,target, s <= caps }` is a six-dimensional convex body
with nonempty interior, and **every point of it is a certified design**: the constraint rules out, it does not
choose. Two questions live on `F` — "does the platform I have conform?", which needs no objective but only
`f(s_base)` evaluated and compared, and "what should I specify for the next one?", which is a point-selection
problem. An axis-by-axis comparison against the selected point is therefore not a pass/fail test: the constraint
bounds the sum, and an axis may exceed its allocation while the platform conforms.

A linear objective, `max sum_i rho_i`, is maximized at a vertex of `F`: the whole budget on one axis and five axes at
zero, non-unique under ties, and moving with the normalization — yaw at 15.02× for `h` = 10 m, the y axis at 22.01×
for `h` = 20 m. That is the correct answer to that objective and an unusable specification; Section 8 shows the
instability belongs to the objective's units, not to the solver.

We therefore report the allocation maximizing `sum_i log rho_i`. It is strictly concave, so with a strictly feasible
point (Slater) the maximizer is unique and interior, and it is scale-invariant, so
`argmax sum_i log(s_i/s_base,i) = argmax sum_i log s_i`: the allocation in absolute units is a property of the
imaging geometry and `lambda_u,target` alone, not of the deployed `Sigma_0`. On a single active row, stationarity
gives `1/s_i = mu g*_i`, hence the **equal budget share** closed form

```
s_i = lambda_u,target / (6 g*_i),   equivalently   g*_i s_i = lambda_u,target / 6  for every i,
```

each axis permitted to contribute the same worst-case pixel variance. This is the D-optimality criterion of optimal
experiment design [5], [6] specialized to a diagonal covariance; the measured `budget_share` under `sup_trace` is 1/6
per axis to 1e-12, and the closed form agrees with the solver to 1.6e-12.

The third rule implemented, `kappa_scaling`, restricts to the ray `s = kappa s_base` and returns the closed-form
conformance margin `kappa*`; it answers the first question and cannot rebalance a budget. The minimization reading,
`min c^T s` under an upper bound on `f`, is degenerate at `s = 0` since every `f` is nondecreasing in `s`; the design
program is necessarily a maximization, and every reported design is active, `f(s*)/lambda = 1.000000` for all nine
constraint–objective pairs.

## 6. Admissible per-axis budgets

Configuration: constraint `exact_lmax`, allocation `geomean`, nadir pose, ground plane `z` = 0, `caps` at 1e3 in
normalized coordinates, stride 1. No row is cap-limited.

|`h` [m]|`r` [px]|`lambda_u,target` [px²]|x [mm]|y [mm]|z [mm]|roll [deg]|pitch [deg]|yaw [deg]|
|---|---|---|---|---|---|---|---|---|
|10|5|2.78|2.27|2.24|8.19|0.0125|0.0124|0.0502|
|10|10|11.11|4.55|4.48|16.37|0.0251|0.0247|0.1004|
|10|20|44.44|9.09|8.96|32.75|0.0502|0.0494|0.2009|
|10|30|100.00|13.64|13.45|49.12|0.0753|0.0741|0.3013|
|10|50|277.78|22.73|22.41|81.87|0.1255|0.1236|0.5022|
|20|5|2.78|4.55|4.48|16.37|0.0125|0.0124|0.0502|
|20|10|11.11|9.09|8.96|32.75|0.0251|0.0247|0.1004|
|20|20|44.44|18.18|17.93|65.50|0.0502|0.0494|0.2009|
|20|30|100.00|27.28|26.89|98.25|0.0753|0.0741|0.3013|
|20|50|277.78|45.46|44.82|163.75|0.1255|0.1236|0.5022|

Pick the altitude and the desired 3σ radius and read the six admissible 1σ tolerances: a platform meeting all six is
certified to keep `lambda_u,max` at or below `(r/3)^2` everywhere in frame, audited at `violation_frac` zero over all
`N` samples. Two identities remove the need for further solves. In fidelity, `rho*` is linear in `lambda_u,target`
and `lambda_u,target = (r/3)^2`, so each `sigma_i` is exactly proportional to `r`. In altitude, translation
tolerances scale linearly with `h` because the translational gain fields `g_i` go as `1/d^2`, while rotation
tolerances are altitude-invariant at nadir: pointing accuracy is the hard constraint and does not ease with altitude.

### Conformance of the deployed platform

At the manuscript's operating point (`r` = 30 px, `lambda_u,target` = 100 px²) the deployed budget gives
`sup_X lambda_max(Sigma_u) = 74.92 px²` at `h` = 10 m (`kappa*` = 1.335) and 26.36 px² at `h` = 20 m (`kappa*` =
3.794), so the platform conforms at both altitudes, at a worst-case 1σ pixel error of 8.66 px and 5.13 px against the
10 px soft-label σ. The cheap bound alone does not establish this: `sup_X tr = 108.61 px²` at 10 m, so `sup_trace`
fails the platform by 8 %, and the eigenvalue condition is what certifies it.

|axis|deployed 1σ|balanced, `h` = 10 m|balanced, `h` = 20 m|`budget_share` at `s_base`, 10 m|
|---|---|---|---|---|
|x|10 mm|13.64 mm|27.28 mm|16.7 %|
|y|10 mm|13.45 mm|26.89 mm|16.7 %|
|z|60 mm|49.12 mm|98.25 mm|42.9 %|
|roll|0.040 deg|0.0753 deg|0.0753 deg|8.5 %|
|pitch|0.040 deg|0.0741 deg|0.0741 deg|9.0 %|
|yaw|0.130 deg|0.3013 deg|0.3013 deg|6.1 %|

The balanced allocation is a different point of `F`, so the table above is a shape diagnosis. The deployed budget is
altitude-heavy and attitude-light relative to balance and passes because of that asymmetry: yaw's unused share pays
for altitude's excess. The altimeter carries 42.9 % of the worst-case budget and is the only channel where tightening
buys meaningful margin at 10 m; yaw has 2.3× of σ slack; only the x and y axes are tighter than balance, by 1.35×,
within the factor the trace bound gives away. The empirical yaw-maneuver signature in `tab:error-norms` is not a
counterexample: sustained rotation violates the small-perturbation premise of the linearization and, at 5 Hz capture,
the synchronization assumption of `methods_sync`. The design bound describes the static budget.

## 7. How the constraint is solved

The cutting-plane loop [1], [2] runs per (constraint, objective) pair. Start at `caps`, the box corner, which
violates the constraint. Query the oracle for `f(s)` and its subgradient; if `f(s) <= lambda_u,target (1 + 1e-9)`,
stop, since the point is feasible for the true constraint and optimal for the outer approximation. Otherwise append
the cut, valid globally by homogeneity plus Danskin's theorem [4], and re-solve over the accumulated polyhedron; a
duplicate cut row means the cut set is exhausted, and the loop stops and flags `cycle`. On an uncertified exit,
project with homogeneity, so the returned design is feasible for its own constraint. Every shipped solution has
`converged` true and `cycle` false. The inner program is built with cvxpy [7]:

|objective|program|backend|`solver`|
|---|---|---|---|
|`vertex_lp_norm`|`max sum_i s_i/w_i` over `A x <= lambda, 0 <= x <= caps`|cvxpy → SCIPY/HiGHS simplex [8]|`"SCIPY"`|
|`geomean`|`max sum_i log(s_i/w_i)`, exponential cone|cvxpy → Clarabel [9], tolerances `1e-12`|`"CLARABEL"`|
|`kappa_scaling`|`max kappa` on the ray `s = kappa s_base`|closed form, `min_k lambda/(a_k . s_base)`|`"closed-form"`|

There is no start heuristic, no feasibility shrink and no fallback path: the log objective's domain is carried by
the cone, and feasibility of the returned design is re-checked by the audit. Clarabel may report
`optimal_inaccurate` at that tolerance; the status is accepted and verified downstream by the golden regression.
Cuts to certify optimality at `h` = 10 m:

|constraint|`vertex_lp_norm`|`kappa_scaling`|`geomean`|
|---|---|---|---|
|`spectral`|6|1|6|
|`sup_trace`|1|1|1|
|`exact_lmax`|3|1|3|

`sup_trace` needs one cut: its single row *is* the polyhedron. `exact_lmax` needs more because each cut linearizes at
one argsup pixel and that pixel migrates as the allocation changes; at `h` = 20 m the vertex solve needs 2 and the
balanced solve 3. A vertex chases the migrating argsup pixel, while the interior point is stable under that
migration. All inner solves run in `rho`, i.e. the cut matrix is column-scaled by `s_base`; that is a semantic
choice, and Section 8 is the measurement establishing that it is not a numerical necessity.

## 8. Normalization: what it does and does not decide

`design_bound_ablation.py` sweeps two independent axes of `solve_design`. `coords` selects the coordinates handed to
the solver, normalized `rho` or raw `s`; the feasible set is identical, so this axis can only affect conditioning.
`weights` selects the units the objective is stated in: `rho`, raw `s` in m² and rad², or `s_deg`, raw units with the
rotational entries divided by `k = (180/pi)^2`. The grid is 3 constraints × 7 cells × 2 altitudes = 66 cells at
stride 4, each loop repeated five times with `perf_counter` medians; figures below are `h` = 10 m unless stated.

### Result 1: the log objective is indifferent to both axes

|constraint|`coords`|`weights`|cuts|IPM iters|inner ms|`dev_c`|`dev_w`|`cond_A_x`|`col_norm_spread`|`kkt_residual`|closed form|
|---|---|---|---|---|---|---|---|---|---|---|---|
|`spectral`|rho|rho|6|192|33.5|—|—|7.39e3|7.39e3|3.4e-16|—|
|`spectral`|rho|s|6|195|35.0|0|1.1e-12|7.39e3|7.39e3|2.4e-15|—|
|`spectral`|s|rho|6|225|34.7|2.2e-12|2.2e-12|1.00|1.00|0|—|
|`spectral`|s|s|6|221|34.5|5.3e-13|1.6e-12|1.00|1.00|0|—|
|`sup_trace`|rho|rho|1|59|6.5|—|—|1.00|7.00|2.4e-12|1.6e-12|
|`sup_trace`|rho|s|1|38|6.4|0|1.1e-7|1.00|7.00|1.3e-7|1.1e-7|
|`sup_trace`|s|rho|1|43|6.5|1.5e-12|1.5e-12|1.00|1.55e3|4.5e-14|4.1e-14|
|`sup_trace`|s|s|1|42|6.4|1.1e-7|1.9e-10|1.00|1.55e3|3.1e-10|1.9e-10|
|`exact_lmax`|rho|rho|3|135|13.7|—|—|6.83|7.01|2.0e-12|—|
|`exact_lmax`|rho|s|3|108|13.8|0|1.0e-9|6.83|7.01|1.3e-9|—|
|`exact_lmax`|s|rho|3|165|14.0|2.1e-12|2.1e-12|34.0|1.60e3|3.7e-13|—|
|`exact_lmax`|s|s|3|111|13.4|1.0e-9|1.2e-11|34.0|1.60e3|2.6e-11|—|

Every cell lands on the same design. Across all three constraints and both altitudes, changing solve coordinates
moves the optimum by at most 2.2e-12 and changing objective units by at most 1.1e-7 — both at the solver's tolerance
floor — with identical cut counts and equal budget shares to 1.8e-8. Normalization is a free choice here.

### Result 2: conditioning is backend-dependent, and normalization is not uniformly better

Raw-`s` solves do not stall: every raw cell converges, in the same number of cuts, to the same optimum, including
the retired `trust-constr` control arm. What raw coordinates cost is a property of the backend.

|backend|`coords`|`spectral` iters / ms|`sup_trace` iters / ms|`exact_lmax` cuts, iters / ms|max dev vs Clarabel|
|---|---|---|---|---|---|
|cvxpy/Clarabel|rho|192 / 33.5|59 / 6.5|3, 135 / 13.7|—|
|cvxpy/Clarabel|s|225 / 34.7|43 / 6.5|3, 165 / 14.0|2.2e-12|
|scipy `trust-constr`|rho|237 / 161|17 / 18.4|4, 135 / 91.9|6.8e-7|
|scipy `trust-constr`|s|664 / 343|56 / 52.3|3, 194 / 104|1.6e-11|

Clarabel is indifferent to the scaling; `trust-constr` degrades 2.8× in iterations and 2.1× in wall time. Nor is
normalization uniformly better conditioned: for `spectral` the raw cut matrix is perfectly scaled, `cond_A_x` and
`col_norm_spread` both 1.0, while the normalized one is 7.4e3. Clarabel agrees with `trust-constr` to 6.8e-7 and is
4.8–12× faster, evidence that the library migration did not move published answers.

### Result 3: the linear objective is decided by its units

|`h`|objective units|`rho*`|winning axis|
|---|---|---|---|
|10 m|`rho`|`[0, 0, 0, 0, 0, 15.021]`|yaw|
|10 m|`s` (m², rad²)|`[0, 0, 2.147, 0, 0, 0]`|z|
|10 m|`s_deg` (m², deg²)|`[0, 0, 0, 0, 0, 15.021]`|yaw|
|20 m|`rho`|`[0, 22.012, 0, 0, 0, 0]`|y|
|20 m|`s` (m², rad²)|`[0, 0, 8.589, 0, 0, 0]`|z|
|20 m|`s_deg` (m², deg²)|`[0, 0, 0, 0, 0, 15.021]`|yaw|

A radians-to-degrees relabelling changes nothing physical — same constraint, same feasible set, same solver — and
changes the "optimal" design; at `h` = 20 m the three conventions nominate three different axes. That is why the
shipped linear objective is stated in `rho`. Nor is the effect confined to one-cut constraints: under `exact_lmax`
the raw-unit program needs 13 cuts instead of 3 and answers `[0, 0, 2.147, 0, 0, 14.970]` against
`[0, 0, 0, 9.841, 9.833, 1.929]` normalized. `spectral` is the exception that identifies the mechanism: its cuts are
one-hot, so its feasible set is an axis-aligned box and every axis reaches its own bound whatever the units.

At stride 1 (`~/catch/design_bound_ablation_stride1.json`, `h` = 10 m, three repeats) the headline cell reproduces:
`sup_trace` with `geomean` gives `rho* = [0.9165, 0.9171, 0.3579, 1.8058, 1.7012, 2.5035]` in one cut, equal shares,
closed-form error 1.6e-12. The ablation is a parameter sweep over `solve_design(..., coords=, weights=)`, not a
separate code path, which is what makes the claim auditable.

## 9. Artifacts and reproduction

```bash
cd /home/mwmaster/git_root/payload_camera_ws/src/birdseye/scripts
mkdir -p ~/catch
python3 -m pytest test_design_bound_lp.py -q          # == design_bound_lp.py --self-test
python3 -W error::RuntimeWarning design_bound_lp.py --lam-target 100 --altitudes 10 20 --pdf
python3 design_bound_ablation.py --stride 4 --altitudes 10 20 --repeats 5
```

`--self-test` delegates to the pytest module. `-W error::RuntimeWarning` is deliberate and has caught real defects: a
divide-by-zero in a closed form, and the retired `trust-constr` backend evaluating the log objective outside its
domain. The matplotlib `Axes3D` warning and Clarabel's inaccuracy warning are expected and are not promoted.

|page|what it shows|
|---|---|
|1|the Section 2 notation, this run's numeric context, and the `lambda_u,target` convention|
|2|FOV fields at `h` = 10 m: `tr`, exact `lambda_max`, their ratio, and the violation mask at `s_base`|
|3|per-axis gain maps, `g_i` in log10 over the frame, with per-axis extrema|
|4|distributions of `tr` and `lambda_max` at `s_base`, with sup and descriptive-only markers|
|5|the Section 4 constraint comparison with the `kappa*` audit|
|6|the point-selection argument of Section 5 and the measured vertex, balanced and ray allocations|
|7|`rho*` bars per axis for `vertex_lp_norm` and `geomean`, hatched where cap-limited|
|8|altitude sweep: `kappa*` against altitude, and `gap_ratio` and the suprema against altitude|
|9|the Section 6 budget table, plus log-log `sigma_i` against `r`|
|10|camera, pose, stride, `s_base`, solver settings, cut counts, certification statement, references|

`scripts/test_design_bound_lp.py` holds 18 test functions over 47 parametrized cases. Ten are analytic checks: the
batch Jacobian against `ProjectionModel.analytic_jacobian`, the eigenvalue sandwich, oracle homogeneity, cut
exactness, `kappa*` exactness, the single-cut closed form, the ordering of `exact_lmax` against `sup_trace`,
per-design feasibility, zero FOV violation, and the fidelity-scaling identity. The rest pin the study against
`scripts/testdata/design_bound_lp_golden_stride4.json` at rtol 1e-6 and assert the invariances of Section 8,
including the one deliberate non-invariance — that the linear objective's vertex moves with the objective's units.
The golden record was generated by the pre-cvxpy code and is never regenerated; two departures are documented, the
retired backend stopping 7.5e-7 short of the `spectral`/`geomean` optimum at `h` = 10 m and needing one extra cut
at `h` = 20 m. `budget_share` is not pinned: at the optimum its row is degenerate, since `spectral` ties all six
entries of `s` and `exact_lmax` has `lambda_max` equal to the smaller eigenvalue to 1e-12, leaving `w` arbitrary.

Environment: Python 3.10.12, numpy 2.2.6, scipy 1.15.2, matplotlib 3.10.1, cvxpy 1.7.5, clarabel 0.11.1, pytest
6.2.5. Timings: pytest 4.0 s, full report 13.1 s, ablation 27 s at stride 4 and 41 s at stride 1.

## 10. Scope limits

- **Nadir pose only.** The sampler assumes the nominal pose, so `d` is constant over the frame. Off nadir the
  rotation must be applied per pixel and `X` becomes the ray–surface intersection set, with per-pixel depth spread
  reaching 15× by 81° of pitch; only the sampler and the definition of `X` change, not the constraint algebra.
- **Diagonal `Sigma_0`.** Cross-axis correlations are not modelled; a full `Sigma_0` makes the constraint linear in
  the 21 free entries of a PSD matrix, i.e. an SDP rather than an LP.
- **Caps are flagged, not applied.** `caps` keeps the program bounded, any axis reaching it is reported, and no
  shipped row is cap-limited.
- **The supremum is a grid maximum.** With stride 1 and edges inclusive the argsup pixel is sampled exactly, so the
  discretization is exact for this geometry; on a tilted or distorted frame it could fall between samples.
- **No Monte Carlo.** Jacobians, eigenvalues and solvers are deterministic; the only randomness is the 20 seeded
  design vectors in `test_cut_exactness`.

## References

1. J. E. Kelley, "The Cutting-Plane Method for Solving Convex Programs", *SIAM Journal* 8(4), 703–712, 1960.
   Outer-approximate a convex feasible set by subgradient half-spaces and re-solve.
2. S. Boyd and L. Vandenberghe, "Localization and Cutting-Plane Methods", Stanford EE364b lecture notes,
   <https://web.stanford.edu/class/ee364b/lectures/localization_methods_notes.pdf>.
3. R. Hettich and K. O. Kortanek, "Semi-infinite programming: theory, methods, and applications", *SIAM Review*
   35(3), 380–429, 1993, doi:10.1137/1035089. The cutting-plane loop is the exchange method for such programs.
4. J. M. Danskin, *The Theory of Max-Min and its Application to Weapons Allocation Problems*, Springer, 1967. The
   maximizing row of a pointwise maximum is a subgradient, so each cut is valid.
5. S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004. Vertex optima of linear
   programs, and the experiment-design section whose D-optimal `log det` criterion is the direct analogue of the
   balanced allocation. *Section numbers unverified — confirm the experiment-design section number before citing it.*
6. F. Pukelsheim, *Optimal Design of Experiments*, SIAM Classics in Applied Mathematics 50, 2006. The D-optimality
   lineage of the balanced objective.
7. S. Diamond and S. Boyd, "CVXPY: A Python-Embedded Modeling Language for Convex Optimization", *Journal of Machine
   Learning Research* 17(83), 1–5, 2016. The modelling layer both solves use.
8. Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex method", *Mathematical Programming
   Computation* 10(1), 119–142, 2018. HiGHS, the LP solver actually called.
9. P. J. Goulart and Y. Chen, "Clarabel: An interior-point solver for conic programs with quadratic objectives",
   arXiv:2405.12762, 2024. The method behind the exponential-cone solve.
10. R. A. Horn and C. R. Johnson, *Matrix Analysis*, 2nd ed., Cambridge University Press, 2012. For an `n x n` PSD
    matrix, `lambda_max <= tr <= n lambda_max`; here the matrix is 2×2.
