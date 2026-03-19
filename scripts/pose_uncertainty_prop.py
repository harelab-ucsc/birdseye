import os
import time
import pickle
import numpy as np
from numpy.linalg import norm, eigh
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.path import Path

np.random.seed(42)
plt.tight_layout()

res = (1920, 1200)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
})


def generate_test_points(
    max_distance=10.0,
    num_points=5,
    z=0.0,
):
    """
    Generate structured world test points along +X, +Y, and X=Y diagonal.

    Parameters
    ----------
    max_distance : float
        Maximum distance from the origin.
    num_points : int
        Number of points along each ray.
    z : float
        Z coordinate of points (e.g., ground plane).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing point arrays for:
            "x_axis"
            "y_axis"
            "diag"

        The arrays "x_axis" and "y_axis" have shape (N,3).
        The arrays "x_axis" and "y_axis" have shape (N+1,3).
    """

    distances = np.linspace(max_distance/num_points,
                            max_distance,
                            num_points)

    x_axis = np.stack([distances, np.zeros_like(distances), np.full_like(distances, z)], axis=1)
    y_axis = np.stack([np.zeros_like(distances), distances, np.full_like(distances, z)], axis=1)

    diag = np.stack([distances, distances, np.full_like(distances, z)], axis=1)

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "diag": diag,
    }


def compute_ground_footprint(camera, altitude, R_wc=None):
    """
    Compute the ground-plane footprint polygon of the camera view.

    Parameters
    ----------
    camera : Camera
        Must provide fx, fy, cx, cy, width, height.
    altitude : float
        Camera height above ground.
    R_wc : (3,3) ndarray, optional
        Rotation from camera frame to world frame.
        Defaults to identity (nadir camera).

    Returns
    -------
    footprint : (4,3) ndarray
        Ground intersection points of the image corners.
    """

    if R_wc is None:
        R_wc = np.eye(3)

    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy

    corners = np.array([
        [0, 0],
        [camera.w, 0],
        [camera.w, camera.h],
        [0, camera.h],
    ])

    footprint = []

    for u, v in corners:
        # normalized camera ray
        x = (u - cx) / fx
        y = (v - cy) / fy
        ray_cam = np.array([x, y, 1.0])

        # rotate into world
        ray_world = R_wc @ ray_cam

        # camera origin
        C = np.array([0, 0, altitude])

        # intersect with ground plane z=0
        t = -C[2] / ray_world[2]

        P = C + t * ray_world
        footprint.append(P)

    return np.array(footprint)


def stack_test_points(points_dict, camera, experiment, tgt_z):

    pts = [np.array([0.0,0.0,tgt_z])]
    labels = ['origin']

    for key, arr in points_dict.items():
        pts.append(arr)
        labels.extend([key]*len(arr))

    Pw = np.vstack(pts)
    labels = np.array(labels)

    # compute camera footprint
    footprint = compute_ground_footprint(camera, experiment.altitude)

    poly = Path(footprint[:, :2])

    inside = poly.contains_points(Pw[:, :2])
    print(f'[RUN]    {len(Pw[inside])} points in view.')

    return Pw[inside], labels[inside], footprint


def plot_covariance_ellipse(S, mean, ax):

    vals, vecs = eigh(S)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # guard against small negative eigenvalues
    vals = np.clip(vals, 0, None)

    major = np.sqrt(vals[0])
    minor = np.sqrt(vals[1])

    theta = np.linspace(0, 2*np.pi, 200)
    circle = np.vstack((np.cos(theta), np.sin(theta)))

    ellipse = vecs @ np.diag([major, minor]) @ circle
    ellipse = ellipse + mean.reshape(2,1)

    ellipse_line, = ax.plot(ellipse[0], ellipse[1])

    origin = mean
    axis1 = origin + major * vecs[:,0]
    axis2 = origin + minor * vecs[:,1]

    axis1_line, = ax.plot([origin[0], axis1[0]], [origin[1], axis1[1]])
    axis2_line, = ax.plot([origin[0], axis2[0]], [origin[1], axis2[1]])

    return ellipse_line, axis1_line, axis2_line


def perturbation_panel(all_results, test_points, scales):

    dof_labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    altitudes = sorted(all_results.keys())
    altitude_cmaps = {
        5: plt.cm.Purples,
        10: plt.cm.Oranges,
        20: plt.cm.Greens,
        50: plt.cm.Blues
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for dof in range(6):
        ax = axes[dof]
        for alt in altitudes:

            if alt not in altitude_cmaps:
                cmap = plt.cm.viridis
            else:
                cmap = altitude_cmaps[alt]
            colors = cmap(np.linspace(0.4, 0.8, len(test_points)))

            for idx, Pw in enumerate(test_points):
                key = tuple(Pw)
                if key not in all_results[alt]:
                    continue
                if "perturbation" not in all_results[alt][key]:
                    continue
                results = all_results[alt][key]["perturbation"][dof]
                major_pred = []
                major_emp = []

                for alpha in scales:
                    if alpha not in results:
                        continue
                    r = results[alpha]
                    major_pred.append(r["major_pred"])
                    major_emp.append(r["major_emp"])

                if len(major_pred) == 0:
                    continue

                ax.loglog(scales, major_pred, color=colors[idx], alpha=0.3)
                ax.loglog(scales, major_emp,  ':', color=colors[idx])

        ax.set_title(dof_labels[dof])
        ax.grid(True)

    axes[0].set_ylabel("1-σ Pixel Uncertainty (pixels)", fontsize=14)
    axes[3].set_ylabel("1-σ Pixel Uncertainty (pixels)", fontsize=14)
    # axes[-1].set_ylim(bottom=1)

    for ax in axes[3:]:
        ax.set_xlabel("Perturbation Scale Factor", fontsize=14)

    # fig.suptitle(
    #     "Pixel Uncertainty vs Pose Perturbation\n"
    #     "(color groups = altitude)"
    # )
    fig.savefig('/home/mwmaster/catch/perturbation.pdf', dpi=300, bbox_inches='tight')


def scaling_panel(all_results, test_points, scales, fontsize=18):

    altitudes = sorted(all_results.keys())

    altitude_cmaps = {
        5: plt.cm.Purples,
        10: plt.cm.Oranges,
        20: plt.cm.Greens,
        50: plt.cm.Blues
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for alt in altitudes:

        if alt not in altitude_cmaps:
            cmap = plt.cm.viridis
        else:
            cmap = altitude_cmaps[alt]

        colors = cmap(np.linspace(0.4, 0.8, len(test_points)))

        for idx, Pw in enumerate(test_points):

            key = tuple(Pw)

            if key not in all_results[alt]:
                continue

            if "scaling" not in all_results[alt][key]:
                continue

            color = colors[idx]

            results_pw = all_results[alt][key]["scaling"]

            major_pred = []
            major_emp  = []
            aniso_pred = []
            aniso_emp  = []
            kl_vals    = []

            for alpha in scales:

                if alpha not in results_pw:
                    continue

                r = results_pw[alpha]

                major_pred.append(r["major_pred"])
                major_emp.append(r["major_emp"])

                aniso_pred.append(r["anisotropy_pred"])
                aniso_emp.append(r["anisotropy_emp"])

                kl_vals.append(r["kl_divergence"])

            if len(major_pred) == 0:
                continue

            # ---- Magnitude ----
            axes[0].loglog(scales, major_pred, color=color, alpha=0.3)
            axes[0].loglog(scales, major_emp, ':', color=color)

            # ---- Shape ----
            axes[1].loglog(scales, aniso_pred, color=color, alpha=0.3)
            axes[1].loglog(scales, aniso_emp, ':', color=color)

            # ---- KL ----
            axes[2].loglog(scales, kl_vals, color=color)

    axes[0].set_ylabel("Pixel Major Axis", fontsize=fontsize)
    # axes[0].set_title("Pixel Uncertainty Magnitude")

    axes[1].set_ylabel("Anisotropy", fontsize=fontsize)
    # axes[1].set_title("Ellipse Shape Behavior")

    axes[2].hlines(1e-1, 0.5e-1, 1.5e2, 'k', linestyles='dashed')
    axes[2].hlines(1e-2, 0.5e-1, 1.5e2, 'k', linestyles='dashed')
    axes[2].set_ylabel("sKLD", fontsize=fontsize)
    axes[2].set_xlabel("Covariance Scale Factor", fontsize=fontsize)
    # axes[2].set_title("Gaussian Approximation Divergence: Analytic vs Monte Carlo")

    for ax in axes:
        ax.grid(True)

    # fig.suptitle(
    #     "Scaling Study: Pixel Uncertainty Propagation\n"
    #     "(color groups = altitude)"
    # )

    # axes[2].legend()
    fig.savefig("/home/mwmaster/catch/scaling.pdf", dpi=300, bbox_inches='tight')


def ellipse_panel(all_results, test_points, alpha, camera):

    altitudes = sorted(all_results.keys())

    altitude_cmaps = {
        5: plt.cm.Purples,
        10: plt.cm.Oranges,
        20: plt.cm.Greens,
        50: plt.cm.Blues
    }

    fig, axes = plt.subplots(1, 4, figsize=(16,4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, alt in enumerate(altitudes):

        ax = axes[i]
        T = SE3.nominal_pose(alt)

        cmap = altitude_cmaps.get(alt, plt.cm.viridis)
        colors = cmap(np.linspace(0.4, 0.8, len(test_points)))

        for idx, Pw in enumerate(test_points):

            key = tuple(Pw)

            if key not in all_results[alt]:
                continue

            if alpha not in all_results[alt][key]["scaling"]:
                continue

            r = all_results[alt][key]["scaling"][alpha]

            S_pred = r["S_pred"]
            S_emp  = r["S_emp"]

            color = colors[idx]

            # ---- nominal projection ----
            Pc = T[:3,:3] @ Pw + T[:3,3]

            u = camera.fx * Pc[0] / Pc[2] + camera.cx
            v = camera.fy * Pc[1] / Pc[2] + camera.cy

            mean = np.array([u, v])

            # analytic ellipse
            ellipse, axis1, axis2 = plot_covariance_ellipse(S_pred, mean, ax)
            ellipse.set_color(color)
            axis1.set_color(color)
            axis2.set_color(color)

            # MC ellipse
            ellipse_mc, _, _ = plot_covariance_ellipse(S_emp, mean, ax)
            ellipse_mc.set_color(color)
            ellipse_mc.set_linestyle('--')

        ax.set_title(f"{alt} m altitude")
        ax.set_aspect("equal")
        ax.set_xlim(850, res[0])
        ax.set_ylim(550, res[1])
        ax.grid(True)
    # plt.suptitle(f"Pixel Uncertainty Ellipses (scale={alpha})")
    fig.savefig('/home/mwmaster/catch/ellipse.pdf', dpi=300, bbox_inches='tight')


# ==========================================================
# SE(3) Geometry
# ==========================================================

class SE3:

    @staticmethod
    def skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def hat(xi):
        rho = xi[:3]
        phi = xi[3:]
        Xi = np.zeros((4,4))
        Xi[:3,:3] = SE3.skew(phi)
        Xi[:3,3] = rho
        return Xi

    @staticmethod
    def perturb(T, xi):
        return expm(SE3.hat(xi)) @ T

    @staticmethod
    def nominal_pose(height):
        T = np.eye(4)
        T[:3,3] = [0, 0, height]
        return T


# ==========================================================
# Camera Model
# ==========================================================

class Camera:

    def __init__(self, w, h, fx, fy, cx, cy):
        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def project(self, T, Pw):
        Pc = T[:3,:3] @ Pw + T[:3,3]
        u = self.fx * Pc[0] / Pc[2] + self.cx
        v = self.fy * Pc[1] / Pc[2] + self.cy
        return np.array([u, v])


# ==========================================================
# Projection Model + Jacobian
# ==========================================================

class ProjectionModel:

    def __init__(self, camera):
        self.camera = camera

    def analytic_jacobian(self, T, Pw):
        R = T[:3,:3]
        t = T[:3,3]

        xc = R @ Pw + t
        x, y, z = xc

        J_xi = np.zeros((3,6))
        J_xi[:, :3] = np.eye(3)
        J_xi[:, 3:] = -SE3.skew(xc)

        J_pixel = np.array([
            [self.camera.fx/z, 0, -self.camera.fx*x/(z**2)],
            [0, self.camera.fy/z, -self.camera.fy*y/(z**2)]
        ])

        return J_pixel @ J_xi

    def numerical_jacobian(self, T, Pw, eps=1e-6):
        J = np.zeros((2,6))
        u0 = self.camera.project(T, Pw)

        for i in range(6):
            d = np.zeros(6)
            d[i] = eps
            T_pert = SE3.perturb(T, d)
            u1 = self.camera.project(T_pert, Pw)
            J[:,i] = (u1 - u0) / eps

        return J


# ==========================================================
# Covariance Models
# ==========================================================

class CovarianceModel:

    @staticmethod
    def realistic(std_devs, eps=1e-6):
        S = eps * np.eye(6)
        for i, std in enumerate(std_devs):
            if i < 3:
                S[i,i] = std**2
            else:
                S[i,i] = (np.pi/180*std)**2  # degree to radian conversion
        return S

    @staticmethod
    def scaled(base_cov, alpha):
        return alpha * base_cov


    @staticmethod
    def perturbed(base_cov, i, alpha):
        Sigma = base_cov.copy()
        Sigma[i,i] *= alpha
        return Sigma


# ==========================================================
# Uncertainty Propagation
# ==========================================================

class UncertaintyPropagator:

    def __init__(self, projection_model):
        self.model = projection_model

    def analytic_covariance(self, T, Pw, Sigma_xi):
        J = self.model.analytic_jacobian(T, Pw)
        return J @ Sigma_xi @ J.T

    def monte_carlo_covariance(self, T, Pw, Sigma_xi, N=20000):
        samples = []
        for _ in range(N):
            xi = np.random.multivariate_normal(np.zeros(6), Sigma_xi)
            T_pert = SE3.perturb(T, xi)
            samples.append(self.model.camera.project(T_pert, Pw))
        samples = np.array(samples)
        return np.cov(samples.T)


# ==========================================================
# Metrics
# ==========================================================

class Metrics:

    @staticmethod
    def ellipse_parameters(S):
        vals, vecs = eigh(S)
        major = np.sqrt(vals[-1])
        minor = np.sqrt(vals[0])
        anisotropy = major/minor if minor > 0 else np.inf
        return major, minor, anisotropy

    # Consider turning the following into a KL-divergence styled metric
    @staticmethod
    def relative_error(S_pred, S_emp):
        return norm(S_pred - S_emp, ord='fro') / norm(S_emp, ord='fro')


    @staticmethod
    def kl_divergence(S1, S2, eps=1e-12):
        """
        KL divergence D_KL(N1 || N2)
        """
        S1 = S1 + eps*np.eye(2)
        S2 = S2 + eps*np.eye(2)

        inv_S2 = np.linalg.inv(S2)

        term_trace = np.trace(inv_S2 @ S1)
        term_logdet = np.log(np.linalg.det(S2) / np.linalg.det(S1))
        n = 2

        return 0.5 * (term_trace - n + term_logdet)

    @staticmethod
    def symmetric_kl(S1, S2):
        return 0.5 * (
            Metrics.kl_divergence(S1, S2) +
            Metrics.kl_divergence(S2, S1)
        )


# ==========================================================
# Experiment Framework
# ==========================================================

class Experiment:

    def __init__(self, camera, altitude):
        self.camera = camera
        self.altitude = altitude
        self.T = SE3.nominal_pose(altitude)
        self.projection = ProjectionModel(camera)
        self.propagator = UncertaintyPropagator(self.projection)


    def validate_jacobian(self, Pw):
        J_a = self.projection.analytic_jacobian(self.T, Pw)
        J_n = self.projection.numerical_jacobian(self.T, Pw)
        return norm(J_a - J_n)


    def scaling_study(self, Pw, base_cov, scales):
        results = {}

        for alpha in scales:
            print(f'[RUN]      Running scale: {alpha:.4f}', end='\r')

            Sigma = CovarianceModel.scaled(base_cov, alpha)

            S_pred = self.propagator.analytic_covariance(self.T, Pw, Sigma)
            S_emp  = self.propagator.monte_carlo_covariance(self.T, Pw, Sigma)

            major_pred, minor_pred, aniso_pred = Metrics.ellipse_parameters(S_pred)
            major_emp, minor_emp, aniso_emp   = Metrics.ellipse_parameters(S_emp)

            fro_err = Metrics.relative_error(S_pred, S_emp)
            kld = Metrics.symmetric_kl(S_pred, S_emp)

            results[alpha] = {
                "S_pred": S_pred,
                "S_emp": S_emp,
                "major_pred": major_pred,
                "major_emp": major_emp,
                "anisotropy_pred": aniso_pred,
                "anisotropy_emp": aniso_emp,
                "frobenius_err": fro_err,
                "kl_divergence": kld
            }

        print('\n[RUN]    Scaling study done.')

        return results


    def perturbation_study(self, Pw, base_cov, scales):
        results = {}
        for i in range(6):
            results[i] = {}
            for alpha in scales:
                print(f'[RUN]      Running perturbation: ({i}, {alpha:.4f})', end='\r')

                Sigma = CovarianceModel.perturbed(base_cov, i, alpha)

                S_pred = self.propagator.analytic_covariance(self.T, Pw, Sigma)
                S_emp  = self.propagator.monte_carlo_covariance(self.T, Pw, Sigma)

                major_pred, minor_pred, aniso_pred = Metrics.ellipse_parameters(S_pred)
                major_emp, minor_emp, aniso_emp   = Metrics.ellipse_parameters(S_emp)

                fro_err = Metrics.relative_error(S_pred, S_emp)
                kld = Metrics.symmetric_kl(S_pred, S_emp)

                results[i][alpha] = {
                    "S_pred": S_pred,
                    "S_emp": S_emp,
                    "major_pred": major_pred,
                    "major_emp": major_emp,
                    "anisotropy_pred": aniso_pred,
                    "anisotropy_emp": aniso_emp,
                    "frobenius_err": fro_err,
                    "kl_divergence": kld
                }
            print('\n')
        print('\n[RUN]    Perturbation study done.')

        return results

# ==========================================================
# Main Execution
# ==========================================================

if __name__ == "__main__":

    load = True
    # load = False

    tgt_z = 0.0

    # Camera intrinsics
    camera = Camera(
        w=res[0],
        h=res[1],
        fx=4264.494512341911,
        fy=4262.892739736864,
        cx=958.4594068961055,
        cy=592.50331737885
    )

    # Spatial test points
    points = generate_test_points(
        max_distance=10,
        num_points=15,
        z=tgt_z
    )

    altitudes = [5, 10, 20, 50]
    # altitudes = [ 10, 20 ]

    all_results = {}

    savename = f'catch/uncertainty_prop_{time.time()}.pkl'
    savename = os.path.join(os.path.expanduser('~'), savename)

    for alt in altitudes:
        print(f'[RUN]\n[RUN]  Experiment at {alt}m.')
        all_results[alt] = {}
        experiment = Experiment(camera, altitude=alt)

        test_points, _, _ = stack_test_points(
            points,
            camera,
            experiment,
            tgt_z)

        std_devs=[0.01, 0.01, 0.15, 0.04, 0.04, 1.50]
        # std_devs=[0.01, 0.01, 1.50, 0.04, 0.04, 1.50]

        base_cov = CovarianceModel.realistic(std_devs=std_devs)

        # scales = np.logspace(-1, 1, 5)
        scales = np.logspace(-1, 2, 25)

        if load:
            with open('/home/mwmaster/catch/25_point_5_10_20_50_lowZnoise.pkl', 'rb') as f:  #low z noise (0.15)
            # with open('/home/mwmaster/catch/5_point_5_10_20_50_lowZnoise.pkl', 'rb') as f:  #low z noise (0.15)
            # with open('/home/mwmaster/catch/5_point_5_10_20_50.pkl', 'rb') as f:  # high z noise (1.5)
                all_results = pickle.load(f)
        else:

            for Pw in test_points:
                all_results[alt][tuple(Pw)] = {}
                print(f"[RUN]  Testing point: {Pw}")

                jac_err = experiment.validate_jacobian(Pw)
                all_results[alt][tuple(Pw)]['jacobian_error'] = jac_err
                print(f"[RUN]    Jacobian error: {jac_err:.3e}")

                print('[RUN]    Perturbation study')
                results = experiment.perturbation_study(Pw, base_cov, scales)
                all_results[alt][tuple(Pw)]['perturbation'] = results

                print('[RUN]    Scaling study on base_cov')
                results = experiment.scaling_study(Pw, base_cov, scales)
                all_results[alt][tuple(Pw)]['scaling'] = results

            with open(savename,'wb') as f:
                pickle.dump(all_results, f)

    perturbation_panel(all_results, test_points, scales)
    scaling_panel(all_results, test_points, scales)
    alpha_to_visualize = scales[len(scales)//2]  # pick mid-scale
    ellipse_panel(all_results,
              test_points,
              alpha_to_visualize,
              camera)
    plt.show()

    with open(savename,'wb') as f:
        pickle.dump(all_results, f)
