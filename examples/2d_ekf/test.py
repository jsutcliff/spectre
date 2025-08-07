import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.text import Annotation

from codegen.python.symforce.ekf.compute_prior import compute_prior
from codegen.python.symforce.ekf.compute_posterior import compute_posterior
from codegen.python.lcmtypes.ekf._params_t import params_t


def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an nstd sigma error ellipse based on the specified covariance
    matrix (cov).
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip.set_visible(False)

    ax.add_artist(ellip)
    return ellip


params = params_t(m=1.0)

N = 100
x0 = -20
y0 = 20
vx = 8
vy = -3

t = np.linspace(0, 10, N)
dt = t[1] - t[0]

x_gt = np.ones((4, N))
x_gt[0, :] = t * vx + x0
x_gt[1, :] = t * vy + y0
x_gt[2, :] *= vx
x_gt[3, :] *= vy

x_filter = np.zeros_like(x_gt)
x_filter[2, 0] = 1e-9
x_filter[3, 0] = 1e-9

p_filter = np.empty((4, 4, N))
p_filter[:, :, 0] = np.identity(4) * 1e3

z = np.ones((5, N))
z[:2, :] = x_gt[:2, :].copy()
z[:2, :] += np.random.normal(0, 1, z[:2, :].shape)
z[4, :] *= np.sqrt(vx**2 + vy**2)
z[4, :] += np.random.normal(0, 1, z[4, :].shape)


u = np.zeros((2, 1))
mask = np.array([1, 1, 0, 0, 1])

r = np.identity(5)
r[0, 0] = 100
r[1, 1] = 100
r[4, 4] = 100


for i in range(N - 1):
    x_hat, p_hat = compute_prior(dt, x_filter[:, i], p_filter[:, :, i], u, params)
    x_filter[:, i + 1], p_filter[:, :, i + 1] = compute_posterior(x_hat, p_hat, z[:, i], mask, r, params)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

gt_scatter = ax[0].scatter(x_gt[0, :], x_gt[1, :], marker=".", label="ground truth")
meas_scatter = ax[0].scatter(z[0, :], z[1, :], marker="x", label="measurements")
filt_scatter = ax[0].scatter(x_filter[0, :], x_filter[1, :], marker="+", label="filtered")

color = filt_scatter.get_facecolors()[0]
cov_ellipse = plot_cov_ellipse(p_filter[:2, :2, 0], x_filter[:2, 0], ax=ax[0], color=color, alpha=0.2)
meas_to_filt_arrow: Annotation = ax[0].annotate(
    "", xy=(0, 0), xytext=(0, 0), arrowprops=dict(color=meas_scatter.get_facecolors()[0], width=1, headwidth=4, headlength=4, shrink=0.1)
)
gt_to_meas_arrow: Annotation = ax[0].annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(color=gt_scatter.get_facecolors()[0], width=1, headwidth=4, headlength=4, shrink=0.1))
meas_to_filt_arrow.set_visible(False)
gt_to_meas_arrow.set_visible(False)

ax[0].legend()
ax[0].set_aspect("equal")
ax[0].grid(True, which="both")

dx_color = ax[1].plot(t, x_gt[2, :], label="dx: ground truth", linestyle="--")[0].get_color()
dy_color = ax[1].plot(t, x_gt[3, :], label="dy: ground truth", linestyle="--")[0].get_color()

ax[1].plot(t, x_filter[2, :], label="dx: filtered", color=dx_color)
ax[1].fill_between(t, x_filter[2, :] + np.sqrt(p_filter[2, 2, :]), x_filter[2, :] - np.sqrt(p_filter[2, 2, :]), alpha=0.3, color=dx_color)

dy_filt_p = ax[1].plot(t, x_filter[3, :], label="dy: filtered", color=dy_color)
ax[1].fill_between(t, x_filter[3, :] + np.sqrt(p_filter[3, 3, :]), x_filter[3, :] - np.sqrt(p_filter[3, 3, :]), alpha=0.3, color=dy_color)

ax[1].legend()
ax[1].grid(True, which="both")


def update_ellipse(ind):
    i = ind["ind"][0]

    gt_pos = x_gt[:2, i]
    z_pos = z[:2, i]
    filt_pos = x_filter[:2, i]
    cov = p_filter[:2, :2, i]

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 1 * np.sqrt(vals)

    cov_ellipse.set_center(xy=filt_pos)
    cov_ellipse.set_width(width)
    cov_ellipse.set_height(height)
    cov_ellipse.set_angle(theta)

    meas_to_filt_arrow.set_position(z_pos)
    meas_to_filt_arrow.xy = filt_pos

    gt_to_meas_arrow.set_position(gt_pos)
    gt_to_meas_arrow.xy = z_pos


def hover(event):
    vis = cov_ellipse.get_visible()
    if event.inaxes == ax[0]:
        ind = None
        cont = False

        for elem in [gt_scatter, meas_scatter, filt_scatter]:
            elem_cont, elem_ind = elem.contains(event)

            if elem_cont:
                cont = True
                ind = elem_ind

        if cont:
            update_ellipse(ind)
            cov_ellipse.set_visible(True)
            meas_to_filt_arrow.set_visible(True)
            gt_to_meas_arrow.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                cov_ellipse.set_visible(False)
                meas_to_filt_arrow.set_visible(False)
                gt_to_meas_arrow.set_visible(False)

                fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show(block=True)
