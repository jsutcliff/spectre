import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from matplotlib.patches import Ellipse
from matplotlib.text import Annotation

from codegen.python.symforce.relative.compute_prior import compute_prior as compute_prior_relative
from codegen.python.symforce.relative.compute_posterior import compute_posterior as compute_posterior_relative
from codegen.python.symforce.identity.compute_prior import compute_prior as compute_prior_identity
from codegen.python.symforce.identity.compute_posterior import compute_posterior as compute_posterior_identity


N = 100
num_entities = 2
x0 = [-20, 10]
y0 = [20, 10]
vx = [8, 1e-9]
vy = [-3, 1e-9]

t = np.linspace(0, 10, N)
dt = t[1] - t[0]

x_gt = np.ones((num_entities, 4, N))

for i in range(num_entities):
    x_gt[i, 0, :] = t * vx[i] + x0[i]
    x_gt[i, 1, :] = t * vy[i] + y0[i]
    x_gt[i, 2, :] *= vx[i]
    x_gt[i, 3, :] *= vy[i]

x_filter = np.zeros_like(x_gt) + np.random.normal(0, 1e-9, size=x_gt.shape)

p_filter = np.empty((num_entities, 4, 4, N))
p_filter[:, :, :, 0] = np.identity(4) * 1e3

z_a = np.zeros_like(x_gt[0, :, :])
z_a[:2, :] = x_gt[0, :2, :].copy()

z_ab = np.ones((2, N))
diff_ab = x_gt[1, :2, :] - x_gt[0, :2, :]
z_ab[0, :] = np.linalg.norm(diff_ab, axis=0)
z_ab[1, :] = np.atan2(diff_ab[1, :], diff_ab[0, :])

# print(z_a)
# print(z_ab)

mask = np.array([1, 1])

for i in range(N - 1):
    r = np.identity(4) * 10

    x_hat_a = x_filter[0, :, i]
    p_hat_a = p_filter[0, :, :, i]
    x_hat_a, p_hat_a = compute_prior_identity(dt, x_hat_a, p_hat_a)
    print("in", x_hat_a, p_hat_a, z_a[:, i], r)
    x_hat_a, p_hat_a = compute_posterior_identity(x_hat_a, p_hat_a, z_a[:, i], np.array([1, 1, 1, 1]), r)

    x_hat_b = x_filter[1, :, i]
    p_hat_b = p_filter[1, :, :, i]
    x_hat_b, p_hat_b = compute_prior_identity(dt, x_hat_b, p_hat_b)

    x_hat_both = np.concatenate((x_hat_a, x_hat_b), axis=0)
    p_hat_both = scipy.linalg.block_diag(p_hat_a, p_hat_b)

    r = np.identity(2)
    r[0, 0] = 100
    r[1, 1] = 1

    x_hat, p_hat = compute_posterior_relative(x_hat_both, p_hat_both, z_ab[:, i], mask, r)

    np.set_printoptions(precision=3)

    x_filter[0, :, i + 1] = x_hat[:4]
    x_filter[1, :, i + 1] = x_hat[4:]
    p_filter[0, :, :, i + 1] = p_hat[:4, :4]
    p_filter[1, :, :, i + 1] = p_hat[4:, 4:]

    # x_filter[0, :, i + 1] = x_hat_a
    # x_filter[1, :, i + 1] = x_hat_b
    # p_filter[0, :, :, i + 1] = p_hat_a
    # p_filter[1, :, :, i + 1] = p_hat_b

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

for i in range(num_entities):
    gt_scatter = ax[0].scatter(x_gt[i, 0, :], x_gt[i, 1, :], marker=".", label="ground truth")
    # meas_scatter = ax[0].scatter(z[0, :], z[1, :], marker="x", label="measurements")
    filt_scatter = ax[0].scatter(x_filter[i, 0, :], x_filter[i, 1, :], marker="+", label="filtered")

# cov_ellipse = plot_cov_ellipse(p_filter[:2, :2, 0], x_filter[:2, 0], ax=ax[0], color=color, alpha=0.2)
# meas_to_filt_arrow: Annotation = ax[0].annotate(
#     "", xy=(0, 0), xytext=(0, 0), arrowprops=dict(color=meas_scatter.get_facecolors()[0], width=1, headwidth=4, headlength=4, shrink=0.1)
# )
# gt_to_meas_arrow: Annotation = ax[0].annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(color=gt_scatter.get_facecolors()[0], width=1, headwidth=4, headlength=4, shrink=0.1))
# meas_to_filt_arrow.set_visible(False)
# gt_to_meas_arrow.set_visible(False)

ax[0].legend()
ax[0].set_aspect("equal")
ax[0].grid(True, which="both")

# dx_color = ax[1].plot(t, x_gt[2, :], label="dx: ground truth", linestyle="--")[0].get_color()
# dy_color = ax[1].plot(t, x_gt[3, :], label="dy: ground truth", linestyle="--")[0].get_color()

# ax[1].plot(t, x_filter[2, :], label="dx: filtered", color=dx_color)
# ax[1].fill_between(t, x_filter[2, :] + np.sqrt(p_filter[2, 2, :]), x_filter[2, :] - np.sqrt(p_filter[2, 2, :]), alpha=0.3, color=dx_color)

# dy_filt_p = ax[1].plot(t, x_filter[3, :], label="dy: filtered", color=dy_color)
# ax[1].fill_between(t, x_filter[3, :] + np.sqrt(p_filter[3, 3, :]), x_filter[3, :] - np.sqrt(p_filter[3, 3, :]), alpha=0.3, color=dy_color)

# ax[1].legend()
# ax[1].grid(True, which="both")


plt.show(block=True)
