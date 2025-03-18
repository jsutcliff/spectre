import numpy as np
import matplotlib.pyplot as plt

from codegen.python.symforce.ekf.compute_prior import compute_prior
from codegen.python.symforce.ekf.compute_posterior import compute_posterior
from codegen.python.lcmtypes.ekf._params_t import params_t


params = params_t(m=1.0)

N = 100
t = np.linspace(0, 10, N)
dt = t[1] - t[0]

x_gt = np.ones((4, N))
x_gt[0, :] = t * 8
x_gt[1, :] = t * -2
x_gt[2, :] *= 8
x_gt[3, :] *= -2

x_filter = np.empty_like(x_gt)
x_filter[:, 0] = x_gt[:, 0]
x_filter[2, 0] = 1e-9
x_filter[3, 0] = 1e-9

p_filter = np.empty((4, 4, N))
p_filter[:, :, 0] = np.identity(4) * 1e2

z = np.ones((5, N))
z[:2, :] = x_gt[:2, :].copy()
z[:2, :] += np.random.normal(0, 1, z[:2, :].shape)
z[4, :] *= np.sqrt(1**2 + 4**2)
z[4, :] += np.random.normal(0, 1, z[4, :].shape)


u = np.zeros((2, 1))
mask = np.array([1, 1, 0, 0, 0])

r = np.identity(5)
r[0, 0] = 100
r[1, 1] = 100
r[4, 4] = 100


for i in range(N - 1):
    x_hat, p_hat = compute_prior(dt, x_filter[:, i], p_filter[:, :, i], u, params)
    x_filter[:, i + 1], p_filter[:, :, i + 1] = compute_posterior(x_hat, p_hat, z[:, i], mask, r, params)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].scatter(x_gt[0, :], x_gt[1, :], marker=".", label="ground truth")
ax[0].scatter(z[0, :], z[1, :], marker="x", label="measurements")
ax[0].scatter(x_filter[0, :], x_filter[1, :], marker="o", label="filtered")
ax[0].legend()
ax[0].set_aspect("equal")
ax[0].grid(True, which="both")

ax[1].plot(t, x_gt[2, :], label="dx: ground truth")
ax[1].plot(t, x_gt[3, :], label="dy: ground truth")

dx_filt_p = ax[1].plot(t, x_filter[2, :], label="dx: filtered")
ax[1].fill_between(t, x_filter[2, :] + np.sqrt(p_filter[2, 2, :]), x_filter[2, :] - np.sqrt(p_filter[2, 2, :]), alpha=0.3, color=dx_filt_p[0].get_color())

dy_filt_p = ax[1].plot(t, x_filter[3, :], label="dy: filtered")
ax[1].fill_between(t, x_filter[3, :] + np.sqrt(p_filter[3, 3, :]), x_filter[3, :] - np.sqrt(p_filter[3, 3, :]), alpha=0.3, color=dy_filt_p[0].get_color())

ax[1].legend()
ax[1].grid(True, which="both")

plt.show(block=True)
