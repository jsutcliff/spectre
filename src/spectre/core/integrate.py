import symforce.symbolic as sf
from typing import Callable


def euler(xdot_fn: Callable, x: sf.Matrix, u: sf.Matrix, dt: sf.Scalar, p: sf.Matrix, num_steps: int = 1) -> sf.Matrix:
    """numerical integration via euler's method"""
    # assert x.SHAPE == xdot.SHAPE
    h = dt / num_steps

    for _ in range(num_steps):
        x += h * xdot_fn(x, u, p)

    return x


def rk4(xdot_fn: Callable, x: sf.Matrix, u: sf.Matrix, dt: sf.Scalar, p: sf.Matrix, num_steps: int = 1) -> sf.Matrix:
    """numerical integration via RK4"""
    # assert x.SHAPE == xdot.SHAPE
    h = dt / num_steps

    for _ in range(num_steps):
        k1 = h * xdot_fn(x, u, p)
        k2 = h * xdot_fn(x + k1 / 2.0, u, p)
        k3 = h * xdot_fn(x + k2 / 2.0, u, p)
        k4 = h * xdot_fn(x + k3, u, p)

        k = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        x += k

    return x


def get_integrator(name: str) -> Callable:
    if name not in INTEGRATORS:
        raise NotImplementedError(f"Integrator: {name} not implemented")

    return INTEGRATORS[name]


INTEGRATORS = {"euler": euler, "rk4": rk4, "default": euler}
