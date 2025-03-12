import symforce.symbolic as sf
from typing import Callable


def euler(xdot_fn: Callable, dt: sf.Scalar, x: sf.Matrix, num_steps: int = 1, **kwargs: sf.Matrix) -> sf.Matrix:
    """numerical integration via euler's method"""
    # assert x.SHAPE == xdot.SHAPE
    h = dt / num_steps

    for _ in range(num_steps):
        x += h * xdot_fn(x=x, **kwargs)

    return x


def rk4(xdot_fn: Callable, dt: sf.Scalar, x: sf.Matrix, num_steps: int = 1, **kwargs: sf.Matrix) -> sf.Matrix:
    """numerical integration via RK4"""
    # assert x.SHAPE == xdot.SHAPE
    h = dt / num_steps

    for _ in range(num_steps):
        k1 = h * xdot_fn(x=x, **kwargs)
        k2 = h * xdot_fn(x=x + k1 / 2.0, **kwargs)
        k3 = h * xdot_fn(x=x + k2 / 2.0, **kwargs)
        k4 = h * xdot_fn(x=x + k3, **kwargs)

        k = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        x += k

    return x


def get_integrator(name: str) -> Callable:
    if name not in INTEGRATORS:
        raise NotImplementedError(f"Integrator: {name} not implemented")

    return INTEGRATORS[name]


INTEGRATORS = {"euler": euler, "rk4": rk4, "default": euler}
