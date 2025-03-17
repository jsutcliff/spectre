from typing import Callable

import symforce.symbolic as sf


def euler(xdot_fn: Callable, dt: sf.Scalar, x: sf.Matrix, num_steps: int = 1, **kwargs: sf.Matrix) -> sf.Matrix:
    """Numerical integration using Euler's method

    Args:
        xdot_fn (Callable): Function to call to get derivatives
        dt (sf.Scalar): Timestep in seconds
        x (sf.Matrix): State vector to forward integrate
        num_steps (int, optional): Number of integration steps. Defaults to 1.

    Returns:
        sf.Matrix: State vector after integration
    """

    h = dt / num_steps

    for _ in range(num_steps):
        x += h * xdot_fn(x=x, **kwargs)

    return x


def rk4(xdot_fn: Callable, dt: sf.Scalar, x: sf.Matrix, num_steps: int = 1, **kwargs: sf.Matrix) -> sf.Matrix:
    """Numerical integration using RK4

    Args:
        xdot_fn (Callable): Function to call to get derivatives
        dt (sf.Scalar): Timestep in seconds
        x (sf.Matrix): State vector to forward integrate
        num_steps (int, optional): Number of integration steps. Defaults to 1.

    Returns:
        sf.Matrix: State vector after integration
    """
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
    """Helper function to get integrator from string

    Args:
        name (str): Name of integration from ["euler", "rk4"]

    Raises:
        NotImplementedError: If specified integrator was not recognized

    Returns:
        Callable: Integration function
    """

    if name not in INTEGRATORS:
        raise NotImplementedError(f"Integrator: {name} not implemented")

    return INTEGRATORS[name]


INTEGRATORS = {"euler": euler, "rk4": rk4, "default": euler}
