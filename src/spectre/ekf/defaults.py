from typing import Type, Callable, Tuple, Any

from symforce import symbolic as sf

from .types import NamedVector


def default_process_model(state_class: Type[Any]):

    def fcn():
        # Default zero derivative
        return state_class()

    return fcn


def default_measurement_model(measurement_class: Type[Any]):

    def fcn():
        # Default zero connection
        return measurement_class()

    return fcn


def default_process_covariance(state_matrix: Type[Any]):

    def fcn():
        # Default identity
        return state_matrix.eye()

    return fcn


def default_measurement_covarience(measurement_matrix: Type[Any]):

    def fcn(dt: sf.Scalar, x: NamedVector, z: NamedVector):
        # print(dt, x.as_matrix())

        # Default zero derivative
        covaraince = measurement_matrix.eye()

        return covaraince

    return fcn


def default_residual(measurement_model_fcn):
    def residual_func(xhat, z, **kwargs):
        return z - measurement_model_fcn(xhat, **kwargs)

    return residual_func


def default_post_state_update():

    def fcn(x, p: sf.Matrix) -> Tuple[Any, sf.Matrix]:
        return (x, p)

    return fcn
