from typing import Type, Callable, Tuple, Any

from symforce import symbolic as sf

from .types import NamedVector


def default_process_model(state_class: Type[Any]):

    def fcn(x: NamedVector):
        # print(x.as_matrix())

        # Default zero derivative
        x_dot = state_class()

        return x_dot

    return fcn


def default_process_covarience(state_matrix: Type[Any]):

    def fcn(dt: sf.Scalar, x: NamedVector):
        # print(dt, x.as_matrix())

        # Default zero derivative
        covaraince = state_matrix.eye()

        return covaraince

    return fcn


def default_post_state_update():

    def fcn(x, P):
        return (x, P)

    return fcn
