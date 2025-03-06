from typing import Type, Callable, Tuple, Any

from symforce import symbolic as sf

from .types import NamedVector


def default_process_model(state_class: Type[Any]) -> Callable[[Any, Any, Any], Any]:

    def fcn(x: NamedVector, u: NamedVector, p: NamedVector):
        print(x.as_matrix(), u.as_matrix(), p.as_matrix())

        # Default zero derivative
        x_dot = state_class()

        return x_dot

    return fcn


def default_process_covarience(state_matrix: Type[Any]) -> Callable[[Any, Any, Any, sf.Scalar], Any]:

    def fcn(x: NamedVector, u: NamedVector, p: NamedVector, dt: sf.Scalar):
        print(x.as_matrix(), u.as_matrix(), p.as_matrix(), dt)

        # Default zero derivative
        covaraince = state_matrix.eye()

        return covaraince

    return fcn


def default_post_state_update() -> Callable[[Any, Any, Any], Tuple[Any, Any]]:

    def fcn(x, P, _):
        return (x, P)

    return fcn
