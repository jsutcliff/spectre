import logging
from typing import Type, Callable, Tuple, Any

from symforce import symbolic as sf


def default_process_model(state_class: Type[Any]) -> Callable:
    """Method for generating default process model function. It's not a good idea to use this unless state variables are not expected to change

    Args:
        state_class (Type[Any]): Type of state vector class. Usually EKFBuilder.StateVector

    Returns:
        Callable: Function the returns zeros the size of state_class
    """

    def fcn():
        # Default zero derivative
        logging.warning("Process model function not set, using default. Consider setting process model for nonzero state vector derivates")
        return state_class()

    return fcn


def default_process_covariance(state_matrix: Type[Any]) -> Callable:
    """Method for generating default process covariance function. Not a good idea to use this

    Args:
        state_class (Type[Any]): Type of state vector class. Usually EKFBuilder.StateVector

    Returns:
        Callable: Function the returns identity matrix the size of state_class
    """

    def fcn():
        # Default identity
        logging.warning("Process covariance function not set, using default (identity matrix)")
        return state_matrix.eye()

    return fcn


def default_measurement_model(measurement_class: Type[Any]) -> Callable:
    """Generates default measurement model function. Not a good idea to use this

    Args:
        measurement_class (Type[Any]): Type of measurement vector. Usually EKFBuilder.MeasurementVector

    Returns:
        Callable: Function that takes no arguments and returns zero vector of the appropraite size
    """

    def fcn():
        # Default zero connection
        return measurement_class()

    return fcn


def default_post_state_update() -> Callable:
    """Generates default function to be called after the EKF state is modified

    Returns:
        Callable: Function that simply passes all input parameters back as a tuple
    """

    def fcn(x, p: sf.Matrix) -> Tuple[Any, sf.Matrix]:
        logging.debug("Using default post state update function")
        return (x, p)

    return fcn


def default_residual(measurement_model_fcn: Callable) -> Callable:
    """Generates default residual function

    Args:
        measurement_model_fcn (Callable): Function that calculates a measurement vector from the current EKF state

    Returns:
        Callable: Function that simply returns the difference between input measurement vector 'z' and the measurement vector returned by measurement_model_fcn
    """

    def residual_func(xhat, z, **kwargs):
        logging.debug("Using default residual function")
        return z - measurement_model_fcn(xhat, **kwargs)

    return residual_func
