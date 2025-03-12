from typing import Callable, Tuple, Sequence, Dict, Union, Any, Protocol

from symforce import symbolic as sf

from ..core.builder import BaseBuilder
from ..core.exponential import expm
from ..core.integrate import get_integrator
from .types import NamedVector
from .defaults import default_process_model, default_process_covarience, default_post_state_update


class EKFBuilder(BaseBuilder):

    class StateVector(NamedVector):
        VARIABLES: Sequence[str] = ()

    class ControlVector(NamedVector):
        VARIABLES: Sequence[str] = ()

    class ParameterVector(NamedVector):
        VARIABLES: Sequence[str] = ()

    class StateMatrix(sf.Matrix):
        SHAPE: Tuple[int, int] = (-1, -1)

    def __init__(self, integrator: str = "euler", integrator_steps: int = 1):

        self.integrator_ = get_integrator(integrator)
        self.integrator_steps_ = integrator_steps

        self.process_model_func_ = default_process_model(self.StateVector)
        self.process_covariance_func_ = default_process_covarience(self.StateMatrix)
        self.post_state_update_func_ = default_post_state_update()

    def set_state_vector(self, variables: Sequence[str]):
        self.StateVector.VARIABLES = variables
        self.StateMatrix.SHAPE = (len(variables), len(variables))

    def set_control_vector(self, variables: Sequence[str]):
        self.ControlVector.VARIABLES = variables

    def set_parameter_vector(self, variables: Sequence[str]):
        self.ParameterVector.VARIABLES = variables

    def set_process_model_func(self, func):
        # TODO verify signature
        self.process_model_func_ = func

    def process_model_(self, x: sf.Matrix, **kwargs: sf.Matrix) -> sf.Matrix:
        args: Dict[str, Any] = {"x": self.StateVector(x)}

        if "u" in kwargs:
            args["u"] = self.ControlVector(kwargs["u"])

        if "p" in kwargs:
            args["p"] = self.ParameterVector(kwargs["p"])

        return self.process_model_func_(**args).as_matrix()

    def set_process_covariance_func(
        self, func: Callable[[StateVector, ControlVector, ParameterVector, sf.Scalar], StateMatrix]
    ):
        # TODO verify signature
        self.process_covariance_func_ = func

    def process_covariance_(self, dt: sf.Scalar, x: sf.Matrix, **kwargs: sf.Matrix) -> sf.Matrix:
        args: Dict[str, Any] = {"dt": dt, "x": self.StateVector(x)}

        if "u" in kwargs:
            args["u"] = self.ControlVector(kwargs["u"])

        if "p" in kwargs:
            args["p"] = self.ParameterVector(kwargs["p"])

        return self.process_covariance_func_(**args)

    def set_post_state_update_func(
        self, func: Callable[[StateVector, StateMatrix, ParameterVector], Tuple[StateVector, StateMatrix]]
    ):
        self.post_state_update_func_ = func

    def post_state_update_(self, x: sf.Matrix, P: sf.Matrix, **kwargs: sf.Matrix) -> Tuple[sf.Matrix, sf.Matrix]:
        args: Dict[str, Any] = {"x": self.StateVector(x), "P": self.StateMatrix(P)}

        if "u" in kwargs:
            args["u"] = self.ControlVector(kwargs["u"])

        if "p" in kwargs:
            args["p"] = self.ParameterVector(kwargs["p"])

        result = self.post_state_update_func_(**args)

        return (result[0].as_matrix(), result[1])

    def compute_state_transition_(self, dt: sf.Scalar, x: sf.Matrix, **kwargs: sf.Matrix):
        """
        A = df/dx       --> Jacobian of system model wrt state
        F = expm(A*dt)  --> state transition matrix
        """

        xdot = self.process_model_(x=x, **kwargs)

        A = xdot.jacobian(x)
        F = expm(A * dt)

        return F

    def compute_prior_(
        self, dt: sf.Scalar, x: sf.Matrix, P: sf.Matrix, **kwargs: sf.Matrix
    ) -> Tuple[sf.Matrix, sf.Matrix]:

        xdot_fun = self.process_model_
        F = self.compute_state_transition_(dt, x, **kwargs)
        Q = self.process_covariance_(dt, x, **kwargs)

        xhat = self.integrator_(xdot_fun, dt, x, num_steps=self.integrator_steps_, **kwargs)
        Phat = F * P * F.T + Q

        xhat, Phat = self.post_state_update_(xhat, Phat, **kwargs)
        return (xhat, Phat)
