from typing import Callable, Tuple, Sequence

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

    def set_process_model_func(self, func: Callable[[StateVector, ControlVector, ParameterVector], StateVector]):
        # TODO verify signature
        self.process_model_func_ = func

    def process_model_(self, x: sf.Matrix, u: sf.Matrix, p: sf.Matrix) -> sf.Matrix:
        x_named = self.StateVector(x)
        u_named = self.ControlVector(u)
        p_named = self.ParameterVector(p)

        return self.process_model_func_(x_named, u_named, p_named).as_matrix()

    def set_process_covariance_func(
        self, func: Callable[[StateVector, ControlVector, ParameterVector, sf.Scalar], StateMatrix]
    ):
        # TODO verify signature
        self.process_covariance_func_ = func

    def process_covariance_(self, x: sf.Matrix, u: sf.Matrix, p: sf.Matrix, dt: sf.Scalar) -> sf.Matrix:
        x_named = self.StateVector(x)
        u_named = self.ControlVector(u)
        p_named = self.ParameterVector(p)

        return self.process_covariance_func_(x_named, u_named, p_named, dt)

    def set_post_state_update_func(
        self, func: Callable[[StateVector, StateMatrix, ParameterVector], Tuple[StateVector, StateMatrix]]
    ):
        self.post_state_update_func_ = func

    def post_state_update_(self, x: sf.Matrix, P: sf.Matrix, p: sf.Matrix) -> Tuple[sf.Matrix, sf.Matrix]:
        x_named = self.StateVector(x)
        P_named = self.StateMatrix(P)
        p_named = self.ParameterVector(p)

        result = self.post_state_update_func_(x_named, P_named, p_named)

        return (result[0].as_matrix(), result[1])

    def compute_state_transition_(self, x: sf.Matrix, u: sf.Matrix, p: sf.Matrix, dt: sf.Scalar):
        """
        A = df/dx       --> Jacobian of system model wrt state
        F = expm(A*dt)  --> state transition matrix
        """

        xdot = self.process_model_(x, u, p)
        A = xdot.jacobian(x)
        F = expm(A * dt)

        return F

    def compute_prior_(
        self, x: sf.Matrix, P: sf.Matrix, u: sf.Matrix, p: sf.Matrix, dt: sf.Scalar
    ) -> Tuple[sf.Matrix, sf.Matrix]:

        xdot_fun = self.process_model_
        F = self.compute_state_transition_(x, u, p, dt)
        Q = self.process_covariance_(x, u, p, dt)

        xhat = self.integrator_(xdot_fun, x, u, dt, p, num_steps=self.integrator_steps_)
        Phat = F * P * F.T + Q

        xhat, Phat = self.post_state_update_(xhat, Phat, p)
        return (xhat, Phat)
