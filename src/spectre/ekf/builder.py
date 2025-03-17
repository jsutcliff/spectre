import os
import inspect
from typing import Callable, Tuple, Sequence, Dict, Union, Any, Protocol, List

from symforce import symbolic as sf
from symforce.values import Values
from symforce.codegen import Codegen, CppConfig, PythonConfig

from ..core.builder import BaseBuilder
from ..core.exponential import expm
from ..core.integrate import get_integrator
from ..core.codegen import generate_cpp_function
from .types import NamedVector
from .defaults import (
    default_process_model,
    default_process_covariance,
    default_measurement_model,
    default_measurement_covarience,
    default_post_state_update,
    default_residual,
)

PROCESS_MODEL_PARAMETERS = ("x", "u", "params")
PROCESS_COVARIANCE_PARAMETERS = ("dt", "x", "u", "params")
POST_STATE_UPDATE_PARAMETERS = ("x", "p", "params")


class EKFBuilder(BaseBuilder):
    """_summary_

    Args:
        BaseBuilder (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    class StateVector(NamedVector):
        """StateVector type of NamedVector that gets configured inside an EKFBuilder

        Args:
            NamedVector (_type_): _description_
        """

        VARIABLES: Sequence[str] = ()

    class ControlVector(NamedVector):
        """ControlVector type of NamedVector that gets configured inside an EKFBuilder

        Args:
            NamedVector (_type_): _description_
        """

        VARIABLES: Sequence[str] = ()

    class ParameterVector(NamedVector):
        """ParameterVector type of NamedVector that gets configured inside an EKFBuilder

        Args:
            NamedVector (_type_): _description_
        """

        VARIABLES: Sequence[str] = ()

    class MeasurementVector(NamedVector):
        """MeasurementVector type of NamedVector that gets configured inside an EKFBuilder

        Args:
            NamedVector (_type_): _description_
        """

        VARIABLES: Sequence[str] = ()

    class StateMatrix(sf.Matrix):
        SHAPE: Tuple[int, int] = (-1, -1)

    class MeasurementMatrix(sf.Matrix):
        SHAPE: Tuple[int, int] = (-1, -1)

    def __init__(self, integrator: str = "euler", integrator_steps: int = 1, include_indentity_measurement: bool = True):

        self.integrator_ = get_integrator(integrator)
        self.integrator_steps_ = integrator_steps

        self.process_model_func_ = default_process_model(self.StateVector)
        self.process_model_func_parameters_ = []
        self.process_covariance_func_ = default_process_covariance(self.StateMatrix)
        self.process_covariance_func_parameters_ = []
        self.measurement_model_func_ = default_measurement_model(self.MeasurementVector)
        self.measurement_covariance_func_ = default_measurement_covarience(self.MeasurementMatrix)
        self.residual_func_ = default_residual(self._measurement_model)
        self.post_state_update_func_ = default_post_state_update()
        self.post_state_update_func_parameters_ = ["x, p"]

        self.include_indentity_measurement_ = include_indentity_measurement

    def set_state_vector(self, variables: Sequence[str]) -> None:
        """Used to assign names to process model states in an EKF

        Args:
            variables (Sequence[str]): List of variable names, must not contain duplicates

        Raises:
            ValueError: If invalid variables names are given
        """

        # Check for duplicates
        if len(variables) != len(set(variables)):
            raise ValueError("Variable list contains duplicate names")

        self.StateVector.VARIABLES = variables
        self.StateMatrix.SHAPE = (len(variables), len(variables))

    def set_control_vector(self, variables: Sequence[str]) -> None:
        """Used to assign names to process model control variables within an EKF

        Args:
            variables (Sequence[str]): List of variable names, must not contain duplicates
        Raises:
            ValueError: If invalid variables names are given
        """

        # Check for duplicates
        if len(variables) != len(set(variables)):
            raise ValueError("Variable list contains duplicate names")

        self.ControlVector.VARIABLES = variables

    def set_parameter_vector(self, variables: Sequence[str]) -> None:
        """Used to assign names to parameters used throughout EKF functions

        Args:
            variables (Sequence[str]): List of variable names, must not contain duplicates

        Raises:
            ValueError: If invalid variables names are given
        """

        # Check for duplicates
        if len(variables) != len(set(variables)):
            raise ValueError("Variable list contains duplicate names")

        self.ParameterVector.VARIABLES = variables

    def set_measurement_vector(self, variables: Sequence[str]) -> None:
        """Used to assign names to measured variables in an EKF

        Args:
            variables (Sequence[str]): List of variable names, must not contain duplicates

        Raises:
            ValueError: If invalid variables names are given
        """

        # Check for duplicates
        if len(variables) != len(set(variables)):
            raise ValueError("Variable list contains duplicate names")

        self.MeasurementVector.VARIABLES = variables
        self.MeasurementMatrix.SHAPE = (len(variables), len(variables))

    def set_process_model_func(self, func: Callable) -> None:
        parameters = inspect.signature(func).parameters.keys()

        for param in parameters:
            if param not in PROCESS_MODEL_PARAMETERS:
                raise ValueError(f"Process model function unknown parameter: {param}, allowed parameters: {PROCESS_MODEL_PARAMETERS}")

            if param == "x" and not self.StateVector.is_configured():
                raise RuntimeError("Process model function requires state variables as 'x', but the state vector has not been assigned yet")

            if param == "u" and not self.ControlVector.is_configured():
                raise RuntimeError("Process model function requires control variables as 'u', but the control vector has not been assigned yet")

            if param == "params" and not self.ParameterVector.is_configured():
                raise RuntimeError("Process model function requires parameters as 'params', but the EKF parameters have not been assigned yet")

        self.process_model_func_ = func
        self.process_model_func_parameters_ = parameters

    def set_process_covariance_func(self, func: Callable) -> None:
        parameters = inspect.signature(func).parameters.keys()

        for param in parameters:
            if param not in PROCESS_COVARIANCE_PARAMETERS:
                raise ValueError(f"Process covariance function unknown parameter: {param}, allowed parameters: {PROCESS_COVARIANCE_PARAMETERS}")

            if param == "x" and not self.StateVector.is_configured():
                raise RuntimeError("Process covariance function requires state variables as 'x', but the state vector has not been assigned yet")

            if param == "u" and not self.ControlVector.is_configured():
                raise RuntimeError("Process covariance function requires control variables as 'u', but the control vector has not been assigned yet")

            if param == "params" and not self.ParameterVector.is_configured():
                raise RuntimeError("Process covariance function requires parameters as 'params', but the EKF parameters have not been assigned yet")

        self.process_covariance_func_ = func
        self.process_covariance_func_parameters_ = parameters

    def set_measurement_model_func(self, func: Callable) -> None:
        # TODO verify signature
        self.measurement_model_func_ = func

    def set_measurement_covariance_func(self, func: Callable) -> None:
        # TODO verify signature
        self.measurement_covariance_func_ = func

    def _process_model(self, x: sf.Matrix, **kwargs: sf.Matrix) -> sf.Matrix:
        """internal function for interfacing with user set process model

        Args:
            x (sf.Matrix): State vector

        Returns:
            sf.Matrix: State vector derivatives
        """

        args: Dict[str, Any] = {}

        if "x" in self.process_model_func_parameters_:
            args["x"] = self.StateVector(x)

        if "u" in self.process_model_func_parameters_:
            args["u"] = self.ControlVector(kwargs["u"])

        if "params" in self.process_model_func_parameters_:
            args["params"] = self.ParameterVector(kwargs["params"])

        return self.process_model_func_(**args).as_matrix()

    def _process_covariance(self, dt: sf.Scalar, x: sf.Matrix, **kwargs: sf.Matrix) -> sf.Matrix:
        """internal function for interfacing with user set process covariance function

        Args:
            dt (sf.Scalar): Timestep in seconds
            x (sf.Matrix): State vector

        Returns:
            sf.Matrix: State covariance square matrix
        """

        args: Dict[str, Any] = {}

        if "dt" in self.process_covariance_func_parameters_:
            args["dt"] = dt

        if "x" in self.process_covariance_func_parameters_:
            args["x"] = self.StateVector(x)

        if "u" in self.process_covariance_func_parameters_:
            args["u"] = self.ControlVector(kwargs["u"])

        if "params" in self.process_covariance_func_parameters_:
            args["params"] = self.ParameterVector(kwargs["params"])

        return self.process_covariance_func_(**args)

    def _measurement_model(self, x: sf.Matrix, **kwargs: sf.Matrix) -> sf.Matrix:
        args: Dict[str, Any] = {"x": self.StateVector(x)}

        if "params" in kwargs:
            args["params"] = self.ParameterVector(kwargs["params"])

        z = self.measurement_model_func_(**args).as_matrix()

        if self.include_indentity_measurement_:
            z_identity = self.StateVector()

            for var in self.StateVector.VARIABLES:
                z_identity[var] = args["x"][var]

            z = z_identity.as_matrix().col_join(z)

        return z

    def _measurement_covariance(self, dt: sf.Scalar, x: sf.Matrix, z: sf.Matrix, **kwargs: sf.Matrix) -> sf.Matrix:
        args: Dict[str, Any] = {"dt": dt, "x": self.StateVector(x), "z": self.MeasurementVector(z)}

        if "u" in kwargs:
            args["u"] = self.ControlVector(kwargs["u"])

        if "params" in kwargs:
            args["params"] = self.ParameterVector(kwargs["params"])

        return self.measurement_covariance_func_(**args)

    def set_post_state_update_func(self, func: Callable) -> None:
        parameters = inspect.signature(func).parameters.keys()

        for param in parameters:
            if param not in POST_STATE_UPDATE_PARAMETERS:
                raise ValueError(f"Post state update function unknown parameter: {param}, allowed parameters: {POST_STATE_UPDATE_PARAMETERS}")

        if "x" not in parameters:
            raise ValueError("Post state update function requires 'x' parameter")

        if "p" not in parameters:
            raise ValueError("Post state update function requires 'p' parameter")

        self.post_state_update_func_ = func
        self.post_state_update_func_parameters_ = parameters

    def _post_state_update(self, x: sf.Matrix, p: sf.Matrix, **kwargs: sf.Matrix) -> Tuple[sf.Matrix, sf.Matrix]:
        args: Dict[str, Any] = {"x": self.StateVector(x), "p": self.StateMatrix(p)}

        if "params" in self.post_state_update_func_parameters_:
            args["params"] = self.ParameterVector(kwargs["params"])

        result = self.post_state_update_func_(**args)

        return (result[0].as_matrix(), result[1])

    def _compute_residual(self, xhat, z, **kwargs):
        return self.residual_func_(xhat, z, **kwargs)

    def _compute_state_transition(self, dt: sf.Scalar, x: sf.Matrix, **kwargs: sf.Matrix) -> sf.Matrix:
        """Internal function that calls the process model to compute state transition matrix

        Args:
            dt (sf.Scalar): Timestep in seconds
            x (sf.Matrix): State vector

        Returns:
            sf.Matrix: F - state transition matrix
        """

        xdot = self._process_model(x=x, **kwargs)
        a = xdot.jacobian(x)
        f = expm(a * dt)

        return f

    def _compute_prior(self, dt: sf.Scalar, x: sf.Matrix, p: sf.Matrix, **kwargs: sf.Matrix) -> Tuple[sf.Matrix, sf.Matrix]:
        """Internal function for EKF prior update

        Args:
            dt (sf.Scalar): Timestep in seconds
            x (sf.Matrix): State vector
            p (sf.Matrix): State covariance matrix
            u (sf.Matrix): (optional) Control matrix
            params (sf.Matrix): (optional) Parameter vector

        Returns:
            Tuple[sf.Matrix, sf.Matrix]: Updated state vector and covariance matrix
        """

        f = self._compute_state_transition(dt, x, **kwargs)
        q = self._process_covariance(dt, x, **kwargs)

        x_hat = self.integrator_(self._process_model, dt, x, num_steps=self.integrator_steps_, **kwargs)
        p_hat = f * p * f.T + q

        x_hat, p_hat = self._post_state_update(x_hat, p_hat, **kwargs)

        return (x_hat, p_hat)

    def _compute_meas_transition(self, x: sf.Matrix, measurement_mask: sf.Matrix, **kwargs: sf.Matrix) -> sf.Matrix:
        z = self._measurement_model(x, **kwargs).multiply_elementwise(measurement_mask)
        h = z.jacobian(x)

        return h

    def _compute_innov_cov(self, dt: sf.Scalar, x_hat, p_hat, z, h, **kwargs: sf.Matrix) -> sf.Matrix:
        r = self._measurement_covariance(dt, x_hat, z, **kwargs)
        s = h * p_hat * h.T + r

        return s

    def _compute_kalman_gain(self, dt: sf.Scalar, x_hat: sf.Matrix, p_hat: sf.Matrix, z: sf.Matrix, h: sf.Matrix, **kwargs: sf.Matrix):
        s = self._compute_innov_cov(dt, x_hat, p_hat, z, h, **kwargs)
        k = p_hat * h.T * (s.inv())

        return k

    def _compute_posterior(self, dt: sf.Scalar, x_hat: sf.Matrix, p_hat: sf.Matrix, z: sf.Matrix, measurement_mask: sf.Matrix, **kwargs: sf.Matrix):

        h = self._compute_meas_transition(x_hat, measurement_mask, **kwargs)
        k = self._compute_kalman_gain(dt, x_hat, p_hat, z, h, **kwargs)

        y = self._compute_residual(x_hat, z, **kwargs)

        x = x_hat + k * y
        p = (self.StateMatrix.eye() - k * h) * p_hat

        x, p = self._post_state_update(x, p, **kwargs)
        return (x, p)

    def _get_compute_prior_io(self):
        inputs = Values()

        inputs["dt"] = sf.Symbol("dt")
        inputs["x"] = self.StateVector.as_symbolic_matrix()
        inputs["p"] = sf.Matrix([[sf.Symbol(f"p_{i}{j}") for i in range(len(self.StateVector.VARIABLES))] for j in range(len(self.StateVector.VARIABLES))])

        args: Dict[str, Any] = {"dt": inputs["dt"], "x": inputs["x"], "P": inputs["p"]}

        if self.ControlVector.VARIABLES:
            inputs["u"] = self.ControlVector.as_symbolic_matrix()
            args["u"] = inputs["u"]

        if self.ParameterVector.VARIABLES:
            params = []

            with inputs.scope("params"):
                for v in self.ParameterVector.VARIABLES:
                    inputs[v] = sf.Symbol(v)
                    params.append(inputs[v])

            args["params"] = sf.Matrix(params)

        x, p = self._compute_prior(**args)

        outputs = Values()

        outputs["x_hat"] = x
        outputs["p_hat"] = p

        return inputs, outputs

    def generate_cpp(self):

        inputs, outputs = self._get_compute_prior_io()
        codegen = Codegen(inputs=inputs, outputs=outputs, config=CppConfig(), name="compute_prior")

        metadata = codegen.generate_function(output_dir="codegen", lcm_bindings_output_dir="codegen")

        for f in metadata.generated_files:
            print("  |- {}".format(os.path.relpath(f, metadata.output_dir)))
