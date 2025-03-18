import logging
import symforce

symforce.set_epsilon_to_number(1e-9)

from spectre.ekf import EKFBuilder
from symforce import symbolic as sf
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

builder = EKFBuilder(integrator="rk4", include_indentity_measurement=True)

builder.set_state_vector(["x", "y", "dx", "dy"])
builder.set_control_vector(["fx", "fy"])
builder.set_parameter_vector(["m"])
builder.set_measurement_vector(["speed"])


def process_model(x: EKFBuilder.StateVector, u: EKFBuilder.ControlVector, params: EKFBuilder.ParameterVector):
    x_dot = builder.StateVector()

    x_dot["x"] = x["dx"]
    x_dot["y"] = x["dy"]
    x_dot["dx"] = u["fx"] / params["m"]
    x_dot["dy"] = u["fy"] / params["m"]

    return x_dot


def measurement_model(x: EKFBuilder.StateVector):
    z = builder.MeasurementVector()

    z["speed"] = sf.sqrt(x["dx"] * x["dx"] + x["dy"] * x["dy"])

    return z


builder.set_process_model_func(process_model)
builder.set_measurement_model_func(measurement_model)


output_dir = Path(__file__).resolve().parent / "codegen"
# builder.debug_print()
# builder.generate_cpp()
builder.generate_python(output_dir=output_dir)
