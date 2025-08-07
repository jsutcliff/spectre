import logging
import symforce

symforce.set_epsilon_to_number(1e-9)

from spectre.ekf import EKFBuilder
from symforce import symbolic as sf
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

builder = EKFBuilder(integrator="rk4", include_indentity_measurement=False)

builder.set_state_vector(["x_a", "y_a", "vx_a", "vy_a", "x_b", "y_b", "vx_b", "vy_b"])
# builder.set_control_vector(["fx", "fy"])
# builder.set_parameter_vector(["m"])
builder.set_measurement_vector(["range", "bearing"])


def process_model(x: EKFBuilder.StateVector):
    x_dot = builder.StateVector()

    x_dot["x_a"] = x["vx_a"]
    x_dot["y_a"] = x["vy_a"]
    x_dot["vx_a"] = 0
    x_dot["vy_a"] = 0

    x_dot["x_b"] = x["vx_b"]
    x_dot["y_b"] = x["vy_b"]
    x_dot["vx_b"] = 0
    x_dot["vy_b"] = 0

    return x_dot


def measurement_model(x: EKFBuilder.StateVector):
    z = builder.MeasurementVector()

    x_distance = x["x_b"] - x["x_a"]
    y_distance = x["y_b"] - x["y_a"]

    z["range"] = sf.sqrt(x_distance * x_distance + y_distance * y_distance)
    z["bearing"] = sf.atan2(y_distance, x_distance)

    return z


builder.set_process_model_func(process_model)
builder.set_measurement_model_func(measurement_model)


output_dir = Path(__file__).resolve().parent / "codegen"
builder.debug_print()
# builder.generate_cpp()
builder.generate_python(output_dir=output_dir, namespace="relative")
