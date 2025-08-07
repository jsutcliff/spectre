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

builder.set_state_vector(["x", "y", "vx", "vy"])


def process_model(x: EKFBuilder.StateVector):
    x_dot = builder.StateVector()

    x_dot["x"] = x["vx"]
    x_dot["y"] = x["vy"]
    x_dot["vx"] = 0
    x_dot["vy"] = 0

    return x_dot


builder.set_process_model_func(process_model)

output_dir = Path(__file__).resolve().parent / "codegen"
builder.debug_print()
# builder.generate_cpp()
builder.generate_python(output_dir=output_dir, namespace="identity")
