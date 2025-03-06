from spectre.ekf import EKFBuilder
from symforce import symbolic as sf


builder = EKFBuilder()

builder.set_state_vector(["x", "y", "dx", "dy"])


def process_model(x: EKFBuilder.StateVector, u: EKFBuilder.ControlVector, p: EKFBuilder.ParameterVector):
    x_dot = builder.StateVector()

    x_dot["x"] = x["dx"]
    x_dot["y"] = x["dy"]
    x_dot["dx"] = 0.0
    x_dot["dy"] = 0.0

    return x_dot


builder.set_process_model_func(process_model)

print(builder.process_model_(sf.Matrix([10, 20, 30, 40]), sf.Matrix([1]), sf.Matrix([2])))
