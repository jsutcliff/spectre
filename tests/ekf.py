from spectre.ekf import EKFBuilder
from symforce import symbolic as sf


builder = EKFBuilder()

builder.set_state_vector(["x", "y", "dx", "dy"])


def process_model(x: EKFBuilder.StateVector):
    x_dot = builder.StateVector()

    x_dot["x"] = x["dx"]
    x_dot["y"] = x["dy"]
    x_dot["dx"] = 0.0
    x_dot["dy"] = 0.0

    return x_dot


builder.set_process_model_func(process_model)

dt = 1.0
x = sf.Symbol("x")
y = sf.Symbol("y")
dx = sf.Symbol("dx")
dy = sf.Symbol("dy")
state = sf.Matrix([x, y, dx, dy])
P = sf.Matrix([[sf.Symbol(f"p_{i}{j}") for i in range(4)] for j in range(4)])

print(state)
print(builder.process_model_(state))
print(builder.compute_state_transition_(dt, state))
print(builder.process_covariance_(dt, state))
print(builder.compute_state_transition_(dt, state))
print(builder.compute_prior_(dt, state, P))
