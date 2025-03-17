from spectre.ekf import EKFBuilder
from symforce import symbolic as sf


builder = EKFBuilder()

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

dt = sf.Symbol("dt")
x = sf.Symbol("x")
y = sf.Symbol("y")
dx = sf.Symbol("dx")
dy = sf.Symbol("dy")
state = sf.Matrix([x, y, dx, dy])
P = sf.Matrix([[sf.Symbol(f"p_{i}{j}") for i in range(4)] for j in range(4)])

meas_x = sf.Symbol("meas_x")
meas_y = sf.Symbol("meas_y")
meas_dx = sf.Symbol("meas_dx")
meas_dy = sf.Symbol("meas_dy")
speed = sf.Symbol("speed")
meas = sf.Matrix([meas_x, meas_y, meas_dx, meas_dy, speed])

measurement_mask = sf.Matrix([sf.Symbol(c) for c in "abcde"])

print(state)
# print(builder.process_model_(state))
# print(builder.compute_state_transition_(dt, state))
# print(builder.process_covariance_(dt, state))
# print(builder.compute_state_transition_(dt, state))
# print(builder.compute_prior_(dt, state, P))

# print(builder.measurement_model_(state))
# print(builder.compute_meas_transition_(state, measurement_mask))
# # print(builder.compute_posterior_(dt, state, P, meas, measurement_mask)[1].SHAPE)


builder.generate_cpp()
