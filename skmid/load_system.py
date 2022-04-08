#%%
from signal import Sigmasks
from turtle import color

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel
from skmid.models import generate_model_parameters


def lotka_volterra():

    # define states and parameters
    (states, _, param) = generate_model_parameters(nx=2, nparam=4)

    # x, y,  number of prey (for example, rabbits), number of some predator (for example, foxes)
    x, y = states[0], states[1]

    # positive real parameters describing the interaction of the two species.
    alpha, beta, gamma, delta = param[0], param[1], param[2], param[3]

    # dx = f(states, param) <==> rhs = f(x, p)
    rhs = [alpha * x - beta * x * y, delta * x * y - gamma * y]

    output = [ca.fabs(x - y)]  # absolute value casadi operator

    model = DynamicModel(
        states=states,
        param=param,
        model_dynamics=rhs,
        output=output,
        state_name=["x", "y"],
        param_name=["alpha", "beta", "gamma", "delta"],
        output_name=["|x-y|"],
    )

    # invoke integrator
    rk4 = RungeKutta4(model=model, fs=10)

    # simulate with starting from 10 prey and 10 predators
    param_val = [2.0 / 3.0, 4.0 / 3.0, 1, 1]  # alpha, beta, gamma, delta

    _ = rk4.simulate(x0=[0.9, 1.8], param=param_val, N_steps=1000)

    df_sim = rk4.x_sim_

    df_sim.plot(x="x", y="y", title="Lotka-Volterra")

    description = (
        "Van der Pol oscillator: non-conservative oscillator with non-linear damping."
        "It evolves in time according to the differential equation:"
        "dx1/dt = mu*(1-x2^2)*x1 - x2"
        "dx2/dt = x1"
        "where mu is the damping coefficient, and x is the position coordinate."
        "The data are generated by RungeKutta4(model=model, fs=100).simulate(x0=[-1, 1], param=4, N_steps=10000)"
        "Further details https://en.wikipedia.org/wiki/Van_der_Pol_oscillator"
    )

    return {
        "state_sim": df_sim,
        "input_sim": None,
        "model_function": model,
        "description": description,
    }


def van_der_pol_oscillator():
    """model dx = f(x,p)
    nx = 2, np=1

    Returns:
        _type_: _description_
    """

    # define states and parameters
    (states, _, param) = generate_model_parameters(nx=2, nparam=1)
    x1, x2 = states[0], states[1]
    mu = param[0]

    # dx = f(states, input) <==> rhs = f(x,u, p)
    rhs = [mu * (1 - x2**2) * x1 - x2, x1]
    model = DynamicModel(states=states, param=param, model_dynamics=rhs)

    # invoke integrator
    rk4 = RungeKutta4(model=model, fs=100)

    #
    _ = rk4.simulate(x0=[-1, 1], param=4, N_steps=10000)

    df_sim = rk4.x_sim_

    df_sim.plot(x="x2", y="x1")

    description = (
        "Van der Pol oscillator: non-conservative oscillator with non-linear damping."
        "It evolves in time according to the differential equation:"
        "dx1/dt = mu*(1-x2^2)*x1 - x2"
        "dx2/dt = x1"
        "where mu is the damping coefficient, and x is the position coordinate."
        "The data are generated by RungeKutta4(model=model, fs=100).simulate(x0=[-1, 1], param=4, N_steps=10000)"
        "Further details https://en.wikipedia.org/wiki/Van_der_Pol_oscillator"
    )

    return {
        "state_sim": df_sim,
        "input_sim": None,
        "model_function": model,
        "description": description,
    }


def lorenz_system():

    # define states and parameters
    (states, _, _) = generate_model_parameters(nx=3)
    x, y, z = states[0], states[1], states[2]

    sigma = 10
    rho = 28
    beta = 8.0 / 3.0

    # dx = f(states) <==> rhs = f(x)
    rhs = [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    model = DynamicModel(states=states, model_dynamics=rhs, state_name=["x", "y", "z"])

    # invoke integrator
    rk4 = RungeKutta4(model=model, fs=100)
    _ = rk4.simulate(x0=[1, 1, 1], N_steps=5000)

    df_sim = rk4.x_sim_

    description = (
        "Lorenz attractor: The equations relate the properties of a two-dimensional fluid layer uniformly warmed from below and cooled from above. "
        "It evolves in time according to the differential equation:"
        "dx/dt = sigma * (y - x)",
        "dy/dt = x*(rho - z) - y," "dz/dt = x*y - beta*z",
        "where sigma=10, rho=20 and beta=8/3 are system parameter, x is proportional to the rate of convection,"
        "y to the horizontal temperature variation, and z to the vertical temperature variation.",
        "The data are generated by RungeKutta4(model=model, fs=100).simulate(x0=[1, 1, 1], N_steps=5000)"
        "Further details https://en.wikipedia.org/wiki/Lorenz_system",
    )

    return {
        "state_sim": df_sim,
        "input_sim": None,
        "model_function": model,
        "description": description,
    }


def chua_circuit():

    # define states and parameters
    (states, _, _) = generate_model_parameters(nx=3)
    x, y, z = states[0], states[1], states[2]

    alpha = 15.6
    beta = 28
    m0 = -1.143
    m1 = -0.714

    h = m1 * x + 0.5 * (m0 - m1) * (ca.fabs(x + 1) - ca.fabs(x - 1))

    dx = alpha * (y - x - h)
    dy = x - y + z
    dz = -beta * y

    # dx = f(states) <==> rhs = f(x)
    rhs = [dx, dy, dz]
    model = DynamicModel(states=states, model_dynamics=rhs, state_name=["x", "y", "z"])

    # invoke integrator
    rk4 = RungeKutta4(model=model, fs=100)
    _ = rk4.simulate(x0=[0.7, 0, 0], N_steps=1000)

    df_sim = rk4.x_sim_

    description = (
        "Lorenz attractor: The equations relate the properties of a two-dimensional fluid layer uniformly warmed from below and cooled from above. "
        "It evolves in time according to the differential equation:"
        "dx/dt = sigma * (y - x)",
        "dy/dt = x*(rho - z) - y," "dz/dt = x*y - beta*z",
        "where sigma=10, rho=20 and beta=8/3 are system parameter, x is proportional to the rate of convection,"
        "y to the horizontal temperature variation, and z to the vertical temperature variation.",
        "The data are generated by RungeKutta4(model=model, fs=100).simulate(x0=[1, 1, 1], N_steps=5000)"
        "Further details https://en.wikipedia.org/wiki/Lorenz_system",
    )

    return {
        "state_sim": df_sim,
        "input_sim": None,
        "model_function": model,
        "description": description,
    }


if __name__ == "__main__":
    """run for testing only"""

    data = lotka_volterra()

    data = van_der_pol_oscillator()

    data = lorenz_system()

    data = chua_circuit()

    print("Done!")
