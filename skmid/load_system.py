#%%
from signal import Sigmasks
from turtle import color

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel
from skmid.models import generate_model_parameters

# temp

#%%
# ref: https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

# (states, _, param) = generate_model_parameters(nx=2, nparam=4)

# x = states[0]  # number of prey (for example, rabbits)
# y = states[1]  # number of some predator (for example, foxes)

# # positive real parameters describing the interaction of the two species.
# alpha = param[0]
# beta = param[1]
# gamma = param[2]
# sigma = param[3]

# # represents the instantaneous growth rates of the two populations;
# # [dx, dy] = f(states, param) <==> rhs = f(x,p)
# rhs = [alpha * x - beta * x * y, sigma * x * y - gamma * y]

# sys = DynamicModel(states=states, param=param, model_dynamics=rhs)

# # numerical evaluation
# x_test = [10, 10]
# param_test = [2 / 3, 4 / 3, 1, 1]
# sys.print_summary()
# rhs_num, y_num = sys.evaluate(state_num=x_test, param_num=param_test)
# print(f"rhs = {rhs_num}, \ny = {y_num}")


# #%%
# rk4 = RungeKutta4(model=sys, fs=1)

# # #%%
# _ = rk4.simulate(x0=[10, 10], param=[2 / 3, 4 / 3, 1, 1])

# # %%
# df_sim = rk4.sim

#%% ----------------

# pendulum

# (states, _, _) = generate_model_parameters(nx=2)


# theta = states[0]  # pendulum angle [rad]
# omega = states[1]  # pendulum angular velocity [rad/s]

# # parameters
# m, L, g = 1.0, 3.0, 9.81  # mass, length, gravity

# # represents the instantaneous growth rates of the two populations;
# # [dtheta, domega] = f(states, input) <==> rhs = f(x,u)
# rhs = [omega,
#        - (g/L) * theta
#        ]

# sys = DynamicModel(states=states, model_dynamics=rhs, state_name=['theta', 'omega'])

# # numerical evaluation
# x_test = [0, 0]

# sys.print_summary()

# rhs_num, y_num = sys.evaluate(state_num=x_test)
# print(f"rhs = {rhs_num}, \ny = {y_num}")


# #%%
# rk4 = RungeKutta4(model=sys, fs=100)

# # #%%
# _ = rk4.simulate(x0=[np.pi/8, 0], N_steps=2000) # start at 45 degrees and 0 rad/s

# df_sim = rk4.x_sim_

# # convert rad in deg for better visualization
# df_sim = df_sim.applymap(lambda x: x * 180 / np.pi)

# df_sim.plot(y =['theta', 'omega'])
# # # %%
# print('Done!')

# %%


def van_der_pol_oscillator():
    """model dx = f(x,p)

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


if __name__ == "__main__":
    """run for testing only"""

    data = lorenz_system()
    print("Done!")
