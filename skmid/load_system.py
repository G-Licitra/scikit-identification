#%%
from signal import Sigmasks

import numpy as np
import pandas as pd

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel
from skmid.models import generate_model_parameters

#%%
# ref: https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

(states, _, param) = generate_model_parameters(nx=2, nparam=4)

x = states[0]  # number of prey (for example, rabbits)
y = states[1]  # number of some predator (for example, foxes)

# positive real parameters describing the interaction of the two species.
alpha = param[0]
beta = param[1]
gamma = param[2]
sigma = param[3]

# represents the instantaneous growth rates of the two populations;
# [dx, dy] = f(states, param) <==> rhs = f(x,p)
rhs = [alpha * x - beta * x * y, sigma * x * y - gamma * y]

sys = DynamicModel(states=states, param=param, model_dynamics=rhs)

# numerical evaluation
x_test = [10, 10]
param_test = [2 / 3, 4 / 3, 1, 1]
sys.print_summary()
rhs_num, y_num = sys.evaluate(state_num=x_test, param_num=param_test)
print(f"rhs = {rhs_num}, \ny = {y_num}")


#%%
rk4 = RungeKutta4(model=sys, fs=1)

# #%%
_ = rk4.simulate(x0=[10, 10], param=[2 / 3, 4 / 3, 1, 1])

# %%
df_sim = rk4.sim
