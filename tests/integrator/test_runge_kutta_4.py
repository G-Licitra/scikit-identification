import json
import os

import casadi as ca
import numpy as np
import pandas as pd
import pytest
from scipy.signal import chirp

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel
from skmid.models import generate_model_attributes

np.random.seed(42)

# # test f(x,u) with multiple input, multiple size, list
# @pytest.fixture
# def generate_input_signal():
#     """Generate input signal"""

#     N = 2000  # Number of samples
#     fs = 500  # Sampling frequency [hz]
#     t = np.linspace(0, (N - 1) * (1 / fs), N)
#     df_input = pd.DataFrame(
#         data={
#             "chirp": 2 * chirp(t, f0=1, f1=10, t1=5, method="logarithmic"),
#             "noise": 2 * np.random.random(N),
#         },
#         index=t,
#     )

#     return (df_input, fs)


# @pytest.fixture
# def load_casadi_data():
#     """Load data and model settings from casadi example"""

#     CWD = os.getcwd()
#     DATA_DIR = "data"
#     SUB_DATA_DIR = "non_linear_model"

#     U = pd.read_csv(
#         filepath_or_buffer=os.path.join(CWD, DATA_DIR, SUB_DATA_DIR, "u_data.csv"), index_col=0
#     )
#     Y = pd.read_csv(
#         filepath_or_buffer=os.path.join(CWD, DATA_DIR, SUB_DATA_DIR, "Y_data.csv"), index_col=0
#     )

#     # reading the data from the file
#     with open(os.path.join(CWD, DATA_DIR, SUB_DATA_DIR, "settings.json"), mode="r") as j_object:
#         settings = json.load(j_object)

#     return (U, Y, settings)


class TestRungeKutta4:
    """Test class for function generate_model_parameters."""

    # def test_model_with_states(self, generate_input_signal):
    #     """Test simulation with model dx=f(x)."""
    #     pass

    # def test_model_with_states_input(self, generate_input_signal):
    #     """Test simulation with model dx=f(x,u)."""
    #     pass

    # def test_model_with_states_input(self, generate_input_signal):
    #     """Test simulation with model dx=f(x,p)."""
    #     pass

    # def test_model_with_states_inputs_param(self, generate_input_signal):
    #     """Test simulation with model dx=f(x,u,p)."""

    #     (df_input, fs) = generate_input_signal

    #     (x, u, param) = generate_model_attributes(nx=2, nu=2, nparam=2)

    #     # assign specific name
    #     x1, x2 = x[0], x[1]
    #     u1, u2 = u[0], u[1]
    #     ka, kb = param[0], param[1]

    # param_truth = [0.1, 0.5]  # ca.DM([0.1, 0.5])

    # # xdot = f(x,u,p) <==> rhs = f(x,u,p)
    # rhs = [u1 - ka * x1, u1 * u2 / x1 - u1 * x2 / x1 - kb * x2]

    # sys = DynamicModel(states=x, inputs=u, param=param, model_dynamics=rhs)

    # # #%%
    # rk4 = RungeKutta4(model=sys, fs=fs)

    # # #%%
    # rk4.simulate(x0=[-1, 1], input=df_input, param=param_truth)

    # rk4.x_sim_

    # def test_match_casadi_example(self, generate_input_signal):

    #     (U, Y, settings) = load_casadi_data

    #     # Define the model
    #     (state, input, param) = generate_model_attributes(state_size=2, input_size=1, parameter_size=4)

    #     y, dy = state[0], state[1]
    #     u = input[0]
    #     M, c, k, k_NL = param[0], param[1], param[2], param[3]
    #     rhs = [dy, (u - k_NL * y**3 - k * y - c * dy) / M]

    #     model = DynamicModel(
    #         state=state,
    #         input=input,
    #         parameter=param,
    #         model_dynamics=rhs,
    #     )

    #     # Call Integrator
    #     fs = settings['fs']
    #     n_steps_per_sample = settings['n_steps_per_sample']

    #     rk4 = RungeKutta4(fs=fs, n_steps_per_sample=n_steps_per_sample).simulate(initial_condition=settings['x0'], input=U, parameter=settings['parameter'])
