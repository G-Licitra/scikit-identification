import json
import os

import casadi as ca
import numpy as np
import pandas as pd
import pytest

from skmid.integrator import RungeKutta4
from skmid.model_training import LeastSquaresRegression
from skmid.models import DynamicModel
from skmid.models import generate_model_attributes

np.random.seed(0)

########### Creating a simulator ##########
@pytest.fixture
def load_non_linear_model_data():
    """Generate input signal"""
    CWD = os.getcwd()
    DATA_DIR = "data"
    SUB_DATA_DIR = "non_linear_model"

    U = pd.read_csv(
        filepath_or_buffer=os.path.join(CWD, DATA_DIR, SUB_DATA_DIR, "u_data.csv"),
        index_col=0,
    )
    Y = pd.read_csv(
        filepath_or_buffer=os.path.join(CWD, DATA_DIR, SUB_DATA_DIR, "y_data.csv"),
        index_col=0,
    )

    # reading the data from the file
    with open(
        os.path.join(CWD, DATA_DIR, SUB_DATA_DIR, "settings.json"), mode="r"
    ) as j_object:
        settings = json.load(j_object)

    return (U, Y, settings)


class TestLeastSquaresRegression:
    def test_algorithm(self, load_non_linear_model_data):
        (U, Y, settings) = load_non_linear_model_data

        # Define the model
        (state, input, param) = generate_model_attributes(
            state_size=2, input_size=1, parameter_size=4
        )
        y, dy = state[0], state[1]
        u = input[0]
        M, c, k, k_NL = param[0], param[1], param[2], param[3]
        rhs = [dy, (u - k_NL * y**3 - k * y - c * dy) / M]
        model = DynamicModel(
            state=state,
            input=input,
            parameter=param,
            model_dynamics=rhs,
        )

        # Call Estimator
        fs = settings["fs"]
        n_steps_per_sample = settings["n_steps_per_sample"]
        estimator = LeastSquaresRegression(
            model=model, fs=fs, n_steps_per_sample=n_steps_per_sample
        )

        # Estimate parameters
        param_guess = settings["param_guess"]
        scale = settings["scale"]
        estimator.fit(U=U, Y=Y, param_guess=param_guess, param_scale=scale)
        param_est = estimator.coef_

        assert ca.norm_inf(param_est * scale - settings["param_truth"]) < 1e-8

        x_fit = estimator.model_fit_


# fs, n_steps_per_sample, N = (
#     settings["fs"],
#     settings["n_steps_per_sample"],
#     settings["N"],
# )

# x0 = settings["initial_condition"]
# param_guess = settings["param_guess"]
# scale = settings["scale"]
# u_data = data["u_data"].values
# y_data = data["y_data"].values

# (state, input, param) = generate_model_parameters(nstate=2, ninput=1, nparam=4)

# y, dy = state[0], state[1]
# u = input[0]

# M, c, k, k_NL = param[0], param[1], param[2], param[3]

# rhs = [dy, (u - k_NL * y**3 - k * y - c * dy) / M]

# model = DynamicModel(
#     states=state,
#     inputs=input,
#     param=param,
#     model_dynamics=rhs,
# )

# estimator = LeastSquaresRegression(
#     model=model, fs=fs, n_steps_per_sample=n_steps_per_sample
# )

# solution = estimator.fit(
#     U=u_data,
#     Y=y_data,
#     initial_condition=x0,
#     param_guess=param_guess,
#     param_scale=scale,
# )

# assert ca.norm_inf(solution["x"] * scale - settings["param_truth"]) < 1e-8

# ############ Identifying the simulated system: single shooting strategy ##########

# # Note, it is in general a good idea to scale your decision variables such
# # that they are in the order of ~0.1..100

# # X_symbolic = all_samples(x0, u_data, ca.repmat(param * scale, 1, N))
# # MX(one_sample_acc10_acc10_acc10_acc10(zeros(2x1), [0.0548814, 0.0715189, 0.0602763, ..., 0.075843, 0.00237874, 0.0813575]', repmat(([1e-06, 0.0001, 1, 1]*param), 10000)){0})

# # def fit(self, )

# rk4 = RungeKutta4(model=model, fs=fs, n_steps_per_sample=n_steps_per_sample)

# N = len(u_data)
# ############ Simulating the system ##########
# all_samples = rk4._RungeKutta4__one_sample.mapaccum("all_samples", N)
# # Function(one_sample_acc10_acc10_acc10_acc10:(i0[2],i1[1x10000],i2[4x10000])->(o0[2x10000]) MXFunction)
# X_symbolic = all_samples(x0, u_data, ca.repmat(param * scale, 1, N))  #

# e = y_data - X_symbolic[0, :].T
# nlp = {"x": param, "f": 0.5 * ca.dot(e, e)}

# solver = _gauss_newton(e, nlp, param)

# sol = solver(x0=settings["param_guess"])

# # print(sol["x"][:4] * scale)
# # print(settings["param_truth"])

# assert ca.norm_inf(sol["x"] * scale - settings["param_truth"]) < 1e-8

# ############ Identifying the simulated system: multiple shooting strategy ##########

# # print(sol["x"][:4] * scale)
# # print(param_truth)

# assert ca.norm_inf(sol["x"][:4] * scale - settings["param_truth"]) < 1e-8
