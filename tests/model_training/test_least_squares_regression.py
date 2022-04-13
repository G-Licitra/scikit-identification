import os

import casadi as ca
import numpy as np
import pandas as pd
import pytest

from skmid.integrator import RungeKutta4
from skmid.model_training import _gauss_newton
from skmid.models import DynamicModel
from skmid.models import generate_model_parameters

np.random.seed(42)


@pytest.fixture
def load_non_linear_model_data():
    """Generate input signal"""
    CWD = os.getcwd()
    DATA_DIR = "data"
    FILENAME = "non_linear_model.csv"

    data = pd.read_csv(
        filepath_or_buffer=os.path.join(CWD, DATA_DIR, FILENAME), index_col=0
    )

    settings = {
        "N": 10000,  # Number of samples
        "fs": 610.1,  # Sampling frequency [hz]
        "param_truth": [5.625e-6, 2.3e-4, 1, 4.69],
        "param_guess": [5, 2, 1, 5],
        "scale": [1e-6, 1e-4, 1, 1],
        "n_steps_per_sample": 10,
        "initial_condition": np.array([0, 0]),
    }

    return (data, settings)


class TestLeastSquaresRegression:
    def test_algorithm(self, load_non_linear_model_data):
        (data, settings) = load_non_linear_model_data

        fs, n_steps_per_sample, N = (
            settings["fs"],
            settings["n_steps_per_sample"],
            settings["N"],
        )

        (state, input, param) = generate_model_parameters(nstate=2, ninput=1, nparam=4)

        y, dy = state[0], state[1]
        u = input[0]

        M, c, k, k_NL = param[0], param[1], param[2], param[3]

        rhs = [dy, (u - k_NL * y**3 - k * y - c * dy) / M]

        model = DynamicModel(
            states=state,
            inputs=input,
            param=param,
            model_dynamics=rhs,
        )

        ########### Creating a simulator ##########

        dt = 1 / fs / n_steps_per_sample

        # Build an integrator for this system: Runge Kutta 4 integrator
        k1, _ = model.model_function(state, input, param)
        k2, _ = model.model_function(state + dt / 2.0 * k1, input, param)
        k3, _ = model.model_function(state + dt / 2.0 * k2, input, param)
        k4, _ = model.model_function(state + dt * k3, input, param)

        states_final = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Create a function that simulates one step propagation in a sample
        one_step = ca.Function("one_step", [state, input, param], [states_final])

        X = state

        for i in range(n_steps_per_sample):
            X = one_step(X, input, param)

        # Create a function that simulates all step propagation on a sample
        one_sample = ca.Function("one_sample", [state, input, param], [X])
        # Function(one_sample:(i0[2],i1,i2[4])->(o0[2]) MXFunction)

        ############ Simulating the system ##########
        all_samples = one_sample.mapaccum("all_samples", N)
        # Function(one_sample_acc10_acc10_acc10_acc10:(i0[2],i1[1x10000],i2[4x10000])->(o0[2x10000]) MXFunction)

        ############ Identifying the simulated system: single shooting strategy ##########

        # Note, it is in general a good idea to scale your decision variables such
        # that they are in the order of ~0.1..100
        x0 = settings["initial_condition"]

        param_guess = settings["param_guess"]
        scale = settings["scale"]
        u_data = data["u_data"].values
        y_data = data["y_data"].values

        X_symbolic = all_samples(x0, u_data, ca.repmat(param * scale, 1, N))
        # MX(one_sample_acc10_acc10_acc10_acc10(zeros(2x1), [0.0548814, 0.0715189, 0.0602763, ..., 0.075843, 0.00237874, 0.0813575]', repmat(([1e-06, 0.0001, 1, 1]*param), 10000)){0})

        e = y_data - X_symbolic[0, :].T
        nlp = {"x": param, "f": 0.5 * ca.dot(e, e)}

        solver = _gauss_newton(e, nlp, param)

        sol = solver(x0=settings["param_guess"])

        print(sol["x"][:4] * scale)
        print(settings["param_truth"])

        assert ca.norm_inf(sol["x"] * scale - settings["param_truth"]) < 1e-8

        ############ Identifying the simulated system: multiple shooting strategy ##########

        # All states become decision variables
        X = ca.MX.sym("X", 2, N)

        Xn = one_sample.map(N, "openmp")(X, u_data.T, ca.repmat(param * scale, 1, N))

        gaps = Xn[:, :-1] - X[:, 1:]

        e = y_data - Xn[0, :].T

        V = ca.veccat(param, X)

        nlp = {"x": V, "f": 0.5 * ca.dot(e, e), "g": ca.vec(gaps)}

        # Multipleshooting allows for careful initialization
        yd = np.diff(y_data, axis=0) * fs
        X_guess = ca.horzcat(y_data, ca.vertcat(yd, yd[-1])).T

        x0 = ca.veccat(param_guess, X_guess)

        solver = _gauss_newton(e, nlp, V)

        sol = solver(x0=x0, lbg=0, ubg=0)

        # print(sol["x"][:4] * scale)
        # print(param_truth)

        assert ca.norm_inf(sol["x"][:4] * scale - settings["param_truth"]) < 1e-8
