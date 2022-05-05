import json
import os

import casadi as ca
import numpy as np
import pandas as pd
import pylab as pl
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


@pytest.fixture
def load_vehicle_2d_data():
    """Generate input signal"""
    CWD = os.getcwd()
    DATA_DIR = "data"
    SUB_DATA_DIR = "vehicle_2d"

    # U = pd.read_csv(
    #     filepath_or_buffer=os.path.join(CWD, DATA_DIR, SUB_DATA_DIR, "u_data.csv"),
    #     index_col=0,
    # )

    data = pl.array(
        pl.loadtxt(
            os.path.join(CWD, DATA_DIR, SUB_DATA_DIR, "data_2d_vehicle.dat"),
            delimiter=", ",
            skiprows=1,
        )
    )

    return data


class TestLeastSquaresRegression:
    def test_algorithm(self, load_non_linear_model_data):
        pass
        # (U, Y, settings) = load_non_linear_model_data

        # # Define the model
        # (state, input, param) = generate_model_attributes(
        #     state_size=2, input_size=1, parameter_size=4
        # )
        # y, dy = state[0], state[1]
        # u = input[0]
        # M, c, k, k_NL = param[0], param[1], param[2], param[3]
        # rhs = [dy, (u - k_NL * y**3 - k * y - c * dy) / M]
        # model = DynamicModel(
        #     state=state,
        #     input=input,
        #     parameter=param,
        #     model_dynamics=rhs,
        # )

        # # Call Estimator
        # fs = settings["fs"]
        # n_steps_per_sample = settings["n_steps_per_sample"]
        # estimator = LeastSquaresRegression(
        #     model=model, fs=fs, n_steps_per_sample=n_steps_per_sample
        # )

        # # Estimate parameters
        # param_guess = settings["param_guess"]
        # scale = settings["scale"]
        # estimator.fit(U=U, Y=Y, param_guess=param_guess, param_scale=scale)
        # param_est = estimator.coef_

        # assert ca.norm_inf(param_est * scale - settings["param_truth"]) < 1e-8

        # x_fit = estimator.model_fit_

    def test_fitting_with_mimo_system(self, load_vehicle_2d_data):

        data = load_vehicle_2d_data

        time_points = data[100:1000:5, 1]

        ydata = pd.DataFrame(
            data=data[100:1000:5, [2, 4, 6, 8]],
            index=time_points,
            columns=["x1", "x2", "x3", "x4"],
        )

        udata = pd.DataFrame(
            data=data[100:1000:5, [9, 10]][:-1, :],
            index=time_points[:-1],
            columns=["u1", "u2"],
        )

        pinit = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]

        # # Define the model
        (x, u, p) = generate_model_attributes(
            state_size=4, input_size=2, parameter_size=6
        )

        rhs = [
            x[3] * np.cos(x[2] + p[0] * u[0]),
            x[3] * np.sin(x[2] + p[0] * u[0]),
            x[3] * u[0] * p[1],
            p[2] * u[1]
            - p[3] * u[1] * x[3]
            - p[4] * x[3] ** 2
            - p[5]
            - (x[3] * u[0]) ** 2 * p[1] * p[0],
        ]

        model = DynamicModel(
            state=x,
            input=u,
            parameter=p,
            model_dynamics=rhs,
        )

        # Call Estimator
        fs = 10
        estimator = LeastSquaresRegression(model=model, fs=fs)

        estimator.fit(U=udata, Y=ydata, param_guess=pinit)

        param_est = estimator.coef_
        x_fit = estimator.model_fit_

        pl.figure()

        pl.subplot2grid((4, 2), (0, 0))
        pl.plot(time_points[1:], x_fit["x1"], label="$X_{fit}$")
        pl.plot(time_points, ydata["x1"], ".", label="$X_{measure}$")
        pl.xlabel("$t$")
        pl.ylabel("$X$", rotation=0)
        pl.legend(loc="upper right")

        pl.subplot2grid((4, 2), (1, 0))
        pl.plot(time_points[1:], x_fit["x2"], label="$Y_{fit}$")
        pl.plot(time_points, ydata["x2"], ".", label="$Y_{measure}$")
        pl.xlabel("$t$")
        pl.ylabel("$X$", rotation=0)
        pl.legend(loc="lower left")

        pl.subplot2grid((4, 2), (2, 0))
        pl.plot(time_points, psihat, label="$\psi_{sim}$")
        pl.plot(time_points, ydata[:, 2], label="$\psi_{meas}$")
        pl.xlabel("$t$")
        pl.ylabel("$\psi$", rotation=0)
        pl.legend(loc="lower left")

        pl.subplot2grid((4, 2), (3, 0))
        pl.plot(time_points, vhat, label="$v_{sim}$")
        pl.plot(time_points, ydata[:, 3], label="$v_{meas}$")
        pl.xlabel("$t$")
        pl.ylabel("$v$", rotation=0)
        pl.legend(loc="upper left")

        pl.subplot2grid((4, 2), (0, 1), rowspan=4)
        pl.plot(xhat, yhat, label="$(X_{sim},\,Y_{sim})$")
        pl.plot(ydata[:, 0], ydata[:, 1], label="$(X_{meas},\,Y_{meas})$")
        pl.xlabel("$X$")
        pl.ylabel("$Y$", rotation=0)
        pl.legend(loc="upper left")

        pl.show()
