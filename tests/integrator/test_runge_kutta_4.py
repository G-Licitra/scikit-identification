import json
import os

import casadi as ca
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest
from scipy.signal import chirp

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel
from skmid.models import generate_model_attributes


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


# test f(x,u) with multiple input, multiple size, list
@pytest.fixture
def generate_input_signal():
    """Generate input signal"""

    N = 2000  # Number of samples
    fs = 500  # Sampling frequency [hz]
    t = np.linspace(0, (N - 1) * (1 / fs), N)
    df_input = pd.DataFrame(
        data={
            "chirp": 2 * chirp(t, f0=1, f1=10, t1=5, method="logarithmic"),
            "noise": 2 * np.random.random(N),
        },
        index=t,
    )

    return (df_input, fs)


@pytest.fixture
def generate_step_signal():
    """Generate Step input signal"""

    N = 1000  # Number of samples
    fs = 25  # Sampling frequency [hz]
    t = np.linspace(0, (N - 1) * (1 / fs), N)
    df_input = pd.DataFrame(index=t).assign(
        step=lambda x: np.where(x.index < t[int(N / 4)], 0, 1)
    )

    return (df_input, fs)


class TestRungeKutta4:
    """Test class for function generate_model_parameters."""

    def test_model_with_states(self):
        """Test simulation with model dx=f(x)."""

        (x, _, _) = generate_model_attributes(
            state_size=1, input_size=0, parameter_size=0
        )

        tau = 1
        sys = DynamicModel(state=x, model_dynamics=[-(1 / tau) * x])

        n_steps = 50
        x0 = [1]
        rk4 = RungeKutta4(model=sys)
        _ = rk4.simulate(initial_condition=x0, n_steps=n_steps)

        df_X = rk4.state_sim_
        df_Y = rk4.output_sim_

        # check equality of dataframe
        pdt.assert_frame_equal(df_X, df_Y)

        # check size of dataframe. Note: +1 is because it includes the initial condition
        assert len(df_X) == n_steps + 1
        assert len(df_Y) == n_steps + 1

        # check frequency is assigned correctly fs=1 default
        assert all(np.diff(df_Y.index) == 1)

        # check no missing values
        assert df_X.notna().all().values
        assert df_Y.notna().all().values

        # check model consistency: convergence with no bias
        assert df_X.iloc[0].values == pytest.approx(x0)
        assert df_X.iloc[-1].values == pytest.approx(0)

    def test_model_with_states_input(self, generate_step_signal):
        """Test simulation with model dx=f(x,u)."""

        (df_input, fs) = generate_step_signal

        (x, u, _) = generate_model_attributes(
            state_size=1, input_size=1, parameter_size=0
        )

        tau, kp = 1, 1
        sys = DynamicModel(
            state=x, input=u, model_dynamics=[-(1 / tau) * x + (kp / tau) * u]
        )

        rk4 = RungeKutta4(model=sys, fs=fs)
        _ = rk4.simulate(initial_condition=[0], input=df_input)

        df_X = rk4.state_sim_
        df_Y = rk4.output_sim_

        # check equality of dataframe
        pdt.assert_frame_equal(df_X, df_Y)

        # check size of dataframe
        assert len(df_X) == len(df_input) + 1
        assert len(df_Y) == len(df_input) + 1

        # check no missing values
        assert df_X.notna().all().values
        assert df_Y.notna().all().values

        # check model consistency: convergence with no bias
        assert df_input.iloc[-1].values == pytest.approx(df_X.iloc[-1].values)

    def test_model_with_states_parameter(self):
        """Test simulation with model dx=f(x,u)."""

        (x, _, tau) = generate_model_attributes(
            state_size=1, input_size=0, parameter_size=1
        )

        sys = DynamicModel(state=x, parameter=tau, model_dynamics=[-(1 / tau) * x])

        n_steps = 50
        x0 = [1]
        tau_num = [1]
        rk4 = RungeKutta4(model=sys)
        _ = rk4.simulate(initial_condition=x0, parameter=tau_num, n_steps=n_steps)

        df_X = rk4.state_sim_
        df_Y = rk4.output_sim_

        # check equality of dataframe
        pdt.assert_frame_equal(df_X, df_Y)

        # check size of dataframe. Note: +1 is because it includes the initial condition
        assert len(df_X) == n_steps + 1
        assert len(df_Y) == n_steps + 1

        # check frequency is assigned correctly fs=1 default
        assert all(np.diff(df_Y.index) == 1)

        # check no missing values
        assert df_X.notna().all().values
        assert df_Y.notna().all().values

        # check model consistency: convergence to 1 with no bias
        assert df_X.iloc[0].values == pytest.approx(x0)
        assert df_X.iloc[-1].values == pytest.approx(0)

    def test_model_with_states_input_parameter(self, generate_step_signal):
        """Test simulation with model dx=f(x,u,p)."""

        (df_input, fs) = generate_step_signal

        (x, u, parameter) = generate_model_attributes(
            state_size=1, input_size=1, parameter_size=2
        )

        tau, kp = parameter[0], parameter[1]
        sys = DynamicModel(
            state=x,
            input=u,
            parameter=parameter,
            model_dynamics=[-(1 / tau) * x + (kp / tau) * u],
        )

        parameter_num = [1, 1]
        rk4 = RungeKutta4(model=sys, fs=fs)
        _ = rk4.simulate(initial_condition=[0], input=df_input, parameter=parameter_num)

        df_X = rk4.state_sim_
        df_Y = rk4.output_sim_

        # check equality of dataframe
        pdt.assert_frame_equal(df_X, df_Y)

        # check size of dataframe
        assert len(df_X) == len(df_input) + 1
        assert len(df_Y) == len(df_input) + 1

        # check no missing values
        assert df_X.notna().all().values
        assert df_Y.notna().all().values

        # check model consistency: convergence with no bias
        assert df_input.iloc[-1].values == pytest.approx(df_X.iloc[-1].values)
