import json
import os
import warnings

import casadi as ca
import numpy as np
import pandas as pd
import pandas.testing as pdt
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


@pytest.fixture
def generate_inpulse_signal():
    """Generate Impulse input signal"""

    N = 500  # Number of samples
    fs = 50  # Sampling frequency [hz]
    t = np.linspace(0, (N - 1) * (1 / fs), N)
    df_input = pd.DataFrame(index=t).assign(inpulse=np.zeros(N))
    df_input.iloc[0] = 1

    return (df_input, fs)


class TestRungeKutta4:
    """Test class for function generate_model_parameters."""

    def test_model_with_states(self):
        """Test simulation with model dx=f(x)."""

        (x, _, _) = generate_model_attributes(
            state_size=1, input_size=0, parameter_size=0
        )

        # initialize first-order model
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

        # initialize first-order model
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

    def test_different_format_input(self, generate_inpulse_signal):
        """."""

        (df_input, fs) = generate_inpulse_signal

        (x, u, _) = generate_model_attributes(
            state_size=1, input_size=1, parameter_size=0
        )

        tau, kp = 1, 1
        sys = DynamicModel(
            state=x, input=u, model_dynamics=[-(1 / tau) * x + (kp / tau) * u]
        )

        rk4 = RungeKutta4(model=sys, fs=fs)

        input_dict = {
            "dataframe": df_input,
            "dataframe_with_non_consistent_fs": df_input.reset_index(drop=True),
            "series": df_input.iloc[:, 0],
            "numpy_array": df_input.values,
            "numpy_array_1dim": df_input.values.squeeze(),
        }

        for format, input_ in input_dict.items():
            message = f"Test failed with input format {format}"

            _ = rk4.simulate(initial_condition=[0], input=input_)

            df_X = rk4.state_sim_
            df_Y = rk4.output_sim_

            # check equality of dataframe
            assert isinstance(df_X, pd.DataFrame), message
            assert isinstance(df_Y, pd.DataFrame), message

            pdt.assert_frame_equal(df_X, df_Y)

            # check size of dataframe
            assert len(df_X) == len(df_input) + 1, message
            assert len(df_Y) == len(df_input) + 1, message

            # check no missing values
            assert df_X.notna().all().values, message
            assert df_Y.notna().all().values, message

            # check model consistency: convergence with no bias
            assert df_input.iloc[-1].values == pytest.approx(
                df_X.iloc[-1].values, rel=1e-7, abs=1e-6
            ), message

    def test_mimo_system(self):

        # generate dirac impulse
        N = 3000  # Number of samples
        fs = 50  # Sampling frequency [hz]
        t = np.linspace(0, (N - 1) * (1 / fs), N)
        df_input = pd.DataFrame(
            data=np.zeros(shape=(N, 2)), columns=["u1", "u2"], index=t
        )
        df_input.iloc[0, :] = [0.1, -0.1]

        (x, u, param) = generate_model_attributes(
            state_size=2, input_size=2, parameter_size=2
        )

        x1, x2 = x[0], x[1]
        u1, u2 = u[0], u[1]
        k1, k2 = param[0], param[1]

        # chemical reactor: mimo
        rhs = [u1 - k1 * x1, u1 * u2 / x1 - u1 * x2 / x1 - k2 * x2]

        sys = DynamicModel(
            state=x,
            input=u,
            parameter=param,
            model_dynamics=rhs,
            parameter_name=["k1", "k2"],
        )

        rk4 = RungeKutta4(model=sys, fs=fs)

        input_dict = {
            "dataframe": df_input,
            "dataframe_with_non_consistent_fs": df_input.reset_index(drop=True),
            "numpy_array": df_input.values,
        }

        for format, input_ in input_dict.items():
            message = f"Test failed with input format {format}"

            _ = rk4.simulate(
                initial_condition=[-0.15, 0.1], input=input_, parameter=[0.1, 0.5]
            )

            df_X = rk4.state_sim_
            df_Y = rk4.output_sim_

            # check equality of dataframe
            assert isinstance(df_X, pd.DataFrame), message
            assert isinstance(df_Y, pd.DataFrame), message

            pdt.assert_frame_equal(df_X, df_Y)

            # check size of dataframe
            assert len(df_X) == len(df_input) + 1, message
            assert len(df_Y) == len(df_input) + 1, message

            # check no missing values
            assert df_X.notna().all().all(), message
            assert df_Y.notna().all().all(), message

            # check model consistency: convergence with no bias
            assert df_input.iloc[-1].values == pytest.approx(
                df_X.iloc[-1].values, rel=1e-3, abs=1e-3
            ), message

    def test_input_sample_frequency(self):

        (x, u, _) = generate_model_attributes(
            state_size=1, input_size=1, parameter_size=0
        )

        tau, kp = 1, 1
        sys = DynamicModel(
            state=x, input=u, model_dynamics=[-(1 / tau) * x + (kp / tau) * u]
        )

        with pytest.raises(ValueError):
            _ = RungeKutta4(model=sys, fs=-12)

            _ = RungeKutta4(model=sys, fs=0)

            _ = RungeKutta4(model=sys, fs=50, n_steps_per_sample=0)

            _ = RungeKutta4(model=sys, fs=50, n_steps_per_sample=0.1)

            _ = RungeKutta4(model=sys, fs=50, n_steps_per_sample=-np.sqrt(0.1))
