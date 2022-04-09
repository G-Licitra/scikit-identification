import casadi as ca
import numpy as np
import pandas as pd
import pytest
from scipy.signal import chirp

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel
from skmid.models import generate_model_parameters

np.random.seed(42)

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


class TestRungeKutta4:
    """Test class for function generate_model_parameters."""

    def test_model_with_states(self, generate_input_signal):
        """Test simulation with model dx=f(x)."""
        pass

    def test_model_with_states_input(self, generate_input_signal):
        """Test simulation with model dx=f(x,u)."""
        pass

    def test_model_with_states_input(self, generate_input_signal):
        """Test simulation with model dx=f(x,p)."""
        pass

    def test_model_with_states_inputs_param(self, generate_input_signal):
        """Test simulation with model dx=f(x,u,p)."""
        pass

        # (df_input, fs) = generate_input_signal

        # (x, u, param) = generate_model_parameters(nx=2, nu=2, nparam=2)

        # # assign specific name
        # x1, x2 = x[0], x[1]
        # u1, u2 = u[0], u[1]
        # ka, kb = param[0], param[1]

        # param_truth = [0.1, 0.5]  # ca.DM([0.1, 0.5])

        # # xdot = f(x,u,p) <==> rhs = f(x,u,p)
        # rhs = [u1 - ka * x1, u1 * u2 / x1 - u1 * x2 / x1 - kb * x2]

        # sys = DynamicModel(states=x, inputs=u, param=param, model_dynamics=rhs)

        # # #%%
        # rk4 = RungeKutta4(model=sys, fs=fs)

        # # #%%
        # rk4.simulate(x0=[-1, 1], input=df_input, param=param_truth)

        # rk4.x_sim_
