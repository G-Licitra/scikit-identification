#%%
import casadi as ca
import numpy as np

from skmid.models import _infer_model_type
from skmid.models import DynamicModel


class RungeKutta4(DynamicModel):
    r"""Create Explicit Runge Kutta order 4 integrator`

    Parameters
    ---------
    ode : casadi.Function
        Function which define the model via ODE, formally,
        f: (x, u, theta) -> rhs.
        where
        rhs: right-hand side,
        x: differential states,
        u: input,
        theta: model Parameters
    N_steps_per_sample: int
        Steps forward within one integration step
    fs: int
        frequency at which the system is sampled
    N_steps: int
        number of integration steps

    Returns
    ---------
    one_sample : casadi.Function
        One step forward model integrator
    all_sample : casadi.Function
        N steps forward model integrator

    """

    def __init__(self, *, model: DynamicModel, fs: int = 1):

        __N_steps_per_sample: int = 1

        dt = 1 / fs / __N_steps_per_sample

        # unpack model
        x = model.states
        u = model.inputs
        param = model.param

        # Function rhs, ordinary differential equation
        Fode = model.Fmodel

        _model_type = _infer_model_type(model)

        if _model_type["struct"] == "f(x,u)":
            # Build an RK4 integrator with param as symbolic variable
            (k1, _) = Fode(x, u)
            (k2, _) = Fode(x + dt / 2.0 * k1, u)
            (k3, _) = Fode(x + dt / 2.0 * k2, u)
            (k4, _) = Fode(x + dt * k3, u)
            xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Create a function that simulates one step propagation in a sample
            one_step = ca.Function("one_step", [x, u], [xf])

            # Carry out forward simulation within the entire sample time
            X = x
            for i in range(__N_steps_per_sample):
                X = one_step(X, u)

            # Create a function that simulates all step propagation on a sample
            self.one_sample = ca.Function(
                "one_sample",
                [x, u],
                [X],
                ["x[k]", "u[k]"],
                ["x[k+1] = x[k] + dt * f(x[k], u[k])"],
            )

        elif _model_type["struct"] == "f(x)":
            # Build an RK4 integrator with param as symbolic variable
            (k1, _) = Fode(x)
            (k2, _) = Fode(x + dt / 2.0 * k1)
            (k3, _) = Fode(x + dt / 2.0 * k2)
            (k4, _) = Fode(x + dt * k3)
            xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Create a function that simulates one step propagation in a sample
            one_step = ca.Function("one_step", [x], [xf])

            # Carry out forward simulation within the entire sample time
            X = x
            for i in range(__N_steps_per_sample):
                X = one_step(X, u)

            # Create a function that simulates all step propagation on a sample
            self.one_sample = ca.Function(
                "one_sample", [x], [X], ["x[k]"], ["x[k+1] = x[k] + dt * f(x[k])"]
            )

        elif _model_type["struct"] == "f(x,p)":
            # Build an RK4 integrator with param as symbolic variable
            (k1, _) = Fode(x, param)
            (k2, _) = Fode(x + dt / 2.0 * k1, param)
            (k3, _) = Fode(x + dt / 2.0 * k2, param)
            (k4, _) = Fode(x + dt * k3, param)
            xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Create a function that simulates one step propagation in a sample
            one_step = ca.Function("one_step", [x, param], [xf])

            # Carry out forward simulation within the entire sample time
            X = x
            for i in range(__N_steps_per_sample):
                X = one_step(X, param)

            # Create a function that simulates all step propagation on a sample
            self.one_sample = ca.Function(
                "one_sample",
                [x, param],
                [X],
                ["x[k]", "param"],
                ["x[k+1] = x[k] + dt * f(x[k], param)"],
            )

        elif _model_type["struct"] == "f(x,u,p)":
            # Build an RK4 integrator with param as symbolic variable
            (k1, _) = Fode(x, u, param)
            (k2, _) = Fode(x + dt / 2.0 * k1, u, param)
            (k3, _) = Fode(x + dt / 2.0 * k2, u, param)
            (k4, _) = Fode(x + dt * k3, u, param)
            xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Create a function that simulates one step propagation in a sample
            one_step = ca.Function("one_step", [x, u, param], [xf])

            # Carry out forward simulation within the entire sample time
            X = x
            for i in range(__N_steps_per_sample):
                X = one_step(X, u, param)

            # Create a function that simulates all step propagation on a sample
            self.one_sample = ca.Function(
                "one_sample",
                [x, u, param],
                [X],
                ["x[k]", "u[k]", "param"],
                ["x[k+1] = x[k] + dt * f(x[k], u[k], param)"],
            )
        else:
            raise ValueError("Model type not supported")

    def simulate(self, x0, input, param=None):

        # derive N_steps.
        N_steps = len(input)

        # get time and attach one extra sample to time
        time = input.index
        ts = time[-1] + (time[1] - time[0])  # sample time
        time = np.concatenate([time.values.reshape(-1, 1), np.array(ts, ndmin=2)])

        # Propagate simulation for N steps and generate trajectory
        all_samples = self.one_sample.mapaccum("all_samples", N_steps)
        x_sim = np.array(all_samples(x0, input.values.T, ca.repmat(param, 1, N_steps)))

        # attach the initial condition to x_sim with dim=[N_steps+1, nx]
        x_sim = np.concatenate([np.array(x0).reshape(1, -1), x_sim.T])

        df_sim = pd.DataFrame(data=x_sim, index=time.reshape(-1))

        return df_sim


#%%----------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.signal import chirp
from skmid.models import generate_model_parameters, DynamicModel
import seaborn as sns

# Choose an excitation signal
np.random.seed(42)
N = 2000  # Number of samples
fs = 500  # Sampling frequency [hz]
t = np.linspace(0, (N - 1) * (1 / fs), N)
df_input = pd.DataFrame(
    data={
        "u1": 2 * chirp(t, f0=1, f1=10, t1=5, method="logarithmic"),
        "u2": 2 * np.random.random(N),
    },
    index=t,
)

x0 = [1, -1]  # Initial Condition x0 = [0;0]; [nx = 2]

# #%%----------------------------------------------------------------

(x, u, param) = generate_model_parameters(nx=2, nu=2, nparam=2)

# assign specific name
x1, x2 = x[0], x[1]
u1, u2 = u[0], u[1]
ka, kb = param[0], param[1]

param_truth = [0.1, 0.5]  # ca.DM([0.1, 0.5])

# xdot = f(x,u,p) <==> rhs = f(x,u,p)
rhs = [u1 - ka * x1, u1 * u2 / x1 - u1 * x2 / x1 - kb * x2]

sys = DynamicModel(states=x, inputs=u, param=param, model_dynamics=rhs)

# #%%
rk4 = RungeKutta4(model=sys, fs=fs)

# #%%
# xsim = rk4.simulate(x0=x0, input=df_input, param=param_truth)

# # %%
# xsim.plot()
