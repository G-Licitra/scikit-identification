#%%
import warnings
from typing import List
from typing import Union

import casadi as ca
import numpy as np
import pandas as pd

from skmid.models import _infer_model_type
from skmid.models import DynamicModel
from skmid.models import generate_model_attributes


class RungeKutta4:
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

    def __init__(
        self, *, model: DynamicModel, fs: int = 1, n_steps_per_sample: int = 1
    ):

        # check input consistency
        if fs <= 0:
            raise ValueError("fs and n_steps_per_sample must be positive.")

        if n_steps_per_sample <= 0:
            raise ValueError("n_steps_per_sample must be positive integer.")

        dt = 1 / fs / n_steps_per_sample
        self.fs = fs  # frequency sample
        self.model = model

        x = model.state
        u = model.input
        p = model.parameter

        # infer model architecture
        nx = self.model._DynamicModel__nx
        nu = self.model._DynamicModel__nu
        np = self.model._DynamicModel__np

        self.__model_type = _infer_model_type(nx=nx, nu=nu, np=np)

        if self.__model_type["struct"] == "f(x,u)":
            # Build an integrator for this system: Runge Kutta 4 integrator
            k1 = model.model_function(x, u)
            k2 = model.model_function(x + dt / 2.0 * k1, u)
            k3 = model.model_function(x + dt / 2.0 * k2, u)
            k4 = model.model_function(x + dt * k3, u)

            xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Create a function that simulates one step propagation in a sample
            one_step_ahead = ca.Function("one_step_ahead", [x, u], [xf])

            X = x
            for i in range(n_steps_per_sample):
                X = one_step_ahead(X, u)

            # Create a function that simulates all step propagation on a sample
            one_sample_ahead = ca.Function(
                "one_sample_ahead",
                [x, u],
                [X],
                ["x[k]", "u[k]"],
                ["x[k+1] = x[k] + dt * f(x[k], u[k])"],
            )

        elif self.__model_type["struct"] == "f(x)":
            # Build an integrator for this system: Runge Kutta 4 integrator
            k1 = model.model_function(x)
            k2 = model.model_function(x + dt / 2.0 * k1)
            k3 = model.model_function(x + dt / 2.0 * k2)
            k4 = model.model_function(x + dt * k3)

            xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Create a function that simulates one step propagation in a sample
            one_step_ahead = ca.Function("one_step_ahead", [x], [xf])

            X = x
            for i in range(n_steps_per_sample):
                X = one_step_ahead(X)

            # Create a function that simulates all step propagation on a sample
            one_sample_ahead = ca.Function(
                "one_sample_ahead",
                [x],
                [X],
                ["x[k]"],
                ["x[k+1] = x[k] + dt * f(x[k])"],
            )

        elif self.__model_type["struct"] == "f(x,p)":
            # Build an integrator for this system: Runge Kutta 4 integrator
            k1 = model.model_function(x, p)
            k2 = model.model_function(x + dt / 2.0 * k1, p)
            k3 = model.model_function(x + dt / 2.0 * k2, p)
            k4 = model.model_function(x + dt * k3, p)

            xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Create a function that simulates one step propagation in a sample
            one_step_ahead = ca.Function("one_step_ahead", [x, p], [xf])

            X = x
            for i in range(n_steps_per_sample):
                X = one_step_ahead(X, p)

            # Create a function that simulates all step propagation on a sample
            one_sample_ahead = ca.Function(
                "one_sample_ahead",
                [x, p],
                [X],
                ["x[k]", "p"],
                ["x[k+1] = x[k] + dt * f(x[k], p)"],
            )

        elif self.__model_type["struct"] == "f(x,u,p)":
            # Build an integrator for this system: Runge Kutta 4 integrator
            k1 = model.model_function(x, u, p)
            k2 = model.model_function(x + dt / 2.0 * k1, u, p)
            k3 = model.model_function(x + dt / 2.0 * k2, u, p)
            k4 = model.model_function(x + dt * k3, u, p)

            xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Create a function that simulates one step propagation in a sample
            one_step_ahead = ca.Function("one_step_ahead", [x, u, p], [xf])

            X = x
            for i in range(n_steps_per_sample):
                X = one_step_ahead(X, u, p)

            # Create a function that simulates all step propagation on a sample
            one_sample_ahead = ca.Function(
                "one_sample_ahead",
                [x, u, p],
                [X],
                ["x[k]", "u[k]", "p"],
                ["x[k+1] = x[k] + dt * f(x[k], u[k], p)"],
            )
        else:
            raise ValueError("Model type not supported")

        self.__one_sample_ahead = one_sample_ahead

    def simulate(
        self,
        initial_condition: Union[List[str], None] = None,
        input=None,
        parameter=None,
        n_steps=100,
    ):

        if initial_condition is None:
            # default initial condition
            initial_condition = self.model._DynamicModel__nx * [0]
        elif len(initial_condition) != self.model._DynamicModel__nx:
            raise ValueError(
                "Initial condition must have the same dimension as the model"
            )

        if input is not None:

            n_steps = len(input)  # overwrite n_steps

            # time vector for simulation x_sim and y_sim
            self.time_ = np.linspace(
                start=0, stop=(n_steps) * (1 / self.fs), num=n_steps + 1
            )

            # if input is provided neglect n_steps prameter
            input = self.__validate_input(input)

        else:
            # time vector for simulation x_sim and y_sim
            self.time_ = np.linspace(
                start=0, stop=(n_steps) * (1 / self.fs), num=n_steps + 1
            )

        # Propagate simulation for N steps and generate trajectory
        k_sample_ahead = self.__one_sample_ahead.mapaccum("x_simulation", n_steps)

        if self.__model_type["struct"] == "f(x,u)":
            state_sim = k_sample_ahead(initial_condition, input.values.T)
        elif self.__model_type["struct"] == "f(x)":
            state_sim = k_sample_ahead(initial_condition)
        elif self.__model_type["struct"] == "f(x,p)":
            state_sim = k_sample_ahead(
                initial_condition, ca.repmat(parameter, 1, n_steps)
            )
        elif self.__model_type["struct"] == "f(x,u,p)":
            state_sim = k_sample_ahead(
                initial_condition, input.values.T, ca.repmat(parameter, 1, n_steps)
            )

        # check if forward simulation is numeric or symbolic
        if isinstance(state_sim, ca.casadi.MX):

            # first check that all output are contained in
            # TODO: change the output_name to match state_name

            # output_propagation_map = self.output_map.map(N_steps)
            output_map = dict()

            for i in range(0, len(self.model.state_name)):
                if self.model.state_name[i] in self.model.output_name:
                    output_map[self.model.state_name[i]] = i

            self.state_sim_ = state_sim
            # output_propagation_map(state_sim)
            # self.output_sim_ = output_propagation_map(state_sim)
            # map state with output for N steps
            self.output_sim_ = None

        else:
            # forward simulation is numeric
            state_sim = np.array(state_sim)

            # check if forward simulation is diverged
            if np.isnan(state_sim).any():
                print(
                    "INFO: Forward simulation nan values, hence, it may have diverged."
                )

            # pack differential state x attaching initial condition x0
            self.state_sim_ = pd.DataFrame(
                data=np.concatenate(
                    [np.array(initial_condition).reshape(1, -1), state_sim.T]
                ),
                index=self.time_,
                columns=self.model.state_name,
            )

            # select available states as output
            self.output_sim_ = self.state_sim_.filter(items=self.model.output_name)

    def __validate_input(self, input):
        """determine the type of input"""

        # get time and attach one extra sample to time
        if isinstance(input, np.ndarray):
            # convert input to pandas dataframe
            input = pd.DataFrame(
                data=input, index=self.time_[:-1], columns=self.model.input_name
            )

        if isinstance(input, pd.DataFrame) or isinstance(
            input, pd.Series
        ):  # check if input is a dataframe

            if isinstance(input, pd.Series):  # check if input is a series
                input = input.to_frame()

            if np.mean(np.diff(input.index)) != 1 / self.fs:
                # Detect sample frequency mismatch
                input.index = self.time_[:-1]
                print(
                    f"""INFO:The input index has a different fs than specified in the object.
                                  The input index has been modified by using fs={self.fs} Hz.
                                """
                )

            if not (set(input.columns) == set(self.model.input_name)):
                # Detect input name mismatch
                print(
                    f"""INFO:The input names is not consistent with the model input.
                                        the simulation has considered the input as if they were {self.model.input_name}.
                                     """
                )
                input.columns = self.model.input_name

            return input


if __name__ == "__main__":  # when run for testing only

    import matplotlib.pyplot as plt
    from scipy.signal import chirp

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

    #%%----------------------------------------------------------------

    (x, u, param) = generate_model_attributes(
        state_size=2, input_size=2, parameter_size=2
    )

    # assign specific name
    x1, x2 = x[0], x[1]
    u1, u2 = u[0], u[1]
    ka, kb = param[0], param[1]

    param_truth = [0.1, 0.5]  # ca.DM([0.1, 0.5])

    # xdot = f(x,u,p) <==> rhs = f(x,u,p)
    rhs = [u1 - ka * x1, u1 * u2 / x1 - u1 * x2 / x1 - kb * x2]

    sys = DynamicModel(
        state=x,
        input=u,
        parameter=param,
        model_dynamics=rhs,
        input_name=["F", "T"],
        state_name=["p", "v"],
        parameter_name=["ka", "kb"],
        output=["p"],
    )

    # #%%
    rk4 = RungeKutta4(model=sys, fs=fs)

    # #%%
    _ = rk4.simulate(initial_condition=x0, input=df_input, parameter=param_truth)

    df_U = rk4.input_
    df_X = rk4.state_sim_
    df_Y = rk4.output_sim_

    df_U.plot(subplots=True)
    df_X.plot(subplots=True)
    df_Y.plot(subplots=False)
    plt.show()
