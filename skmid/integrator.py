#%%
import warnings

import casadi as ca
import numpy as np
import pandas as pd

from skmid.models import _infer_model_type
from skmid.models import DynamicModel


class RungeKutta4:
    def __init__(
        self, *, model: DynamicModel, fs: int = 1, n_steps_per_sample: int = 1
    ):

        dt = 1 / fs / n_steps_per_sample
        self.fs = fs  # frequency sample
        self.model = model

        state = model.states
        input = model.inputs
        param = model.param

        # infer model architecture
        # nx = len(model.state_name)
        # nu = len(model.input_name) if model.input_name is not None else None
        # np = len(model.param_name) if model.param_name is not None else None
        nx = self.model._DynamicModel__nx
        nu = self.model._DynamicModel__nu
        np = self.model._DynamicModel__np

        self.__model_type = _infer_model_type(nx=nx, nu=nu, np=np)

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

        self.__one_sample = one_sample


# class RungeKutta4:
#     r"""Create Explicit Runge Kutta order 4 integrator`

#     Parameters
#     ---------
#     ode : casadi.Function
#         Function which define the model via ODE, formally,
#         f: (x, u, theta) -> rhs.
#         where
#         rhs: right-hand side,
#         x: differential states,
#         u: input,
#         theta: model Parameters
#     N_steps_per_sample: int
#         Steps forward within one integration step
#     fs: int
#         frequency at which the system is sampled
#     N_steps: int
#         number of integration steps

#     Returns
#     ---------
#     one_sample : casadi.Function
#         One step forward model integrator
#     all_sample : casadi.Function
#         N steps forward model integrator

#     """

#     def __init__(
#         self, *, model: DynamicModel, fs: int = 1, n_steps_per_sample: int = 1
#     ):

#         __dt = 1 / fs / n_steps_per_sample
#         self.fs = fs  # frequency sample
#         self.model = model

#         x = model.states
#         u = model.inputs
#         param = model.param

#         # infer model architecture
#         nx = len(model.state_name)
#         nu = len(model.input_name) if model.input_name is not None else None
#         np = len(model.param_name) if model.param_name is not None else None
#         self.__model_type = _infer_model_type(nx=nx, nu=nu, np=np)

#         if self.__model_type["struct"] == "f(x,u)":
#             # Build an RK4 integrator with param as symbolic variable
#             (k1, _) = self.model.model_function(x, u)
#             (k2, _) = self.model.model_function(x + __dt / 2.0 * k1, u)
#             (k3, _) = self.model.model_function(x + __dt / 2.0 * k2, u)
#             (k4, _) = self.model.model_function(x + __dt * k3, u)
#             xf = x + __dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

#             # Create a function that simulates one step propagation in a sample
#             one_step_ahead = ca.Function("one_step", [x, u], [xf])

#             # Carry out forward simulation within the entire sample time
#             X = x
#             for i in range(n_steps_per_sample):
#                 X = one_step_ahead(X, u)

#             # Create a function that simulates all step propagation on a sample
#             self.one_step_ahead = ca.Function(
#                 "one_sample",
#                 [x, u],
#                 [X],
#                 ["x[k]", "u[k]"],
#                 ["x[k+1] = x[k] + __dt * f(x[k], u[k])"],
#             )

#         elif self.__model_type["struct"] == "f(x)":
#             # Build an RK4 integrator with param as symbolic variable
#             (k1, _) = self.model.model_function(x)
#             (k2, _) = self.model.model_function(x + __dt / 2.0 * k1)
#             (k3, _) = self.model.model_function(x + __dt / 2.0 * k2)
#             (k4, _) = self.model.model_function(x + __dt * k3)
#             xf = x + __dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

#             # Create a function that simulates one step propagation in a sample
#             one_step_ahead = ca.Function("one_step", [x], [xf])

#             # Carry out forward simulation within the entire sample time
#             X = x
#             for i in range(n_steps_per_sample):
#                 X = one_step_ahead(X)

#             # Create a function that simulates all step propagation on a sample
#             self.one_step_ahead = ca.Function(
#                 "one_sample", [x], [X], ["x[k]"], ["x[k+1] = x[k] + __dt * f(x[k])"]
#             )

#         elif self.__model_type["struct"] == "f(x,p)":
#             # Build an RK4 integrator with param as symbolic variable
#             (k1, _) = self.model.model_function(x, param)
#             (k2, _) = self.model.model_function(x + __dt / 2.0 * k1, param)
#             (k3, _) = self.model.model_function(x + __dt / 2.0 * k2, param)
#             (k4, _) = self.model.model_function(x + __dt * k3, param)
#             xf = x + __dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

#             # Create a function that simulates one step propagation in a sample
#             one_step_ahead = ca.Function("one_step_ahead", [x, param], [xf])

#             # Carry out forward simulation within the entire sample time
#             X = x
#             for i in range(n_steps_per_sample):
#                 X = one_step_ahead(X, param)

#             # Create a function that simulates all step propagation on a sample
#             self.one_step_ahead = ca.Function(
#                 "one_sample",
#                 [x, param],
#                 [X],
#                 ["x[k]", "param"],
#                 ["x[k+1] = x[k] + __dt * f(x[k], param)"],
#             )

#         elif self.__model_type["struct"] == "f(x,u,p)":
#             # Build an RK4 integrator with param as symbolic variable
#             (k1, _) = self.model.model_function(x, u, param)
#             (k2, _) = self.model.model_function(x + __dt / 2.0 * k1, u, param)
#             (k3, _) = self.model.model_function(x + __dt / 2.0 * k2, u, param)
#             (k4, _) = self.model.model_function(x + __dt * k3, u, param)
#             xf = x + __dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

#             # Create a function that simulates one step propagation in a sample
#             one_step_ahead = ca.Function("one_step", [x, u, param], [xf])

#             # Carry out forward simulation within the entire sample time
#             X = x
#             for i in range(n_steps_per_sample):
#                 X = one_step_ahead(X, u, param)

#             # Create a function that simulates all step propagation on a sample
#             self.one_step_ahead = ca.Function(
#                 "one_sample",
#                 [x, u, param],
#                 [X],
#                 ["x[k]", "u[k]", "param"],
#                 ["x[k+1] = x[k] + __dt * f(x[k], u[k], param)"],
#             )
#         else:
#             raise ValueError("Model type not supported")

#         # geate output map y = g(x)
#         self.output_map = ca.Function("output_map", [x], [self.model.output])

#     def simulate(self, *, x0, input=None, param=None, N_steps=100):

#         if input is not None:

#             N_steps = len(input)  # overwrite N_steps

#             # time vector for simulation x_sim and y_sim
#             self.time_ = np.linspace(
#                 start=0, stop=(N_steps) * (1 / self.fs), num=N_steps + 1
#             )

#             # if input is provided neglect n_steps prameter
#             input = self.__validate_input(input)
#             self.input_ = input

#         else:
#             # time vector for simulation x_sim and y_sim
#             self.time_ = np.linspace(
#                 start=0, stop=(N_steps) * (1 / self.fs), num=N_steps + 1
#             )

#         # Propagate simulation for N steps and generate trajectory
#         k_step_ahead = self.one_step_ahead.mapaccum("x_simulation", N_steps)

#         if self.__model_type["struct"] == "f(x,u)":
#             x_sim = k_step_ahead(x0, input.values.T)
#         elif self.__model_type["struct"] == "f(x)":
#             x_sim = k_step_ahead(x0)
#         elif self.__model_type["struct"] == "f(x,p)":
#             x_sim = k_step_ahead(x0, ca.repmat(param, 1, N_steps))
#         elif self.__model_type["struct"] == "f(x,u,p)":
#             x_sim = k_step_ahead(x0, input.values.T, ca.repmat(param, 1, N_steps))

#         # check if forward simulation is numeric or symbolic
#         if isinstance(x_sim, ca.casadi.MX):

#             # first check that all output are contained in
#             # TODO: change the output_name to match state_name

#             # output_propagation_map = self.output_map.map(N_steps)
#             output_map = dict()

#             for i in range(0, len(self.model.state_name)):
#                 if self.model.state_name[i] in self.model.output_name:
#                     output_map[self.model.state_name[i]] = i

#             self.state_sim_ = x_sim
#             # output_propagation_map(x_sim)
#             # self.output_sim_ = output_propagation_map(x_sim)
#             # map state with output for N steps
#             self.output_sim_ = None

#         else:
#             # forward simulation is numeric
#             x_sim = np.array(x_sim)
#             # map state with output for N_steps + 1 (include x0 mapping)
#             output_propagation_map = self.output_map.map(N_steps + 1)

#             # pack differential state x attaching initial condition x0
#             self.state_sim_ = pd.DataFrame(
#                 data=np.concatenate([np.array(x0).reshape(1, -1), x_sim.T]),
#                 index=self.time_,
#                 columns=self.model.state_name,
#             )

#             # pack output in dataframe
#             self.output_sim_ = pd.DataFrame(
#                 data=output_propagation_map(self.state_sim_.values.T).full().T,
#                 index=self.time_,
#                 columns=self.model.output_name,
#             )

#     def __validate_input(self, input):
#         """determine the type of input"""

#         # get time and attach one extra sample to time
#         if isinstance(input, np.ndarray):
#             # convert input to pandas dataframe
#             input = pd.DataFrame(
#                 data=input, index=self.time_[:-1], columns=self.model.input_name
#             )

#         if isinstance(input, pd.DataFrame):  # check if input is a dataframe

#             if isinstance(input, pd.Series):  # check if input is a series
#                 input = input.to_frame()

#             if np.mean(np.diff(input.index)) != 1 / self.fs:
#                 # Detect sample frequency mismatch
#                 input.index = self.time_[:-1]
#                 warnings.warn(
#                     f"""The input index has a different fs than specified in the object.
#                                   The input index has been modified by using fs={self.fs} Hz.
#                                 """
#                 )

#             if not (set(input.columns) == set(self.model.input_name)):
#                 # Detect input name mismatch
#                 warnings.warn(
#                     f"""The input names is not consistent with the model input.
#                                         the simulation has considered the input as if they were {self.model.input_name}.
#                                      """
#                 )
#                 input.columns = self.model.input_name

#             return input


# if __name__ == "__main__":  # when run for testing only

#     from scipy.signal import chirp
#     from skmid.models import generate_model_parameters, DynamicModel

#     # Choose an excitation signal
#     np.random.seed(42)
#     N = 2000  # Number of samples
#     fs = 500  # Sampling frequency [hz]
#     t = np.linspace(0, (N - 1) * (1 / fs), N)
#     df_input = pd.DataFrame(
#         data={
#             "u1": 2 * chirp(t, f0=1, f1=10, t1=5, method="logarithmic"),
#             "u2": 2 * np.random.random(N),
#         },
#         index=t,
#     )

#     x0 = [1, -1]  # Initial Condition x0 = [0;0]; [nx = 2]

#     #%%----------------------------------------------------------------

#     (x, u, param) = generate_model_parameters(nstate=2, ninput=2, nparam=2)

#     # assign specific name
#     x1, x2 = x[0], x[1]
#     u1, u2 = u[0], u[1]
#     ka, kb = param[0], param[1]

#     param_truth = [0.1, 0.5]  # ca.DM([0.1, 0.5])

#     # xdot = f(x,u,p) <==> rhs = f(x,u,p)
#     rhs = [u1 - ka * x1, u1 * u2 / x1 - u1 * x2 / x1 - kb * x2]

#     sys = DynamicModel(states=x, inputs=u, param=param, model_dynamics=rhs)

#     # #%%
#     rk4 = RungeKutta4(model=sys, fs=fs)

#     # #%%
#     rk4.simulate(x0=x0, input=df_input, param=param_truth)

#     df_xsim = rk4.x_sim_

#     print(df_xsim.head())
