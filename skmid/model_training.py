from typing import List
from typing import Union

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel


def _gauss_newton(e, nlp, V):
    r"""Gauss-Newton solver`

    Parameters
    ---------
    e : casadi
        cost function
    nlp: dict
        Steps forward within one integration step
    V : casadi
        optimization variables

    Returns
    ---------
    solver : casadi
        nlp solver

    """

    # Use just-in-time compilation to speed up the evaluation
    if ca.Importer.has_plugin("clang"):
        print('just-in-time compilation with compiler= "clang" ')
        with_jit = True
        compiler = "clang"
    elif ca.Importer.has_plugin("shell"):
        print('just-in-time compilation with compiler= "shell" ')
        with_jit = True
        compiler = "shell"
    else:
        print(
            "WARNING; running without jit. This may result in very slow evaluation times"
        )
        with_jit = False
        compiler = ""

    J = ca.jacobian(e, V)
    H = ca.triu(ca.mtimes(J.T, J))
    sigma = ca.MX.sym("sigma")
    hessLag = ca.Function(
        "nlp_hess_l",
        {"x": V, "lam_f": sigma, "hess_gamma_x_x": sigma * H},
        ["x", "p", "lam_f", "lam_g"],
        ["hess_gamma_x_x"],
        dict(jit=with_jit, compiler=compiler),
    )

    return ca.nlpsol(
        "solver", "ipopt", nlp, dict(hess_lag=hessLag, jit=with_jit, compiler=compiler)
    )


class LeastSquaresRegression:
    def __init__(
        self,
        *,
        model: DynamicModel,
        fs: int = 1,
        n_steps_per_sample: int = 1,
        strategy: str = "multiple-shooting",
    ):

        self.model = model
        self.fs = fs  # frequency sample
        self.n_steps_per_sample = n_steps_per_sample
        self.strategy = strategy

        self.__check_parameter_class_consistency()

        self.integrator = RungeKutta4(
            model=model, fs=fs, n_steps_per_sample=n_steps_per_sample
        )

    def __check_parameter_class_consistency(self):

        if not isinstance(self.model, DynamicModel):
            raise ValueError(
                "model input is expected to receive as input DynamicModel object."
            )
        if self.fs <= 0:
            raise ValueError("fs and n_steps_per_sample must be positive.")

        if self.n_steps_per_sample <= 0:
            raise ValueError("n_steps_per_sample must be positive integer.")

    def __check_parameter_fit_method_consistency(
        self, U: Union[pd.DataFrame, pd.Series], Y: Union[pd.DataFrame, pd.Series]
    ):

        if not isinstance(U, (pd.DataFrame, pd.Series)):
            raise ValueError(
                "U input is expected to receive as input pandas.DataFrame or pandas.Series object."
            )
        if not isinstance(Y, (pd.DataFrame, pd.Series)):
            raise ValueError(
                "Y input is expected to receive as input pandas.DataFrame or pandas.Series object."
            )

        # cast pandas.Series to pandas.DataFrame
        if isinstance(U, pd.Series):
            U = U.to_frame()

        # cast pandas.Series to pandas.DataFrame
        if isinstance(Y, pd.Series):
            Y = Y.to_frame()

        # check dimention consistency between U and Y
        if (U is not None) and (len(U) + 1 != len(Y)):
            raise ValueError(
                f"Inconsistent Data size between Y and U. It is expected dim(Y)=N+1 and dim(U)=N. Currently dim(Y)={len(Y)} and dim(U)={len(U)}."
            )

        if len(self.model.output_name) != Y.shape[1]:
            raise ValueError(
                f"The number of columns of Y must be equal to the number of model output. Currently Y has {Y.shape[1]} columns while the model has {len(self.model.output_name)} output."
            )

        if self.__n_input != U.shape[1]:
            raise ValueError(
                f"The number of columns of U must be equal to the number of model input. Currently U has {U.shape[1]} columns while the model has {len(self.model.input_name)} input."
            )

        if self.model.parameter is None:
            raise ValueError(
                "Dynamic model has no parameter. Please set the parameter before fitting."
            )

        if (
            (self.__param_guess is not None)
            and (self.__param_guess is not list)
            and (len(self.__param_guess) != self.__n_param)
        ):
            raise ValueError(
                "param_guess is expected to be a list with lenght equal to model:parameter."
            )

        if (
            (self.__param_scale is not None)
            and (self.__param_scale is not list)
            and (len(self.__param_scale) != self.__n_param)
        ):
            raise ValueError(
                "param_scale is expected to be a list with lenght equal to model:parameter."
            )

        # TODO: add check to input name
        # match name between U and model:input_name
        U = U.filter(items=self.model.input_name)
        Y = Y.filter(items=self.model.output_name)

        # discard initial condition in Y and prepare U and Y for fitting
        Y = Y.values[1:]
        U = U.values.T

        # Prepare U and Y
        # U = U.values.T (1, 10000)
        # Y = Y.values[1:] (10000, 1) # discard initial condition

        return (U, Y)

    def fit(
        self,
        *,
        U: Union[pd.DataFrame, pd.Series],
        Y: Union[pd.DataFrame, pd.Series],
        param_guess: Union[List[str], None] = None,
        param_scale: Union[List[str], None] = None,
        state_guess: Union[List[str], None] = None,
    ):

        # Retrive number of model parameters and states
        self.__n_state = self.model._DynamicModel__nx
        self.__n_param = self.model._DynamicModel__np
        self.__n_input = self.model._DynamicModel__nu
        self.__n_output = len(self.model.output_name)
        self.__param_guess = param_guess
        self.__param_scale = param_scale

        (U, Y) = self.__check_parameter_fit_method_consistency(U, Y)

        self.__n_shootings = len(Y)  # equal to the time series length

        # state_guess = self.__construct_state_guess(state_guess)

        if state_guess is None and (self.__n_output == self.__n_state):
            # case full state available with no initial state guess
            state_guess = Y

        # __check_param_input(self, Y, U, )

        # Note:
        # dim(U) = (n_inputs, n_shootings): 'numpy.ndarray'
        # dim(X) = (n_states, n_shootings): 'casadi.casadi.MX'
        # dim(Y) = (n_shootings, n_outputs): 'numpy.ndarray'

        # Construct continuity condtion for multiple-shooting approach
        X = ca.MX.sym("X", self.__n_state, self.__n_shootings)
        Xn = self.integrator._RungeKutta4__one_sample_ahead.map(
            self.__n_shootings, "openmp"
        )(
            X,
            U,
            ca.repmat(self.model.parameter * self.__param_scale, 1, self.__n_shootings),
        )
        gaps = Xn[:, :-1] - X[:, 1:]

        # Construct cost function
        e = Y - Xn[0, :].T

        # stack all optimization variable into a vector
        V = ca.veccat(self.model.parameter, X)

        nlp = {"x": V, "f": 0.5 * ca.dot(e, e), "g": ca.vec(gaps)}

        # Multipleshooting allows for careful initialization
        # yd = np.diff(Y, axis=0) * self.fs
        # X_guess = ca.horzcat(Y, ca.vertcat(yd, yd[-1])).T
        X_guess = state_guess.T

        x0 = ca.veccat(self.__param_guess, X_guess)
        solver = _gauss_newton(e, nlp, V)
        sol = solver(x0=x0, lbg=0, ubg=0)

        # array of shape (n_features, ) or (n_targets, n_features)
        self.coef_ = np.squeeze(sol["x"][: self.__n_param].full())
        # TODO add index to dataframe
        self.model_fit_ = pd.DataFrame(
            data=sol["x"][self.__n_param :]
            .full()
            .reshape((self.__n_shootings, self.__n_state)),
            columns=self.model.state_name,
        )

    def predict(self, U, Y):
        pass

    def summary(self):
        pass


if __name__ == "__main__":  # when run for testing only

    import os
    import json

    from skmid.models import generate_model_attributes

    # Load data and model settings
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
        input_name=["u"],
        parameter_name=["M", "c", "k", "k_NL"],
        state_name=["y", "dy"],
        output=["y"],
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
    # state_guess = Y, ca.vertcat(yd, yd[-1])

    # create initial condition
    Yg = Y.values[1:]
    yd = np.diff(Yg, axis=0) * fs
    yd = np.concatenate([yd, yd[-1].reshape(1, -1)], axis=0)
    state_guess = np.concatenate([Yg, yd], axis=1)

    # X_guess = ca.horzcat(Y, ca.vertcat(yd, yd[-1])).T
    # (2, 10000)

    estimator.fit(
        U=U, Y=Y, param_guess=param_guess, param_scale=scale, state_guess=state_guess
    )
    param_est = estimator.coef_

    assert ca.norm_inf(param_est * scale - settings["param_truth"]) < 1e-8

    x_fit = estimator.model_fit_
