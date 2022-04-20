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
        strategy: str = "multiple-shooting"
    ):
        self.model = model
        self.fs = fs  # frequency sample
        self.n_steps_per_sample = n_steps_per_sample
        self.strategy = strategy

        self.integrator = RungeKutta4(
            model=model, fs=fs, n_steps_per_sample=n_steps_per_sample
        )

    def fit(self, *, U=None, Y, param_guess=None, param_scale=None):

        # Prepare U and Y
        U = U.values.T
        Y = Y.values[1:]  # discard initial condition

        # Retrive number of model parameters and states
        n_states = self.model._DynamicModel__nx
        n_param = self.model._DynamicModel__np
        n_shootings = len(Y)  # equal to the time series length

        # Note:
        # dim(U) = (n_inputs, n_shootings): 'numpy.ndarray'
        # dim(X) = (n_states, n_shootings): 'casadi.casadi.MX'
        # dim(Y) = (n_shootings, n_outputs): 'numpy.ndarray'

        # Construct continuity condtion for multiple-shooting approach
        X = ca.MX.sym("X", n_states, n_shootings)
        Xn = self.integrator._RungeKutta4__one_sample_ahead.map(n_shootings, "openmp")(
            X, U, ca.repmat(self.model.parameter * param_scale, 1, n_shootings)
        )
        gaps = Xn[:, :-1] - X[:, 1:]

        # Construct cost function
        e = Y - Xn[0, :].T

        # stack all optimization variable into a vector
        V = ca.veccat(self.model.parameter, X)

        nlp = {"x": V, "f": 0.5 * ca.dot(e, e), "g": ca.vec(gaps)}

        # Multipleshooting allows for careful initialization
        yd = np.diff(Y, axis=0) * self.fs
        X_guess = ca.horzcat(Y, ca.vertcat(yd, yd[-1])).T

        x0 = ca.veccat(param_guess, X_guess)
        solver = _gauss_newton(e, nlp, V)
        sol = solver(x0=x0, lbg=0, ubg=0)

        # array of shape (n_features, ) or (n_targets, n_features)
        self.coef_ = np.squeeze(sol["x"][:n_param].full())
        # TODO add index to dataframe
        self.model_fit_ = pd.DataFrame(
            data=sol["x"][n_param:].full().reshape((n_shootings, n_states)),
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
