from typing import Any
from typing import Union

import casadi as ca
import pandas as pd


def generate_model_parameters(
    nx: int, nu: Union[int, None] = None, nparam: Union[int, None] = None
):
    r"""Generate casADi symbol parameters.

    Args:
        nx (int): The dimension of the differential state vector
        nu (int): The dimension of the control input vector
        nparam (Union[str, None], optional): parameter. The dimension of the parameter vector. Defaults to None.

    Returns:
        (x, u, param): the symbolic state, control input and parameter vector, respectively.

    Examples
    ----------
    >>> from skmid.models import generate_model_parameters
    >>> (x, u, param) = generate_model_parameters(nx=2, nu=2, nparam=2)
    """

    if nx == 0:
        raise ValueError("nx must be >= 1")
    else:
        x = ca.MX.sym("x", nx)

    u = None if (nu == None) or (nu == 0) else ca.MX.sym("u", nu)
    param = None if (nparam == None) or (nparam == 0) else ca.MX.sym("param", nparam)
    return (x, u, param)


class DynamicModel:
    r"""
    Casadi Class System
    Formulation:
    \dot(x(t)) = f(x(x), u(t), \theta)
    y(t) = g(f(x))

    with:
    - $x(t) \in \Reˆ{n_{x}}$ differential states
    - $u(t) \in \Reˆ{n_{u}}$ control inputs
    - $\theta \in \Reˆ{n_{\theta}}$ model parameters
    - $\dot(x(t)) \in \Reˆ{n_{x}}$ model dynamics defined as Ordinary Differential Equation (ODE)
    - $y(t) \in \Reˆ{n_{y}}$$ model output

    Parameters
    ---------
    input1: array-like
        explain input1
    input2: None or str (default: None)
        explain input2
    figsize: tuple (default: (11.7, 8.27))
        explain figsize

    Returns
    ---------
    p_plot: Figure

    Examples
    ---------

    >>> input1 = np.random.rand(100)
    >>> input2 = 2
    >>> p_plot = func(input1, input2)

    """

    # class variable: every instance will inherit this value
    # nationality = "italy"

    def __init__(
        self,
        states=list,
        inputs=None,
        param=None,
        model_dynamics=list,
        output=None,
        state_name=None,
        input_name=None,
        param_name=None,
        output_name=None,
    ):

        self.states = states
        self.inputs = inputs
        self.param = param
        self.model_dynamics = ca.vcat(model_dynamics)
        # if output y(t) is not specified, then set y = x(t)
        self.output = states if output == None else ca.vcat(output)

        # get dimentions
        self.nx = states.shape[0]  # differential states
        self.nu = None if input == None else inputs.shape[0]  # control input
        self.np = None if param == None else param.shape[0]  # model parameters
        self.ny = self.nx if output == None else output.shape[0]  # model parameters

        # assign names, if specified
        self.state_name = (
            ["x" + str(i + 1) for i in range(self.nx)]
            if state_name is None
            else state_name
        )
        self.input_name = (
            ["u" + str(i + 1) for i in range(self.nu)]
            if input_name is None
            else input_name
        )
        self.param_name = (
            ["p" + str(i + 1) for i in range(self.np)]
            if param_name is None
            else param_name
        )
        self.output_name = (
            ["y" + str(i + 1) for i in range(self.ny)]
            if output_name is None
            else output_name
        )

        self.__check_input_consistency()

        # Construct Symbolic Function RHS

        # Form an ode function
        if self.np == None:  # CASE parameters are not specified
            self.Fmodel = ca.Function(
                "model",
                [self.states, self.inputs],
                [self.model_dynamics, self.output],
                ["x(y)", "u(t)"],
                ["xdot(t) = f(x(t),u(t))", "y(t) = g(x(t))"],
            )
        elif (
            self.np == None and self.nu == None
        ):  # CASE parameters AND input are not specified
            self.Fmodel = ca.Function(
                "model",
                [self.states],
                [self.model_dynamics, self.output],
                ["x(y)"],
                ["xdot(t) = f(x(t))", "y(t) = g(x(t))"],
            )
        else:  # CASE states, input and parameters are specified
            self.Fmodel = ca.Function(
                "model",
                [self.states, self.inputs, self.param],
                [self.model_dynamics, self.output],
                ["x(y)", "u(t)", "p"],
                ["xdot(t) = f(x(t),u(t),p)", "y(t) = g(x(t))"],
            )

    def print_summary(self):
        """Print info about"""
        print("Input Summary\n-----------------")
        print(f"states    = {self.state_name}")
        print(f"inputs    = {self.input_name}")
        print(f"parameter = {self.param_name}")
        print(f"output    = {self.output_name}")
        print("\nDimension Summary\n-----------------")
        self.Fmodel.print_dimensions()

    def __check_input_consistency(self):
        """Check if Input class are consistent"""
        if self.nx != self.model_dynamics.size()[0]:
            raise ValueError(
                "Input class is not consistent. states and model_dynamics must have the same dimension."
            )
        elif self.nx != len(self.state_name):
            raise ValueError(
                "Input class is not consistent. state and state_name must have the same dimension."
            )
        elif self.nu != len(self.input_name):
            raise ValueError(
                "Input class is not consistent. state and state_name must have the same dimension."
            )
        elif self.np != len(self.param_name):
            raise ValueError(
                "Input class is not consistent. param and param_name must have the same dimension."
            )
        elif self.ny != len(self.output_name):
            raise ValueError(
                "Input class is not consistent. output and output_name must have the same dimension."
            )

    def evaluate(self, x0, u, param):
        # TODO: add case with not param
        (rhs_num, y_num) = self.Fmodel(x0, u0, param)
        return (rhs_num, y_num)

    def _validate_params(self):
        """Validate input params."""
        return None

    def print_ode(self):
        print(self.model_dynamics)

    def print_output(self):
        print(self.output)

    def __repr__(self):
        return self.model.print_dimensions()  # string to print


if __name__ == "__main__":  # when run for testing only

    (x, u, param) = generate_model_parameters(nx=2, nu=2, nparam=2)

    # assign specific name
    x1, x2 = x[0], x[1]
    u1, u2 = u[0], u[1]
    ka, kb = param[0], param[1]

    # xdot = f(x,u,p) <==> rhs = f(x,u,p)
    rhs = [u1 - ka * x1, u1 * u2 / x1 - u1 * x2 / x1 - kb * x2]

    #%%
    sys = DynamicModel(states=x, inputs=u, param=param, model_dynamics=rhs)

    # sys.print_dimensions()
    # sys.print_ode()

    # numerical evaluation ===================================================
    x_test = [0.1, -0.1]
    u_test = [0.2, -0.1]
    theta_test = [0.1, 0.5]
    rhs_num, y_num = sys.evaluate(x_test, u_test, theta_test)

    print(f"rhs = {rhs_num}, \ny = {y_num}")

    # rhs = [0.19, 0.05],
    # y = [0.1, -0.1]
