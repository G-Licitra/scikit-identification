from typing import Any
from typing import Union

import casadi as ca
import pandas as pd


def generate_model_parameters(nx: int, nu: int, nparam: Union[str, None] = None):
    """Generate casADi symbol parameters.

    Args:
        nx (int): The dimension of the differential state vector
        nu (int): The dimension of the control input vector
        nparam (Union[str, None], optional): parameter. The dimension of the parameter vector. Defaults to None.

    Returns:
        (x, u, param): the symbolic state, control input and parameter vector, respectively.
    """
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    param = None if nparam == None else ca.MX.sym("param", nparam)  # model parameters
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
        states,
        inputs,
        param=None,
        model_dynamics=None,
        output=None,
        name_state=None,
        name_input=None,
        name_output=None,
    ):

        """Constructor. It runs every instance is created"""
        self.states = states
        self.inputs = inputs
        self.param = param

        # get dimentions
        self.nx = states.shape[0]  # differential states
        self.nu = inputs.shape[0]  # control input
        self.np = None if param == None else param.shape[0]  # model parameters

        self.model_dynamics = ca.vcat(model_dynamics)

        # if output y(t) is not specified, then set y = x(t)
        self.output = states if output == None else ca.vcat(output)
        self.ny = self.nx if output == None else output.shape[0]  # model parameters

        # Construct Model
        # Form an ode function
        print(self.np)
        if self.np == None:
            self.Fmodel = ca.Function(
                "model",
                [self.states, self.inputs],
                [self.model_dynamics, self.output],
                ["x(y)", "u(t)"],
                ["xdot(t) = f(x(t),u(t))", "y(t) = g(x(t))"],
            )
        else:
            self.Fmodel = ca.Function(
                "model",
                [self.states, self.inputs, self.param],
                [self.model_dynamics, self.output],
                ["x(y)", "u(t)", "theta"],
                ["xdot(t) = f(x(t),u(t),theta)", "y(t) = g(x(t))"],
            )

    def _validate_params(self):
        """Validate input params."""
        return None

    def print_dimensions(self):
        self.Fmodel.print_dimensions()

    def print_ode(self):
        print(self.model_dynamics)

    def print_output(self):
        print(self.output)

    def evaluate(self, x0, u0, param):
        # TODO: add case with not param
        (rhs_num, y_num) = self.Fmodel(x0, u0, param)
        return (rhs_num, y_num)

    def __repr__(self):
        f"""
        Dynamic Model with:
        - states:{self.nx} = [x1, x2]
        - inputs:{self.nu} = [u1, u2]
        - parameter:{self.param} = [p1,p2]
        """

        return "self.model.print_dimensions()"  # string to print


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
    sys.print_dimensions()
    sys.print_ode()

    # numerical evaluation ===================================================
    x_test = [0.1, -0.1]
    u_test = [0.2, -0.1]
    theta_test = [0.1, 0.5]
    rhs_num, y_num = sys.evaluate(x_test, u_test, theta_test)

    print(f"rhs = {rhs_num}, \ny = {y_num}")

    # rhs = [0.19, 0.05],
    # y = [0.1, -0.1]
