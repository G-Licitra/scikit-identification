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
        self.nx = states.shape[0]
        self.nu = None if inputs == None else inputs.shape[0]  # control input
        self.np = None if param == None else param.shape[0]  # model parameters
        self.ny = self.nx if output == None else len(output)  # model output

        self.__match_attributes(state_name, input_name, param_name, output_name)

        self.__check_attribute_consistency()

        self.__infer_model_type()

        # Construct Symbolic Dynamic Model
        if self.__model_type["struct"] == "f(x,u)":
            self.Fmodel = ca.Function(
                "model",
                [self.states, self.inputs],
                [self.model_dynamics, self.output],
                self.__model_type["model_input"],
                self.__model_type["model_output"],
            )
        elif self.__model_type["struct"] == "f(x)":
            self.Fmodel = ca.Function(
                "model",
                [self.states],
                [self.model_dynamics, self.output],
                self.__model_type["model_input"],
                self.__model_type["model_output"],
            )
        elif self.__model_type["struct"] == "f(x,p)":
            self.Fmodel = ca.Function(
                "model",
                [self.states, self.param],
                [self.model_dynamics, self.output],
                self.__model_type["model_input"],
                self.__model_type["model_output"],
            )
        elif self.__model_type["struct"] == "f(x,u,p)":
            self.Fmodel = ca.Function(
                "model",
                [self.states, self.inputs, self.param],
                [self.model_dynamics, self.output],
                self.__model_type["model_input"],
                self.__model_type["model_output"],
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

    def __match_attributes(self, state_name, input_name, param_name, output_name):
        """Assign names, if specified"""
        self.state_name = (
            ["x" + str(i + 1) for i in range(self.nx)]
            if state_name is None
            else state_name
        )

        if self.nu != None:
            self.input_name = (
                ["u" + str(i + 1) for i in range(self.nu)]
                if input_name is None
                else input_name
            )
        else:
            self.input_name = None

        if self.np != None:
            self.param_name = (
                ["p" + str(i + 1) for i in range(self.np)]
                if param_name is None
                else param_name
            )
        else:
            self.param_name = None

        self.output_name = (
            ["y" + str(i + 1) for i in range(self.ny)]
            if output_name is None
            else output_name
        )

    def __check_attribute_consistency(self):
        """Check if Input class are consistent"""
        if self.nx != self.model_dynamics.size()[0]:
            raise ValueError(
                "Input class is not consistent. states and model_dynamics must have the same dimension."
            )

        if self.nx != len(self.state_name):
            raise ValueError(
                "Input class is not consistent. state and state_name must have the same dimension."
            )

        if (self.nu is not None) and self.nu != len(self.input_name):
            raise ValueError(
                "Input class is not consistent. state and state_name must have the same dimension."
            )

        if (self.np is not None) and (self.np != len(self.param_name)):
            raise ValueError(
                "Input class is not consistent. param and param_name must have the same dimension."
            )

        if self.ny != len(self.output_name):
            raise ValueError(
                "Input class is not consistent. output and output_name must have the same dimension."
            )

    def __infer_model_type(self):
        """Infer which inputs the model receives."""

        # CASE parameters are not specified
        if self.nx is not None and self.nu is not None and self.np is None:
            self.__model_type = {
                "struct": "f(x,u)",
                "model_input": ["x(t)", "u(t)"],
                "model_output": ["xdot(t) = f(x(t), u(t))", "y(t) = g(x(t))"],
            }
        elif (
            self.nx is not None and self.nu is None and self.np is None
        ):  # CASE parameters AND input are not specified
            self.__model_type = {
                "struct": "f(x)",
                "model_input": ["x(t)"],
                "model_output": ["xdot(t) = f(x(t))", "y(t) = g(x(t))"],
            }
        elif (
            self.nx is not None and self.nu is None and self.np is not None
        ):  # CASE input are not specified
            self.__model_type = {
                "struct": "f(x,p)",
                "model_input": ["x(t)", "p"],
                "model_output": ["xdot(t) = f(x(t), p)", "y(t) = g(x(t))"],
            }
        else:  # CASE states, input and parameters are specified
            self.__model_type = {
                "struct": "f(x,u,p)",
                "model_input": ["x(t)", "u(t)", "p"],
                "model_output": ["xdot(t) = f(x(t), u(t), p)", "y(t) = g(x(t))"],
            }

    def evaluate(self, x=list[float], u=None, param=None):

        error_str = (
            "Input mishmatch. Please check that the inputs"
            "are consistent with class attributes."
        )

        """Evaluate numerically Model Mapping"""
        if self.__model_type["struct"] == "f(x,u)":
            if x is not None and u is not None and param is None:
                (rhs_num, y_num) = self.Fmodel(x, u)
            else:
                raise ValueError(error_str)

        if self.__model_type["struct"] == "f(x)":
            if x is not None and u is None and param is None:
                (rhs_num, y_num) = self.Fmodel(x)
            else:
                raise ValueError(error_str)

        if self.__model_type["struct"] == "f(x,p)":
            if x is not None and u is None and param is not None:
                (rhs_num, y_num) = self.Fmodel(x, param)
            else:
                raise ValueError(error_str)

        if self.__model_type["struct"] == "f(x,u,p)":
            if x is not None and u is not None and param is not None:
                (rhs_num, y_num) = self.Fmodel(x, u, param)
            else:
                raise ValueError(error_str)

        # Wrap values in pandas dataframe
        Xval = pd.DataFrame(data=rhs_num.full().T, columns=self.state_name)
        Yval = pd.DataFrame(data=y_num.full().T, columns=self.output_name)

        return (Xval, Yval)

    def __print_ode(self):
        print(self.model_dynamics)

    def __print_output(self):
        print(self.output)

    def __repr__(self):
        return self.model.print_dimensions()  # string to print


# class LTImodel(DynamicModel):

#     def __init__(
#         self,
#         A=None,
#         B=None,
#         C=None,
#         param=None,
#         state_name=None,
#         input_name=None,
#         param_name=None,
#         output_name=None,
#     ):


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
