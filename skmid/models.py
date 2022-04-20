from typing import Any
from typing import Union
from typing import List

import casadi as ca
import pandas as pd


def _infer_model_type(nx: Union[int, None], nu: Union[int, None], np: Union[int, None]):
    """Infer which inputs the model receives."""

    # CASE parameters are not specified
    if nx is not None and nu is not None and np is None:
        _model_type = {
            "struct": "f(x,u)",
            "model_input": ["x(t)", "u(t)"],
            "model_output": ["xdot(t) = f(x(t), u(t))"],
        }
    elif (
        nx is not None and nu is None and np is None
    ):  # CASE parameters AND input are not specified
        _model_type = {
            "struct": "f(x)",
            "model_input": ["x(t)"],
            "model_output": ["xdot(t) = f(x(t))"],
        }
    elif (
        nx is not None and nu is None and np is not None
    ):  # CASE input are not specified
        _model_type = {
            "struct": "f(x,p)",
            "model_input": ["x(t)", "p"],
            "model_output": ["xdot(t) = f(x(t), p)"],
        }
    else:  # CASE states, input and parameters are specified
        _model_type = {
            "struct": "f(x,u,p)",
            "model_input": ["x(t)", "u(t)", "p"],
            "model_output": ["xdot(t) = f(x(t), u(t), p)"],
        }
    return _model_type


def generate_model_attributes(
    state_size: int,
    input_size: Union[int, None] = None,
    parameter_size: Union[int, None] = None,
):
    """Generate casADi symbol parameters.

    Args:
        state_size (int): The dimension of the differential state vector
        input_size (Union[int, None], optional): The dimension of the control input vector
        nparam (Union[int, None], optional): parameter. The dimension of the parameter vector. Defaults to None.

    Returns:
        (state, input, parameter): the symbolic state, control input and parameter vector, respectively.

    Examples
    ----------
    >>> from skmid.models import generate_model_attributes
    >>> (state, input, parameter) = generate_model_attributes(nx=2, nu=2, nparam=2)
    """

    if state_size == 0:
        raise ValueError("state_size must be >= 1")
    else:
        state = ca.MX.sym("x", state_size)

    input = (
        None
        if (input_size == None) or (input_size == 0)
        else ca.MX.sym("u", input_size)
    )
    parameter = (
        None
        if (parameter_size == None) or (parameter_size == 0)
        else ca.MX.sym("p", parameter_size)
    )
    return (state, input, parameter)


class DynamicModel:
    """
    Declaration of Symbolic Dynamic Model formulated as:

    \dot(x(t)) = f(x(x), u(t), p)
    y(t) = g(f(x))

    with:
    - $x(t) \in \Re^{n_{x}}$ differential states
    - $u(t) \in \Re^{n_{u}}$ control inputs
    - $p \in \Re^{n_{p}}$ model parameters
    - $\dot(x(t)) \in \Re^{n_{x}}$ model dynamics defined as Ordinary Differential Equation (ODE)
    - $y(t) \in \Re^{n_{y}}$$ model output

    Parameters
    ----------
    states : list[casadi.MX]
        Symbolic differential states $x(t) \in \Re^{n_{x}}$.
    inputs : casadi.MX, default=None
        Symbolic control inputs $u(t) \in \Re^{n_{u}}$.
    param : casadi.MX, default=True
        Symbolic model parameters $p \in \Re^{n_{p}}$.
    model_dynamics : list[casadi.MX]
        List of symbolic equations which define the model dynamics.
    output : casadi.MX, default=None
        List of symbolic equations which define the model output. If not specicient output=states, i.e. y(t)=x(t)
    state_name : list[str], default=None
        Differential state labels, defaulting to [x1, x2, ..., x_{n_{x}}].
    input_name : list[str], default=None
        Input labels, defaulting to [u1, u2, ..., x_{n_{u}}].
    param_name : list[str], default=None
        Parameter labels, defaulting to [p1, p2, ..., p_{n_{p}}].
    output_name : list[str], default=None
        output labels, defaulting to [y1, y2, ..., y_{n_{y}}].

    See Also
    --------
    DynamicModel.print_summary : Print info about model inputs, output and their corresponding dimension.
    DynamicModel.evaluate : Numerical evaludation of the model.


    Examples
    --------
    Construct simple dynamic model with one differential state, one input.

    >>> (x, u, _) = generate_model_attributes(nx=1, nu=1)
    >>> model = DynamicModel(states=x, inputs=u, model_dynamics=[2*x**2 + u])
    >>> model.print_summary()
    Input Summary
    -----------------
    states    = ['x1']
    inputs    = ['u1']
    parameter = None
    output    = ['y1']

    Dimension Summary
    -----------------
    Number of inputs: 2
    Input 0 ("x(t)"): 1x1
    Input 1 ("u(t)"): 1x1
    Number of outputs: 2
    Output 0 ("xdot(t) = f(x(t), u(t))"): 1x1
    Output 1 ("y(t) = g(x(t))"): 1x1

    Note that output is equal to states when not specified.

    Construct Lorenz system (more info [here](https://en.wikipedia.org/wiki/Lorenz_system).

    >>> (states, inputs, param) = generate_model_attributes(nx=3, nu=3, nparam=3)

    Define sub-variables for better readibility of the equation

    >>> (x,y,z) = states[0], states[1], states[2]
    >>> (sigma, rho, beta) = param[0], param[1], param[2]
    >>> model = DynamicModel(
    ...        states=states,
    ...        param=param,
    ...        model_dynamics=model_dynamics,
    ...        state_name = ['x', 'y', 'z'], # ensure the correct order
    ...        param_name = ['sigma', 'rho', 'beta'])

    >>> model.print_summary()
    Input Summary
    -----------------
    states    = ['x', 'y', 'z']
    inputs    = None
    parameter = ['sigma', 'rho', 'beta']
    output    = ['y1', 'y2', 'y3']

    Dimension Summary
    -----------------
    Number of inputs: 2
    Input 0 ("x(t)"): 3x1
    Input 1 ("p"): 3x1
    Number of outputs: 2
    Output 0 ("xdot(t) = f(x(t), p)"): 3x1
    Output 1 ("y(t) = g(x(t))"): 3x1

    Evalute function. Lorenz used the following parameter sigma=10, rho=8/3, beta=28.
    The x would represent the initial condition set at x0=0.0, y0=40.0, z0=0.01
    >>> (Xdot_val, Y_val) = model.evaluate(state_num=[0.0, 40.0, 0.01], param_num=[10, 8/3, 28])
    >>> Xdot_val
        x     y     z
    0  400.0 -40.0 -0.28

    >>> Y_val
        y1    y2    y3
    0  0.0  40.0  0.01

    """

    # TODO adjust input
    def __init__(
        self,
        *,
        state=List[ca.casadi.MX],
        input: Union[List[ca.casadi.MX], None] = None,
        parameter: Union[List[ca.casadi.MX], None] = None,
        model_dynamics=List[ca.casadi.MX],
        output: Union[List[str], None] = None,
        state_name: Union[List[str], None] = None,
        input_name: Union[List[str], None] = None,
        parameter_name: Union[List[str], None] = None,
    ):

        self.state = state
        self.input = input
        self.parameter = parameter
        self.output = output

        if isinstance(model_dynamics, list):
            self.model_dynamics = ca.vcat(model_dynamics)
        else:
            raise ValueError("model_dynamics must be a list of casadi.MX")

        # get dimentions
        self.__nx = state.shape[0]  # different states
        self.__nu = None if input == None else input.shape[0]  # control input
        self.__np = (
            None if parameter == None else parameter.shape[0]
        )  # model parameters

        self.__match_attributes(state_name, input_name, parameter_name)

        self.__check_attribute_consistency()

        self.__validate_output()

        _model_type = _infer_model_type(nx=self.__nx, nu=self.__nu, np=self.__np)

        # Construct Symbolic Dynamic Model
        if _model_type["struct"] == "f(x,u)":
            self.model_function = ca.Function(
                "model",
                [self.state, self.input],
                [self.model_dynamics],
                _model_type["model_input"],
                _model_type["model_output"],
            )
        elif _model_type["struct"] == "f(x)":
            self.model_function = ca.Function(
                "model",
                [self.state],
                [self.model_dynamics],
                _model_type["model_input"],
                _model_type["model_output"],
            )
        elif _model_type["struct"] == "f(x,p)":
            self.model_function = ca.Function(
                "model",
                [self.state, self.parameter],
                [self.model_dynamics],
                _model_type["model_input"],
                _model_type["model_output"],
            )
        elif _model_type["struct"] == "f(x,u,p)":
            self.model_function = ca.Function(
                "model",
                [self.state, self.input, self.parameter],
                [self.model_dynamics],
                _model_type["model_input"],
                _model_type["model_output"],
            )

    def print_summary(self):
        """Print info about model inputs, output and their corresponding dimension."""
        print("Input Summary\n-----------------")
        print(f"states    = {self.state_name}")
        print(f"inputs    = {self.input_name}")
        print(f"parameter = {self.param_name}")
        print(f"output    = {self.output_name}")
        print("\nDimension Summary\n-----------------")
        self.model_function.print_dimensions()

    def evaluate(self, *, state_num=List[float], input_num=None, parameter_num=None):
        """Numerical evaludation of the model."""

        error_str = """Input mishmatch. Please check that the inputs are consistent with class attributes."""

        _model_type = _infer_model_type(nx=self.__nx, nu=self.__nu, np=self.__np)

        if _model_type["struct"] == "f(x,u)":
            if (
                state_num is not None
                and input_num is not None
                and parameter_num is None
            ):
                rhs_num = self.model_function(state_num, input_num)
            else:
                raise ValueError(error_str)

        if _model_type["struct"] == "f(x)":
            if state_num is not None and input_num is None and parameter_num is None:
                rhs_num = self.model_function(state_num)
            else:
                raise ValueError(error_str)

        if _model_type["struct"] == "f(x,p)":
            if (
                state_num is not None
                and input_num is None
                and parameter_num is not None
            ):
                rhs_num = self.model_function(state_num, parameter_num)
            else:
                raise ValueError(error_str)

        if _model_type["struct"] == "f(x,u,p)":
            if (
                state_num is not None
                and input_num is not None
                and parameter_num is not None
            ):
                rhs_num = self.model_function(state_num, input_num, parameter_num)
            else:
                raise ValueError(error_str)

        # Wrap values in pandas dataframe
        model_dynamics_num = pd.DataFrame(
            data=rhs_num.full().T, columns=self.state_name
        )

        return model_dynamics_num

    def __match_attributes(self, state_name, input_name, param_name):
        """Assign names to attributes, if specified"""
        self.state_name = (
            ["x" + str(i + 1) for i in range(self.__nx)]
            if state_name is None
            else state_name
        )

        if self.__nu is not None:
            self.input_name = (
                ["u" + str(i + 1) for i in range(self.__nu)]
                if input_name is None
                else input_name
            )
        else:
            self.input_name = None

        if self.__np is not None:
            self.param_name = (
                ["p" + str(i + 1) for i in range(self.__np)]
                if param_name is None
                else param_name
            )
        else:
            self.param_name = None

    def __check_attribute_consistency(self):
        """Check if Input class are consistent"""
        if self.__nx != self.model_dynamics.size()[0]:
            # case dim(states) != dim(rhs)
            raise ValueError(
                "Input class is not consistent. states and model_dynamics must have the same dimension."
            )

        if self.__nx != len(self.state_name):
            raise ValueError(
                "Input class is not consistent. state and state_name must have the same dimension."
            )

        if (self.__nu is not None) and self.__nu != len(self.input_name):
            raise ValueError(
                "Input class is not consistent. state and state_name must have the same dimension."
            )

        if (self.__np is not None) and (self.__np != len(self.param_name)):
            raise ValueError(
                "Input class is not consistent. param and param_name must have the same dimension."
            )

    def __validate_output(self):

        if self.output is None:
            # full-state available at output
            self.output_name = self.state_name
        else:
            # check if element in output list is contained in state_name list
            n = len(self.output)
            res = any(
                self.output == self.state_name[i : i + n]
                for i in range(len(self.state_name) - n + 1)
            )

            if res:
                # All output elements are part of state verctor
                self.output_name = self.output
            else:
                message_error = f"""the following element are defined in output but not in the state vector: {list(set(self.state_name) - set(self.output))}. Output can be either the full state vector or a subset of the state vector.
                isf state_name has not been specified, please specify output=['x1', 'x2',..., 'xn'] where n is the number of states."""

                raise ValueError(message_error)


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

    (x, u, param) = generate_model_attributes(
        state_size=2, input_size=2, parameter_size=2
    )

    # assign specific name
    x1, x2 = x[0], x[1]
    u1, u2 = u[0], u[1]
    ka, kb = param[0], param[1]

    # xdot = f(x,u,p) <==> rhs = f(x,u,p)
    rhs = [u1 - ka * x1, u1 * u2 / x1 - u1 * x2 / x1 - kb * x2]

    #%%
    sys = DynamicModel(
        state=x,
        input=u,
        parameter=param,
        model_dynamics=rhs,
        state_name=["p", "v"],
        parameter_name=["ka", "kb"],
        input_name=["Force", "Torque"],
        output=["p"],
    )

    # numerical evaluation ===================================================
    x_test = [0.1, -0.1]
    u_test = [0.2, -0.1]
    theta_test = [0.1, 0.5]

    sys.print_summary()

    rhs_num = sys.evaluate(state_num=x_test, input_num=u_test, parameter_num=theta_test)

    print(f"rhs = {rhs_num}")

    # rhs =       p     v
    #         0  0.19  0.05
