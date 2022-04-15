import casadi as ca
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skmid.models import DynamicModel
from skmid.models import generate_model_attributes


class TestDynamicModel:
    """Test class DynamicModel"""

    def test_instance_type(self):
        """Check if function returns a pandas Series."""
        (x, u, param) = generate_model_attributes(
            state_size=2, input_size=1, parameter_size=1
        )

        model = DynamicModel(
            states=x,
            inputs=u,
            param=param,
            model_dynamics=[param * x[0] + u, x[1]],
            output=[x**2],
        )

        assert isinstance(model, DynamicModel)

    def test_check_attribute_inconsistency(self):
        """Test whether error are raised when inputs are inconsistent."""

        (x, u, param) = generate_model_attributes(
            state_size=2, input_size=2, parameter_size=2
        )

        # assign specific name
        x1, x2 = x[0], x[1]
        u1, u2 = u[0], u[1]
        ka, kb = param[0], param[1]

        # xdot = f(x,u,p) <==> rhs = f(x,u,p)
        rhs = [u1 - ka * x1, u1 * u2 / x1 - u1 * x2 / x1 - kb * x2]

        # Case states and model_dynamics do not have the same dimension
        with pytest.raises(ValueError):
            model = DynamicModel(states=x1, inputs=u, param=param, model_dynamics=rhs)

        # Case state and state_name do not have the same dimension
        with pytest.raises(ValueError):
            model = DynamicModel(
                states=x,
                inputs=u,
                param=param,
                model_dynamics=rhs,
                state_name=["a", "b", "c"],
            )

        # Case input and input_name do not have the same dimension
        with pytest.raises(ValueError):
            model = DynamicModel(
                states=x, inputs=u, param=param, model_dynamics=rhs, input_name=["a"]
            )

        # Case param and param_name do not have the same dimension
        with pytest.raises(ValueError):
            model = DynamicModel(
                states=x,
                inputs=u,
                param=param,
                model_dynamics=rhs,
                param_name=["p1", "p2", "p3"],
            )

        # Case output and output_name do not have the same dimension
        with pytest.raises(ValueError):
            model = DynamicModel(
                states=x, inputs=u, param=param, model_dynamics=rhs, output_name=["y1"]
            )

        # Case model_dynamics is not a list
        with pytest.raises(ValueError):
            model = DynamicModel(
                states=x[0],
                inputs=u[0],
                param=param[0],
                model_dynamics=param[0] * x[0] ** 2 + u[0],
            )

        # Case output is not a list
        with pytest.raises(ValueError):
            model = DynamicModel(
                states=x[0],
                inputs=u[0],
                param=param[0],
                model_dynamics=[param[0] * x[0] ** 2 + u[0]],
                output=x[0] ** 2,
            )

    def test_output(self):
        """Test output under several case scenarios."""

        (x, u, param) = generate_model_attributes(
            state_size=2, input_size=2, parameter_size=2
        )

        # assign specific name
        x1, x2 = x[0], x[1]
        u1, u2 = u[0], u[1]
        ka, kb = param[0], param[1]

        # Case f(x, u)
        # ----------------------------------------------------------------
        model = DynamicModel(
            states=x,
            inputs=u,
            model_dynamics=[x1 + x2 + u1 + u2, -x1 - x2 - u1 - u2],
            output=[2 * (x1 + x2), x1, x2 / 2],
        )

        (Xval, Yval) = model.evaluate(state_num=[1, 1], input_num=[1, 1])

        Xtarget = pd.DataFrame(data={"x1": 4.0, "x2": -4.0}, index=[0])
        Ytarget = pd.DataFrame(data={"y1": 4.0, "y2": 1.0, "y3": 0.5}, index=[0])
        assert_frame_equal(Xval, Xtarget, check_dtype=False)
        assert_frame_equal(Yval, Ytarget, check_dtype=False)

        # case with naming state, input and output
        model = DynamicModel(
            states=x,
            inputs=u,
            model_dynamics=[x1 + x2 + u1 + u2, -x1 - x2 - u1 - u2],
            output=[2 * (x1 + x2), x1, x2 / 2],
            state_name=["p", "q"],
            input_name=["ux", "uy"],
            output_name=["a", "b", "c"],
        )

        (Xval, Yval) = model.evaluate(state_num=[-1, -1], input_num=[-1, -1])

        Xtarget = pd.DataFrame(data={"p": -4.0, "q": 4.0}, index=[0])
        Ytarget = pd.DataFrame(data={"a": -4.0, "b": -1.0, "c": -0.5}, index=[0])
        assert_frame_equal(Xval, Xtarget, check_dtype=False)
        assert_frame_equal(Yval, Ytarget, check_dtype=False)

        # Case f(x)
        # ----------------------------------------------------------------
        model = DynamicModel(
            states=x,
            model_dynamics=[x1 + x2**2, -np.log(x1) - x2],
            output=[2 * (x1 + x2) + np.sqrt(x1)],
            state_name=["wx", "wy"],
            output_name=["y(t)"],
        )

        x_init = [1, 2]
        (Xval, Yval) = model.evaluate(state_num=x_init)

        Xtarget = pd.DataFrame(
            data={
                "wx": x_init[0] + x_init[1] ** 2,
                "wy": -np.log(x_init[0]) - x_init[1],
            },
            index=[0],
        )
        Ytarget = pd.DataFrame(
            data={"y(t)": 2 * (x_init[0] + x_init[1]) + np.sqrt(x_init[0])}, index=[0]
        )
        assert_frame_equal(Xval, Xtarget, check_dtype=False)
        assert_frame_equal(Yval, Ytarget, check_dtype=False)

        # Case f(x,p)
        # ----------------------------------------------------------------
        model = DynamicModel(
            states=x,
            param=param,
            model_dynamics=[x1 + x2**ka, -np.log(x1) - x2],
            output=[kb * (x1 + x2) + np.sqrt(x1)],
            state_name=["wx", "wy"],
        )

        x_init = [1, 2]
        k_num = [2, 2]

        (Xval, Yval) = model.evaluate(state_num=x_init, param_num=k_num)

        Xtarget = pd.DataFrame(
            data={
                "wx": x_init[0] + x_init[1] ** k_num[0],
                "wy": -np.log(x_init[0]) - x_init[1],
            },
            index=[0],
        )
        Ytarget = pd.DataFrame(
            data={"y1": k_num[1] * (x_init[0] + x_init[1]) + np.sqrt(x_init[0])},
            index=[0],
        )
        assert_frame_equal(Xval, Xtarget, check_dtype=False)
        assert_frame_equal(Yval, Ytarget, check_dtype=False)

        # Case f(x,u,p)
        # ----------------------------------------------------------------
        model = DynamicModel(
            states=x,
            inputs=u,
            param=param,
            model_dynamics=[x1 + x2**ka + u1, -np.log(x1) - x2 + u1 * u2],
            output=[kb * (x1 + x2) + np.sqrt(x1)],
            input_name=["W", "V"],
        )

        x_init = [1, 2]
        k_num = [2, 2]
        u_init = [0.2, 5]

        (Xval, Yval) = model.evaluate(
            state_num=x_init, input_num=u_init, param_num=k_num
        )

        Xtarget = pd.DataFrame(
            data={
                "x1": x_init[0] + x_init[1] ** k_num[0] + u_init[0],
                "x2": -np.log(x_init[0]) - x_init[1] + u_init[0] * u_init[1],
            },
            index=[0],
        )
        Ytarget = pd.DataFrame(
            data={"y1": k_num[1] * (x_init[0] + x_init[1]) + np.sqrt(x_init[0])},
            index=[0],
        )
        assert_frame_equal(Xval, Xtarget, check_dtype=False)
        assert_frame_equal(Yval, Ytarget, check_dtype=False)
