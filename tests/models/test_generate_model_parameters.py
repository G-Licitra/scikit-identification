import casadi as ca
import pandas as pd
import pytest

from skmid.models import generate_model_parameters


class TestGenerateModelParameters:
    """Test class for function generate_model_parameters."""

    def test_function_output(self):
        """Test output size for different scenarios"""

        # Case when x,u, and param are required
        (x, u, param) = generate_model_parameters(nstate=5, ninput=3, nparam=2)
        assert x.size() == (5, 1) and u.size() == (3, 1) and param.size() == (2, 1)

        # Case when x,u are only required
        (x, u, param) = generate_model_parameters(nstate=4, ninput=1, nparam=0)
        assert x.size() == (4, 1) and u.size() == (1, 1) and param == None

        (x, u, param) = generate_model_parameters(nstate=4, ninput=1)
        assert x.size() == (4, 1) and u.size() == (1, 1) and param == None

        # case when only x is specified
        (x, u, param) = generate_model_parameters(nstate=6, ninput=0)
        assert x.size() == (6, 1) and u == None and param == None

        (x, u, param) = generate_model_parameters(nstate=6)
        assert x.size() == (6, 1) and u == None and param == None

    def test_input_not_integer(self):
        """Test robustness against input which are not integer"""
        with pytest.raises(NotImplementedError):
            (x, u, param) = generate_model_parameters(nstate=5, ninput="st", nparam=0.5)

    def test_zero_state_scenario(self):
        """Test output when x is not defined or zero"""
        with pytest.raises(ValueError):
            (x, u, param) = generate_model_parameters(nstate=0, ninput=3, nparam=2)

        with pytest.raises(TypeError):
            (x, u, param) = generate_model_parameters(ninput=3)

    def test_negative_dimention_scenario(self):
        """Test when input is set negative"""
        with pytest.raises(RuntimeError):
            (x, u, param) = generate_model_parameters(nstate=5, ninput=-3)
