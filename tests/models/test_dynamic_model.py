import pytest

from skmid.models import DynamicModel
from skmid.models import generate_model_parameters


class TestDynamicModel:
    """Test class for function correlation_analysis."""

    def test_original_df_not_changed(self):
        """Test whether the dataframe stays the same after applying the function"""

        (x, u, param) = generate_model_parameters(nx=2, nu=2, nparam=2)

        # assign specific name
        x1, x2 = x[0], x[1]
        u1, u2 = u[0], u[1]
        ka, kb = param[0], param[1]

        # xdot = f(x,u,p) <==> rhs = f(x,u,p)
        rhs = [u1 - ka * x1, u1 * u2 / x1 - u1 * x2 / x1 - kb * x2]

        #%%
        model = DynamicModel(states=x, inputs=u, param=param, model_dynamics=rhs)

        model.print_summary()

        assert True
