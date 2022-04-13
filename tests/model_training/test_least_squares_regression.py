import os

import casadi as ca
import numpy as np
import pandas as pd
import pytest

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel
from skmid.models import generate_model_parameters

np.random.seed(42)


@pytest.fixture
def load_non_linear_model_data():
    """Generate input signal"""
    CWD = os.getcwd()
    DATA_DIR = "data"
    FILENAME = "non_linear_model.csv"

    data = pd.read_csv(
        filepath_or_buffer=os.path.join(CWD, DATA_DIR, FILENAME), index_col=0
    )

    settings = {
        "N": 10000,  # Number of samples
        "fs": 610.1,  # Sampling frequency [hz]
        "param_truth": [5.625e-6, 2.3e-4, 1, 4.69],
        "param_guess": [5, 2, 1, 5],
        "scale": [1e-6, 1e-4, 1, 1],
        "n_steps_per_sample": 10,
    }

    return (data, settings)


class TestLeastSquaresRegression:
    def test_algorithm(self, load_non_linear_model_data):
        (data, settings) = load_non_linear_model_data
        print(settings)
        data.head()
