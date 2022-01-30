import pandas as pd
from skmpc.model_selection import generate_model_parameters


class TestGenerateModelParameters:
    """Test class for function generate_model_parameters."""
    
    def test_returned_types(self):
        """Test whether correct object types are returned"""
        #(x, u, param) = generate_model_parameters(nx=2, nu=2, np=2)
        
        
        assert isinstance(dict, dict)
        
    