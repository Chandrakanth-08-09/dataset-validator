# tests/test_sensitivity_checker.py
import pytest
import pandas as pd
from dataset_validator.sensitivity_level import sensitivity_level

@pytest.mark.unit
def test_sensitivity_level_low():
    # Quasi-identifiers and sensitive attributes
    data = {
        "Age": [25, 25, 30, 30],
        "Gender": ["M", "M", "F", "F"],
        "Salary": [50000, 50000, 70000, 70000]
    }
    df = pd.DataFrame(data)
    attr_json = {
        "Quasi Identifiers": ["Age", "Gender"],
        "Sensitive Attributes": ["Salary"]
    }

    result = sensitivity_level(df, attr_json, k_threshold=2, l_threshold=1, t_threshold=0.5)
    assert result == "Low"  # Everything passes

@pytest.mark.unit
def test_sensitivity_level_medium():
    # l-diversity fails (all Salary same in group)
    data = {
        "Age": [25, 25, 30, 30],
        "Gender": ["M", "M", "F", "F"],
        "Salary": [50000, 50000, 50000, 50000]  # No diversity
    }
    df = pd.DataFrame(data)
    attr_json = {
        "Quasi Identifiers": ["Age", "Gender"],
        "Sensitive Attributes": ["Salary"]
    }

    result = sensitivity_level(df, attr_json, k_threshold=2, l_threshold=2, t_threshold=0.5)
    assert result == "Medium"

@pytest.mark.unit
def test_sensitivity_level_high():
    # k-anonymity fails (group size < k_threshold)
    data = {
        "Age": [25, 30],
        "Gender": ["M", "F"],
        "Salary": [50000, 70000]
    }
    df = pd.DataFrame(data)
    attr_json = {
        "Quasi Identifiers": ["Age", "Gender"],
        "Sensitive Attributes": ["Salary"]
    }

    result = sensitivity_level(df, attr_json, k_threshold=2, l_threshold=1, t_threshold=0.5)
    assert result == "High"