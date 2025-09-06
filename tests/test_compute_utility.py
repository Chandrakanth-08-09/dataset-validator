import numpy as np
import pandas as pd
import pytest
from dataset_validator.compute_utility import (   # replace with your filename
    compute_ncp,
    compute_dm,
    compute_query_utility,
    compute_distance_utility,
    compute_suppression_utility,
    compute_composite_utility
)

# ------------------ 1. NCP Tests ------------------
@pytest.mark.unit
def test_ncp_perfect_utility():
    df_orig = pd.DataFrame({"Age": [20, 30, 40, 50]})
    df_deid = df_orig.copy()
    U_ncp = compute_ncp(df_orig, df_deid)
    assert np.isclose(U_ncp, 1.0), f"Expected 1.0, got {U_ncp}"

@pytest.mark.unit
def test_ncp_full_generalization():
    df_orig = pd.DataFrame({"Age": [20, 30, 40, 50]})
    df_deid = pd.DataFrame({"Age": [20, 50, 20, 50]})  # widened to full range
    U_ncp = compute_ncp(df_orig, df_deid)
    assert 0.0 <= U_ncp < 1.0, f"Expected less than 1.0, got {U_ncp}"

# ------------------ 2. DM Tests ------------------

@pytest.mark.unit
def test_dm_perfect_utility():
    """Case: No generalization → categories unchanged → Utility = 1.0"""
    df_orig = pd.DataFrame({"Gender": ["M", "F", "M", "F"]})
    df_deid = df_orig.copy()
    U_dm = compute_dm(df_orig, df_deid)
    assert np.isclose(U_dm, 1.0), f"Expected 1.0, got {U_dm}"

@pytest.mark.unit
def test_dm_worst_case():
    """Case: Full suppression/generalization → all values identical → Utility = 0.0"""
    df_orig = pd.DataFrame({"Gender": ["M", "F", "M", "F"]})
    df_deid = pd.DataFrame({"Gender": ["*"] * 4})
    U_dm = compute_dm(df_orig, df_deid)
    assert np.isclose(U_dm, 0.0), f"Expected 0.0, got {U_dm}"

@pytest.mark.unit
def test_dm_partial_generalization_class_size_change():
    """Case: Some grouping into bigger equivalence classes → Utility strictly between 0 and 1"""
    df_orig = pd.DataFrame({"Gender": ["M", "F", "M", "F"]})
    df_deid = pd.DataFrame({"Gender": ["X", "X", "X", "X"]})  # collapse all into one class
    U_dm = compute_dm(df_orig, df_deid)
    assert 0.0 <= U_dm < 1.0, f"Expected less than 1.0, got {U_dm}"

@pytest.mark.unit
def test_dm_partial_generalization_label_change():
    """Case: Labels changed but class sizes preserved → Utility should drop below 1.0"""
    df_orig = pd.DataFrame({"Gender": ["M", "F", "M", "F"]})
    df_deid = pd.DataFrame({"Gender": ["X", "X", "Y", "Y"]})  # sizes same as orig, but labels different
    U_dm = compute_dm(df_orig, df_deid)
    assert 0.0 <= U_dm < 1.0, f"Expected less than 1.0 due to label change, got {U_dm}"

@pytest.mark.unit
def test_dm_no_categorical_columns():
    """Case: Only numeric attributes → returns 1.0 (no categorical loss)"""
    df_orig = pd.DataFrame({"Age": [20, 30, 40, 50]})
    df_deid = df_orig.copy()
    U_dm = compute_dm(df_orig, df_deid)
    assert np.isclose(U_dm, 1.0), f"Expected 1.0, got {U_dm}"

# ------------------ 3. Query Utility Tests ------------------

@pytest.mark.unit
def test_query_perfect():
    df_orig = pd.DataFrame({"Age": [10, 20, 30]})
    df_deid = df_orig.copy()
    U_query = compute_query_utility(df_orig, df_deid)
    assert np.isclose(U_query, 1.0), f"Expected 1.0, got {U_query}"

@pytest.mark.unit
def test_query_with_error():
    df_orig = pd.DataFrame({"Age": [10, 20, 30]})
    df_deid = pd.DataFrame({"Age": [100, 200, 300]})
    U_query = compute_query_utility(df_orig, df_deid)
    assert 0.0 <= U_query < 1.0, f"Expected between 0 and 1, got {U_query}"

# ------------------ 4. Distance Preservation Tests ------------------

@pytest.mark.unit
def test_distance_perfect():
    df_orig = pd.DataFrame({"Age": [10, 20, 30]})
    df_deid = df_orig.copy()
    U_dist = compute_distance_utility(df_orig, df_deid, sample_size=3)
    assert np.isclose(U_dist, 1.0), f"Expected 1.0, got {U_dist}"

@pytest.mark.unit
def test_distance_randomized():
    np.random.seed(0)
    df_orig = pd.DataFrame({"Age": np.arange(50)})
    df_deid = pd.DataFrame({"Age": np.random.permutation(np.arange(50))})
    U_dist = compute_distance_utility(df_orig, df_deid, sample_size=10)
    assert 0.0 <= U_dist <= 1.0, f"Expected valid range, got {U_dist}"

# ------------------ 5. Suppression Utility Tests ------------------

@pytest.mark.unit
def test_suppression_none():
    df_orig = pd.DataFrame({"Age": [10, 20, 30]})
    df_deid = df_orig.copy()
    U_supp = compute_suppression_utility(df_orig, df_deid)
    assert np.isclose(U_supp, 1.0), f"Expected 1.0, got {U_supp}"

@pytest.mark.unit
def test_suppression_partial():
    df_orig = pd.DataFrame({"Age": [10, 20, 30]})
    df_deid = pd.DataFrame({"Age": [10, 20]})
    U_supp = compute_suppression_utility(df_orig, df_deid)
    assert np.isclose(U_supp, 2/3), f"Expected 0.67, got {U_supp}"

# ------------------ 6. Composite Utility Tests ------------------

@pytest.mark.unit
def test_composite_all_perfect():
    df_orig = pd.DataFrame({
        "Age": [10, 20, 30],
        "Gender": ["M", "F", "M"]
    })
    df_deid = df_orig.copy()
    result = compute_composite_utility(df_orig, df_deid)
    assert np.isclose(result["U_composite"], 1.0), f"Expected 1.0, got {result}"

@pytest.mark.unit
def test_composite_with_generalization():
    df_orig = pd.DataFrame({
        "Age": [10, 20, 30, 40],
        "Gender": ["M", "F", "M", "F"]
    })
    df_deid = pd.DataFrame({
        "Age": [10, 40, 10, 40],     # generalized
        "Gender": ["*", "*", "*", "*"]  # suppressed
    })
    result = compute_composite_utility(df_orig, df_deid)
    assert 0.0 <= result["U_composite"] < 1.0, f"Expected between 0 and 1, got {result}"
