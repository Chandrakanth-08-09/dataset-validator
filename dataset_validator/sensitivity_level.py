import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

def sensitivity_level(df: pd.DataFrame, attr_json: dict, k_threshold=5, l_threshold=2, t_threshold=0.2):
    """
    Measure dataset sensitivity using k-anonymity, l-diversity, and t-closeness.
    Optimized with Pandas/NumPy.
    """
    quasi_cols = attr_json.get("Quasi Identifiers", [])
    sensitive_cols = attr_json.get("Sensitive Attributes", [])

    if not quasi_cols or not sensitive_cols:
        return {"Error": "Need both quasi-identifiers and sensitive attributes for checks"}
    # 1. k-Anonymity
    group_sizes = df.groupby(quasi_cols).size()
    k_value = group_sizes.min()
    k_pass = k_value >= k_threshold

    # 2. l-Diversity
    l_values = df.groupby(quasi_cols)[sensitive_cols].nunique().min().min()
    l_pass = l_values >= l_threshold

    # 3. t-Closeness
    t_values = []
    for s in sensitive_cols:
        overall_dist = df[s].value_counts(normalize=True)

        # Precompute category probabilities
        overall_dict = overall_dist.to_dict()

        # Compute group distributions in one pass
        group_counts = df.groupby(quasi_cols)[s].value_counts(normalize=True).unstack(fill_value=0)

        # Align with overall distribution categories
        group_counts = group_counts.reindex(columns=overall_dict.keys(), fill_value=0)

        # Convert to NumPy arrays for faster distance calculation
        overall = np.array(list(overall_dict.values()))

        # Compute Wasserstein distances row-wise
        for row in group_counts.values:
            t_values.append(wasserstein_distance(overall, row))

    t_value = max(t_values) if t_values else 0
    t_pass = t_value <= t_threshold

    # Final Sensitivity Decision
    if not k_pass:
        level = "High"
    elif not l_pass or not t_pass:
        level = "Medium"
    else:
        level = "Low"
    return level
