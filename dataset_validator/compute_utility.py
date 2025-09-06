import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ------------------ 1. Numeric Information Loss (NCP) ------------------
def compute_ncp(original: pd.DataFrame, deid: pd.DataFrame):
    """
    Computes Normalized Certainty Penalty (NCP) for numeric attributes.
    Measures loss of precision due to interval generalization.
    """
    ncp_total = 0
    num_count = 0
    
    for col in original.select_dtypes(include=[np.number]).columns:
        if col not in deid.columns:
            continue
        range_orig = original[col].max() - original[col].min()
        range_deid = deid[col].max() - deid[col].min()
        if range_orig == 0: 
            continue
        unique_orig = original[col].nunique()
        unique_deid = deid[col].nunique()

        if unique_orig == unique_deid and original[col].equals(deid[col]):
            # identical column
            continue
        loss = (unique_orig - unique_deid) / unique_orig    
        ncp_total += loss
        num_count += 1

    if num_count == 0:
        return 1.0  # no numeric attributes, treat as max utility
    ncp_dataset = ncp_total / num_count
    return max(0.0, 1 - ncp_dataset)  # higher = better utility


# ------------------ 2. Categorical Information Loss (DM) ------------------
def compute_dm(original: pd.DataFrame, deid: pd.DataFrame):
    """
    Computes Discernibility Metric (DM) for categorical attributes.
    Improved version:
    - Uses equivalence class sizes AND category fidelity.
    - Ensures that if labels change (generalization), utility decreases.
    """
    cat_cols = original.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) == 0:
        return 1.0

    grouped_orig = original.groupby(list(cat_cols)).size()
    grouped_deid = deid.groupby(list(cat_cols)).size()

    DM_orig = np.sum(grouped_orig.values ** 2)
    DM_deid = np.sum(grouped_deid.values ** 2)

    n = len(original)
    DM_max = n ** 2

    # Base DM utility (class sizes)
    if DM_max == DM_orig:
        base_utility = 1.0
    else:
        base_utility = 1 - (DM_deid - DM_orig) / (DM_max - DM_orig)

    # Category fidelity penalty (checks if values changed)
    fidelity = 0
    for col in cat_cols:
        overlap = len(set(original[col].unique()) & set(deid[col].unique()))
        fidelity += overlap / max(1, len(set(original[col].unique())))
    fidelity /= len(cat_cols)

    # Final utility: combine both
    U_dm = base_utility * fidelity
    return max(0.0, min(1.0, U_dm))

# ------------------ 3. Query Utility ------------------
def compute_query_utility(original: pd.DataFrame, deid: pd.DataFrame, queries=None):
    """
    Computes query accuracy utility (U_query) between original and de-identified datasets.
    Default queries: mean and count for each column.
    """
    if queries is None:
        queries = []
        for col in original.select_dtypes(include=[np.number]).columns:
            queries.append(("mean", col))
            queries.append(("count", col))
        for col in original.select_dtypes(exclude=[np.number]).columns:
            queries.append(("count", col))

    errors = []
    for qtype, col in queries:
        if col not in original.columns or col not in deid.columns:
            continue
        if qtype == "mean":
            val_orig = original[col].mean()
            val_deid = deid[col].mean()
        elif qtype == "count":
            val_orig = original[col].count()
            val_deid = deid[col].count()
        else:
            continue
        err = abs(val_orig - val_deid) / max(1e-9, abs(val_orig))
        errors.append(err)

    query_error = np.mean(errors) if errors else 1.0
    return max(0.0, 1 - query_error)


# ------------------ 4. Distance Preservation ------------------
def compute_distance_utility(original: pd.DataFrame, deid: pd.DataFrame, sample_size=500):
    """
    Computes distance preservation utility using Spearman correlation.
    Ensures that record-to-record relationships are preserved.
    """
    sample_size = min(sample_size, len(original))
    idx = np.random.choice(len(original), size=sample_size, replace=False)

    orig_sample = original.iloc[idx].select_dtypes(include=[np.number]).fillna(0)
    deid_sample = deid.iloc[idx].select_dtypes(include=[np.number]).fillna(0)

    if orig_sample.shape[1] == 0 or deid_sample.shape[1] == 0:
        return 1.0  # no numeric data to compare

    dist_orig = np.linalg.norm(orig_sample.values[:, None] - orig_sample.values[None, :], axis=2).flatten()
    dist_deid = np.linalg.norm(deid_sample.values[:, None] - deid_sample.values[None, :], axis=2).flatten()

    rho, _ = spearmanr(dist_orig, dist_deid)
    return (rho + 1) / 2 if not np.isnan(rho) else 0.0


# ------------------ 5. Suppression Utility ------------------
def compute_suppression_utility(original: pd.DataFrame, deid: pd.DataFrame):
    """
    Computes suppression utility: proportion of records retained.
    """
    return len(deid) / max(1, len(original))


# ------------------ 6. Composite Utility ------------------
def compute_composite_utility(original: pd.DataFrame, deid: pd.DataFrame,
                              weights={"query":0.3, "ncp":0.25, "dm":0.25, "dist":0.1, "supp":0.1}):
    """
    Composite utility function combining:
    - Query accuracy (U_query)
    - Numeric info loss (U_ncp)
    - Categorical info loss (U_dm)
    - Distance preservation (U_dist)
    - Suppression impact (U_supp)
    """
    U_query = compute_query_utility(original, deid)
    U_ncp = compute_ncp(original, deid)
    U_dm = compute_dm(original, deid)
    U_dist = compute_distance_utility(original, deid)
    U_supp = compute_suppression_utility(original, deid)

    total_w = sum(weights.values())
    w = {k: v/total_w for k,v in weights.items()}

    U_composite = (w["query"]*U_query +
                   w["ncp"]*U_ncp +
                   w["dm"]*U_dm +
                   w["dist"]*U_dist +
                   w["supp"]*U_supp)

    return {
        "U_query": U_query,
        "U_ncp": U_ncp,
        "U_dm": U_dm,
        "U_dist": U_dist,
        "U_supp": U_supp,
        "U_composite": U_composite
    }
