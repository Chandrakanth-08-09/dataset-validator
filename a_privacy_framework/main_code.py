# =========================================================
# PRIVACY-PRESERVING DATASET VALIDATION FRAMEWORK
# =========================================================

import os
import pandas as pd
import numpy as np  
import json
from cryptography.fernet import Fernet

# ----------------------------
# 1. GLOBAL CONFIG
# ----------------------------
RAW_DATA_PATH = "data/1_raw/synthetic_healthcare_dataset.csv"
PROCESSED_PATH = "data/2_processed"
ENCRYPTION_KEY_PATH = "fernet.key"

SENSITIVE_KEYWORDS = {
    "direct": ["id", "name", "email", "phone", "address", "contact", "ssn"],
    "quasi": ["age", "zip", "gender", "city", "dob", "location", "region"],
    "sensitive": [
        "disease", "diagnosis", "medical", "health", "salary",
        "income", "blood", "pressure", "glucose", "heart", "result"
    ]
}

# Generate/load encryption key
def load_or_create_key():
    if not os.path.exists(ENCRYPTION_KEY_PATH):
        key = Fernet.generate_key()
        with open(ENCRYPTION_KEY_PATH, "wb") as f:
            f.write(key)
    else:
        with open(ENCRYPTION_KEY_PATH, "rb") as f:
            key = f.read()
    return Fernet(key)

fernet = load_or_create_key()

# ----------------------------
# 2. LOAD DATASET
# ----------------------------
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    print("\n✅ Dataset Loaded Successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3))
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Conceptual encryption
    encrypted = fernet.encrypt(df.to_csv(index=False).encode())
    return df, encrypted

# ----------------------------
# 3. AUTO-INFERENCE
# ----------------------------
def infer_attribute_categories(df):
    direct, quasi, sensitive = set(), set(), set()
    for col in df.columns:
        col_lower = col.lower()
        for key, words in SENSITIVE_KEYWORDS.items():
            if any(word in col_lower for word in words):
                if key == "direct": direct.add(col)
                elif key == "quasi": quasi.add(col)
                elif key == "sensitive": sensitive.add(col)
        if col not in direct and col not in quasi and col not in sensitive:
            dtype = str(df[col].dtype)
            unique_ratio = df[col].nunique() / len(df)
            if "object" in dtype:
                if unique_ratio > 0.7: direct.add(col)
                else: quasi.add(col)
            elif "int" in dtype or "float" in dtype:
                if "score" in col_lower or "level" in col_lower:
                    sensitive.add(col)
                else: quasi.add(col)
            else:
                quasi.add(col)
    return list(direct), list(quasi), list(sensitive)

# ----------------------------
# 4. USER PREFERENCES
# ----------------------------
def collect_user_preferences(df):
    print("\n🔹 USER PREFERENCE LAYER")
    print("Available columns:", list(df.columns))
    direct_attrs = input("Enter Direct Identifiers (comma-separated, press Enter to skip): ").split(",")
    quasi_attrs = input("Enter Quasi Identifiers (comma-separated, press Enter to skip): ").split(",")
    sensitive_attrs = input("Enter Sensitive Attributes (comma-separated, press Enter to skip): ").split(",")

    direct_attrs = [c.strip() for c in direct_attrs if c.strip()]
    quasi_attrs = [c.strip() for c in quasi_attrs if c.strip()]
    sensitive_attrs = [c.strip() for c in sensitive_attrs if c.strip()]

    if not direct_attrs and not quasi_attrs and not sensitive_attrs:
        print("\n⚙️ Running auto-inference...")
        direct_attrs, quasi_attrs, sensitive_attrs = infer_attribute_categories(df)
        print("Direct:", direct_attrs)
        print("Quasi:", quasi_attrs)
        print("Sensitive:", sensitive_attrs)

    # Default weights and privacy settings
    w_direct = float(input("Weight Direct (default 1.0): ") or 1.0)
    w_quasi = float(input("Weight Quasi (default 0.5): ") or 0.5)
    w_sensitive = float(input("Weight Sensitive (default 0.9): ") or 0.9)
    consent = input("Consent to reuse (yes/no, default no): ").strip().lower() or "no"
    alpha = float(input("Privacy tolerance α (0.1–1.0, default 0.5): ") or 0.5)
    ws = float(input("Sensitivity weight scaling ws (0.1–2.0, default 1.0): ") or 1.0)

    user_config = {
        "direct_identifiers": direct_attrs,
        "quasi_identifiers": quasi_attrs,
        "sensitive_attributes": sensitive_attrs,
        "weights": {"direct": w_direct, "quasi": w_quasi, "sensitive": w_sensitive},
        "consent": consent,
        "alpha": alpha,
        "ws": ws
    }
    print("\n✅ User preferences recorded!")
    return user_config

# ----------------------------
# 5. SENSITIVITY SCORE
# ----------------------------
def compute_sensitivity_score(df, user_config):
    direct_cols = user_config["direct_identifiers"]
    quasi_cols = user_config["quasi_identifiers"]
    sensitive_cols = user_config["sensitive_attributes"]
    ws = user_config["ws"]

    total_cols = len(direct_cols) + len(quasi_cols) + len(sensitive_cols)
    if total_cols == 0: return 0.0
    score = 0
    score += len(direct_cols) * user_config["weights"]["direct"] * ws
    score += len(quasi_cols) * user_config["weights"]["quasi"] * ws
    score += len(sensitive_cols) * user_config["weights"]["sensitive"] * ws
    sensitivity_score = score / total_cols
    print(f"\n🔹 Sensitivity Score: {sensitivity_score:.3f}")
    return sensitivity_score

# ----------------------------
# 6. PRIVACY RISK (k,l,t)
# ----------------------------
def compute_privacy_risk_dynamic(df, user_config, k=3, l=2, t=0.2,
                                 alpha_k=0.4, alpha_l=0.3, alpha_t=0.3):
    qis = user_config["quasi_identifiers"]
    sens = user_config["sensitive_attributes"]

    # k-anonymity
    if qis:
        grouped = df.groupby(qis).size()
        min_group = grouped.min() if not grouped.empty else 0
        k_risk = max(0, (k - min_group)/k)
    else: k_risk = 1.0

    # l-diversity
    if sens and qis:
        l_risks = []
        for s in sens:
            counts = []
            for _, group in df.groupby(qis):
                valid_values = group[s][group[s] != "REDACTED"]
                counts.append(valid_values.nunique(dropna=True))
            min_div = min(counts) if counts else 0
            l_risks.append(max(0, (l - min_div)/l))
        l_risk = max(l_risks) if l_risks else 1.0
    else: l_risk = 1.0

    # t-closeness
    if sens and qis:
        t_risks = []
        for s in sens:
            global_dist = df[s].value_counts(normalize=True)
            max_tv = 0
            for _, group in df.groupby(qis):
                group_dist = group[s].value_counts(normalize=True)
                all_idx = global_dist.index.union(group_dist.index)
                g = global_dist.reindex(all_idx, fill_value=0).values
                h = group_dist.reindex(all_idx, fill_value=0).values
                tv = 0.5 * np.abs(g - h).sum()
                max_tv = max(max_tv, tv)
            t_risks.append(max_tv)
        t_risk = max(t_risks)
    else: t_risk = 1.0

    combined_risk = alpha_k*k_risk + alpha_l*l_risk + alpha_t*t_risk

    print("\n🔹 Privacy Risk Metrics (Dynamic Weights)")
    print(f"k-risk: {k_risk:.3f}, l-risk: {l_risk:.3f}, t-risk: {t_risk:.3f}")
    print(f"Combined Privacy Risk Score: {combined_risk:.3f}")

    return {
        "k_risk": k_risk,
        "l_risk": l_risk,
        "t_risk": t_risk,
        "combined_risk": combined_risk,
        "weights": {"alpha_k": alpha_k, "alpha_l": alpha_l, "alpha_t": alpha_t}
    }

# ----------------------------
# 7. ENFORCEMENT LAYER
# ----------------------------
def enforce_privacy_adaptive(df, user_config, alpha=None, max_iter=10):
    """
    Adaptive enforcement to progressively reduce combined privacy risk below threshold.
    Returns:
        df_enforced: enforced dataset
        applied_methods: dict of applied methods per column
        risk_history: list of risk reports per iteration
    """
    df_enforced = df.copy()
    applied_methods = {col: [] for col in df.columns}
    alpha = alpha or user_config.get("alpha", 0.5)
    risk_history = []

    # Track binning for numeric quasi-identifiers
    numeric_bins = {col: 5 for col in user_config["quasi_identifiers"] if pd.api.types.is_numeric_dtype(df[col])}

    for iteration in range(1, max_iter + 1):
        print(f"\n⚡ Iteration {iteration} – Enforcement Step")

        # 1️⃣ Suppress direct identifiers if not already suppressed
        for col in user_config["direct_identifiers"]:
            if col in df_enforced.columns and "Suppressed" not in applied_methods[col]:
                df_enforced[col] = "REDACTED"
                applied_methods[col].append("Suppressed")

        # 2️⃣ Generalize quasi-identifiers progressively
        for col in user_config["quasi_identifiers"]:
            if col in df_enforced.columns:
                try:
                    if pd.api.types.is_numeric_dtype(df_enforced[col]):
                        # Reduce bin count each iteration (min 1)
                        bins = max(1, numeric_bins[col] - iteration)
                        df_enforced[col] = pd.qcut(df_enforced[col], q=bins, duplicates="drop", labels=False)
                        applied_methods[col].append(f"Generalized-{bins}bins")
                    else:
                        # Collapse rare categories into 'OTH'
                        top_categories = df_enforced[col].value_counts().index[:max(1, 5 - iteration)]
                        df_enforced[col] = df_enforced[col].apply(lambda x: x if x in top_categories else 'OTH')
                        applied_methods[col].append("Generalized-OTH")
                except Exception as e:
                    print(f"⚠ Could not generalize '{col}': {e}")

        # 3️⃣ Apply Differential Privacy on numeric sensitive attributes
        for col in user_config["sensitive_attributes"]:
            if col in df_enforced.columns and pd.api.types.is_numeric_dtype(df_enforced[col]):
                sensitivity = df_enforced[col].max() - df_enforced[col].min()
                # Noise grows with iteration
                scale = sensitivity * 0.05 * iteration
                noise = np.random.laplace(loc=0.0, scale=scale, size=len(df_enforced))
                df_enforced[col] = df_enforced[col] + noise
                applied_methods[col].append(f"DP-Laplace-iter{iteration}")

        # 4️⃣ Shuffle quasi-identifiers optionally
        for col in user_config["quasi_identifiers"]:
            if col in df_enforced.columns:
                df_enforced[col] = df_enforced[col].sample(frac=1.0, random_state=iteration).reset_index(drop=True)
                applied_methods[col].append("Shuffled")

        # 5️⃣ Compute current risk
        risk_report = compute_privacy_risk_dynamic(df_enforced, user_config)
        combined_risk = risk_report["combined_risk"]
        risk_history.append(risk_report)

        print(f"➡ Combined Risk after iteration {iteration}: {combined_risk:.3f}")

        # Stop if risk is below threshold
        if combined_risk <= alpha:
            print("✅ Privacy risk below threshold. Stopping enforcement.")
            break

    print("\n✅ Enforcement Layer Completed")
    print(f"Final Combined Risk: {risk_history[-1]['combined_risk']:.3f}")
    return df_enforced, applied_methods, risk_history
# =========================================================
# 6️⃣ Compliance Check
# =========================================================
def check_compliance(user_config, risk_report, alpha_threshold=0.5):
    """
    Verify dataset compliance:
    - User consent
    - Privacy risk below threshold
    """
    consent_ok = user_config.get("consent", "no").lower() == "yes"
    risk_ok = risk_report["combined_risk"] <= alpha_threshold

    print("\n🔹 Compliance Check")
    print(f"User Consent Provided: {'✅' if consent_ok else '❌'}")
    print(f"Privacy Risk ≤ α ({alpha_threshold}): {'✅' if risk_ok else '❌'}")

    compliance_status = consent_ok and risk_ok
    print(f"Overall Compliance Status: {'✅ COMPLIANT' if compliance_status else '❌ NON-COMPLIANT'}")
    return compliance_status


# =========================================================
# 7️⃣ Chunking + Save + Summary
# =========================================================
def save_dataset_chunks(df, save_path, chunk_size=500):
    """
    Split dataset into smaller CSV chunks for storage or processing.
    """
    os.makedirs(save_path, exist_ok=True)
    n_chunks = (len(df) + chunk_size - 1) // chunk_size

    print(f"\n🔹 Saving Dataset in {n_chunks} chunk(s) (chunk_size={chunk_size})")
    for i, start in enumerate(range(0, len(df), chunk_size), 1):
        end = min(start + chunk_size, len(df))
        chunk_file = os.path.join(save_path, f"dataset_chunk_{i}.csv")
        df.iloc[start:end].to_csv(chunk_file, index=False)
        print(f"  ↪ Chunk {i} saved: rows {start}-{end-1} → {chunk_file}")

    print("✅ All chunks saved successfully.")


def summarize_enforcement(df, applied_methods, sensitivity_score, risk_history, user_config):
    """
    Print a concise summary of the enforcement process.
    """
    print("\n--- ENFORCEMENT SUMMARY ---")
    print("Dataset Shape:", df.shape)
    print("User Consent:", user_config.get("consent", "no"))
    print(f"Sensitivity Score: {sensitivity_score:.3f}")
    print(f"Final Combined Privacy Risk: {risk_history[-1]['combined_risk']:.3f}")

    print("\nApplied Methods per Column:")
    for col, methods in applied_methods.items():
        if methods:
            print(f"  {col}: {', '.join(methods)}")

    print("\nIteration-wise Privacy Risk:")
    for i, r in enumerate(risk_history, 1):
        print(f"  Iteration {i}: k={r['k_risk']:.3f}, l={r['l_risk']:.3f}, t={r['t_risk']:.3f}, Combined={r['combined_risk']:.3f}")

# ----------------------------
# 8. MAIN
# ----------------------------
def main():
    # 1️⃣ Load dataset
    df, encrypted = load_dataset(RAW_DATA_PATH)

    # 2️⃣ Collect user preferences
    user_config = collect_user_preferences(df)

    # 3️⃣ Compute initial sensitivity score
    sensitivity_score = compute_sensitivity_score(df, user_config)

    # 4️⃣ Compute initial privacy risk
    initial_risk = compute_privacy_risk_dynamic(df, user_config)

    # 5️⃣ Enforcement Layer – Apply adaptive privacy-preserving methods
    print("\n🔹 Enforcement Layer – Adaptive Privacy Methods")
    df_enforced, applied_methods, risk_history = enforce_privacy_adaptive(
        df, user_config, alpha=user_config.get("alpha", 0.5), max_iter=10
    )

    # 6️⃣ Compliance Check
    compliant = check_compliance(
        user_config, risk_history[-1], alpha_threshold=user_config.get("alpha", 0.5)
    )
    print(f"\n🔹 Compliance Status: {'COMPLIANT ✅' if compliant else 'NON-COMPLIANT ❌'}")

    # 7️⃣ Chunking + Save
    save_dataset_chunks(df_enforced, PROCESSED_PATH, chunk_size=500)
    print(f"🔹 Dataset saved in chunks at: {PROCESSED_PATH}")

    # 8️⃣ Summary
    print("\n--- Summary ---")
    print("Dataset Shape:", df.shape)
    print("User Consent:", user_config["consent"])
    print(f"Sensitivity Score: {sensitivity_score:.3f}")
    print("Final Privacy Report:", json.dumps(risk_history[-1], indent=2))

    print("\n--- Enforced Dataset Preview ---")
    print(df_enforced.head(3))

    # print("\n--- Applied Methods per Column ---")
    # for col, methods in applied_methods.items():
    #     if methods:
    #         print(f"{col}: {', '.join(methods)}")

    print("\n--- Iteration-wise Privacy Risk ---")
    for i, r in enumerate(risk_history, 1):
        print(
            f"Iteration {i}: k-risk={r['k_risk']:.3f}, "
            f"l-risk={r['l_risk']:.3f}, t-risk={r['t_risk']:.3f}, "
            f"Combined Risk={r['combined_risk']:.3f}"
        )

    return df_enforced, applied_methods, risk_history, compliant

if __name__=="__main__":
    main()
