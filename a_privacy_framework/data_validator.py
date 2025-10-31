import os
import json
import hashlib
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from cryptography.fernet import Fernet
from typing import Dict, Any, List, Optional


class DataValidator:
    def __init__(self,
                 k: int = 3,
                 l: int = 2,
                 t: float = 0.2,
                 dp_epsilon: float = 1.0,
                 p_rr: float = 0.1,
                 max_iters: int = 5,
                 swap_fraction: float = 0.1,
                 consent_expiry_days: int = 365,
                 custom_keywords: Optional[Dict[str, List[str]]] = None,
                 custom_weights: Optional[Dict[str, float]] = None,
                 before_path: str = "data/1_raw",
                 after_path: str = "data/2_processed",
                 reports_path: str = "data/3_reports"):

        self.k = k
        self.l = l
        self.t = t
        self.dp_epsilon = dp_epsilon
        self.p_rr = p_rr
        self.max_iters = max_iters
        self.swap_fraction = swap_fraction
        self.consent_expiry_days = consent_expiry_days
        self.before_path = before_path
        self.after_path = after_path
        self.reports_path = reports_path

        # Sensitivity classification defaults
        self.attribute_weights = custom_weights or {
            "direct_identifier": 1.0,
            "quasi_identifier": 0.7,
            "sensitive_attribute": 0.9,
            "other": 0.4
        }

        self.keywords = {
            "direct_identifier": ["name", "email", "phone", "ssn", "address", "contact"],
            "quasi_identifier": ["age", "dob", "zip", "postcode", "ethnicity", "gender"],
            "sensitive_attribute": ["genetic", "lab", "disease"],
            "other": []
        }

        if custom_keywords:
            for cat, kws in custom_keywords.items():
                if cat in self.keywords:
                    self.keywords[cat].extend(kws)
                else:
                    self.keywords[cat] = kws

        self.key_file = os.path.join(self.before_path, "encryption.key")
        self.key = self._load_or_create_key()

    # -------------------- Utilities --------------------
    def _load_dataset(self, path: str) -> pd.DataFrame:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        else:
            return pd.read_parquet(path)

    def _load_or_create_key(self):
        os.makedirs(self.before_path, exist_ok=True)
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_file, "wb") as f:
            f.write(key)
        return key

    def _convert_for_json(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(np.float32(obj))
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return str(obj)

    def _hash_report(self, report):
        report_bytes = json.dumps(report, sort_keys=True, default=self._convert_for_json).encode()
        return hashlib.sha256(report_bytes).hexdigest()

    # -------------------- Sensitivity --------------------
    def _detect_category(self, col_name: str):
        col_lower = col_name.lower()
        for category, kw_list in self.keywords.items():
            if any(kw in col_lower for kw in kw_list):
                return category
        return "other"

    def classify_sensitivity(self, dataset: pd.DataFrame):
        present_attributes = dataset.columns.tolist()
        attribute_categories = {col: self._detect_category(col) for col in present_attributes}
        weights = [self.attribute_weights[attribute_categories[col]] for col in present_attributes]
        sensitivity_score = sum(weights) / len(weights) if weights else 0

        if sensitivity_score >= 0.8:
            classification = "High"
        elif sensitivity_score >= 0.6:
            classification = "Medium"
        else:
            classification = "Low"

        return {
            "metric": "sensitivity-classification",
            "attributes_evaluated": attribute_categories,
            "sensitivity_score": float(np.float32(sensitivity_score)),
            "classification": classification
        }

    # -------------------- Privacy Risk --------------------
    def _detect_columns(self, df: pd.DataFrame):
        direct_identifiers, quasi_identifiers, sensitive_attributes = [], [], []
        n_rows = len(df)
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            dtype = df[col].dtype
            if dtype == object and nunique / n_rows > 0.7:
                direct_identifiers.append(col)
            elif dtype == object or (dtype in [int, float] and nunique / n_rows < 0.1):
                quasi_identifiers.append(col)
            else:
                sensitive_attributes.append(col)
        return direct_identifiers, quasi_identifiers, sensitive_attributes

    def check_k_anonymity(self, df: pd.DataFrame, quasi_identifiers):
        if not quasi_identifiers:
            return {"metric": "k-anonymity", "status": "No quasi-identifiers detected"}
        grouped = df.groupby(quasi_identifiers).size()
        min_group_size = grouped.min() if not grouped.empty else 0
        violating = int((grouped < self.k).sum()) if not grouped.empty else 0
        return {
            "metric": "k-anonymity",
            "k_required": self.k,
            "min_group_size": int(min_group_size),
            "num_violating_groups": violating,
            "is_satisfied": bool(min_group_size >= self.k)
        }

    def check_l_diversity(self, df: pd.DataFrame, quasi_identifiers, sensitive_attributes):
        if not sensitive_attributes or not quasi_identifiers:
            return {"metric": "l-diversity", "status": "No sensitive attributes or QIs detected"}
        min_diversity, violating_counts = {}, {}
        for sens in sensitive_attributes:
            counts = [group[sens].nunique(dropna=True) for _, group in df.groupby(quasi_identifiers)]
            min_diversity[sens] = min(counts) if counts else 0
            violating_counts[sens] = sum(c < self.l for c in counts)
        return {
            "metric": "l-diversity",
            "l_required": self.l,
            "min_diversity": min_diversity,
            "violating_groups_per_sensitive": violating_counts,
            "is_satisfied": all(v >= self.l for v in min_diversity.values())
        }

    def check_t_closeness(self, df: pd.DataFrame, quasi_identifiers, sensitive_attributes):
        if not sensitive_attributes or not quasi_identifiers:
            return {"metric": "t-closeness", "status": "No sensitive attributes or QIs detected"}
        results = {}
        for sens in sensitive_attributes:
            global_dist = df[sens].value_counts(normalize=True)
            max_tv = 0
            for _, group in df.groupby(quasi_identifiers):
                group_dist = group[sens].value_counts(normalize=True)
                all_index = global_dist.index.union(group_dist.index)
                g = global_dist.reindex(all_index, fill_value=0).values
                h = group_dist.reindex(all_index, fill_value=0).values
                tv = 0.5 * np.abs(g - h).sum()
                max_tv = max(max_tv, tv)
            results[sens] = {
                "max_distance": float(np.float32(max_tv)),
                "t_required": self.t,
                "is_satisfied": bool(max_tv <= self.t)
            }
        return {"metric": "t-closeness", "results": results}

    def compute_combined_risk(self, k_report, l_report, t_report):
        k_risk = max(0.0, (self.k - k_report.get("min_group_size", 0)) / self.k)
        l_risks = [max(0.0, (self.l - v) / self.l) for v in l_report.get("min_diversity", {}).values()]
        l_risk = max(l_risks) if l_risks else 1.0
        t_vals = [v["max_distance"] for v in t_report.get("results", {}).values()] if "results" in t_report else [1.0]
        t_risk = max(t_vals)
        combined = 0.4 * k_risk + 0.3 * l_risk + 0.3 * t_risk
        return {
            "k_risk": float(np.float32(k_risk)),
            "l_risk": float(np.float32(l_risk)),
            "t_risk": float(np.float32(t_risk)),
            "combined_score": float(np.float32(combined))
        }

    def assess_privacy(self, df: pd.DataFrame):
        direct_ids, qis, sens = self._detect_columns(df)
        k_report = self.check_k_anonymity(df, qis)
        l_report = self.check_l_diversity(df, qis, sens)
        t_report = self.check_t_closeness(df, qis, sens)
        combined = self.compute_combined_risk(k_report, l_report, t_report)
        return {
            "agent": "Privacy Risk Assessment",
            "direct_identifiers": direct_ids,
            "quasi_identifiers": qis,
            "sensitive_attributes": sens,
            "k_report": k_report,
            "l_report": l_report,
            "t_report": t_report,
            "combined_risk": combined
        }

    # -------------------- Enforcement --------------------
    def enforce_based_on_risk(self, df, privacy_report):
        df_copy = df.copy()
        qis = privacy_report["quasi_identifiers"]
        sens = privacy_report["sensitive_attributes"]
        k_failed = not privacy_report["k_report"]["is_satisfied"]
        l_failed = not privacy_report["l_report"]["is_satisfied"]
        t_failed = not all(v["is_satisfied"] for v in privacy_report["t_report"].get("results", {}).values())

        # k-anonymity → generalize
        if k_failed and qis:
            df_copy = self.generalize_quasi_identifiers(df_copy, qis)

        # l-diversity → perturb sensitive attributes
        if l_failed and sens:
            df_copy = self.perturb_sensitive_attributes(df_copy, sens, noise_fraction=0.05)

        # t-closeness → apply DP to numeric sensitive attributes
        if t_failed and sens:
            df_copy = self.apply_differential_privacy(df_copy, sens, epsilon=self.dp_epsilon)

        # Remove direct identifiers always
        df_copy = self.enforce_direct_identifier_masking(df_copy, privacy_report["direct_identifiers"])
        return df_copy

    def generalize_quasi_identifiers(self, df, qis):
        df_copy = df.copy()
        for col in qis:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = (df_copy[col] // 10) * 10  # simple binning by tens
            else:
                df_copy[col] = df_copy[col].astype(str).str[0]  # first character only
        return df_copy

    def perturb_sensitive_attributes(self, df, sens, noise_fraction=0.05):
        df_copy = df.copy()
        for col in sens:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                noise = df_copy[col] * noise_fraction * np.random.randn(len(df_copy))
                df_copy[col] += noise
            else:
                df_copy[col] = df_copy[col].apply(lambda x: x[::-1] if pd.notnull(x) else x)  # simple perturb
        return df_copy

    def apply_differential_privacy(self, df, sensitive_attributes, epsilon=1.0):
        df_copy = df.copy()
        for col in sensitive_attributes:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                sensitivity = df_copy[col].max() - df_copy[col].min()
                scale = sensitivity / epsilon
                noise = np.random.laplace(0, scale, len(df_copy))
                df_copy[col] += noise
        return df_copy

    # -------------------- Compliance --------------------
    def enforce_direct_identifier_masking(self, df: pd.DataFrame, direct_identifiers: List[str]):
        cleaned = df.copy()
        for col in direct_identifiers:
            if col in cleaned.columns:
                cleaned.drop(columns=[col], inplace=True)
        return cleaned

    def enforce_consent_rules(self, df: pd.DataFrame):
        cleaned = df.copy()
        issues = []
        for col in df.columns:
            if col.lower() == "consent_flag":
                invalid = cleaned[cleaned[col].astype(str).str.lower() != "yes"]
                if not invalid.empty:
                    issues.append(f"{len(invalid)} records without valid consent")
            if col.lower() == "consent_date":
                try:
                    cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                    now = datetime.now()
                    expired = cleaned[(now - cleaned[col]).dt.days > self.consent_expiry_days]
                    if not expired.empty:
                        issues.append(f"{len(expired)} records with expired consent")
                except Exception as e:
                    issues.append(f"Consent date parsing error: {e}")
            if col.lower() in ["right_to_erasure", "delete_flag"]:
                to_delete = cleaned[cleaned[col].astype(str).str.lower() == "yes"]
                if not to_delete.empty:
                    cleaned = cleaned[cleaned[col].astype(str).str.lower() != "yes"]
                    issues.append(f"{len(to_delete)} records deleted due to erasure request")
        return cleaned, issues

    def check_compliance(self, df: pd.DataFrame, direct_identifiers: List[str]):
        df_clean = self.enforce_direct_identifier_masking(df, direct_identifiers)
        df_clean, consent_issues = self.enforce_consent_rules(df_clean)
        status = "Compliant" if not consent_issues else "Non-Compliant"
        report = {
            "agent": "Compliance Agent",
            "compliance_checklist": {
                "Direct_Identifiers": f"Dropped {len(direct_identifiers)} columns",
                "Consent": consent_issues if consent_issues else "All consent rules satisfied",
                "Sectoral": "No sectoral rules applied"
            },
            "final_status": status
        }
        return {"dataset": df_clean, "report": report}

    # -------------------- Repository --------------------
    def store_repository(self, dataset, privacy_report, compliance_report, sensitivity_report):
        validation_report = {
            "privacy": privacy_report,
            "compliance": compliance_report,
            "sensitivity": sensitivity_report,
        }
        validation_report["tamper_proof_hash"] = self._hash_report(validation_report)

        os.makedirs(self.reports_path, exist_ok=True)
        with open(os.path.join(self.reports_path, "validation_report.json"), "w") as f:
            json.dump(validation_report, f, indent=4, default=self._convert_for_json)

        return validation_report

    # -------------------- Unified Validation --------------------
    def validate_dataset(self, dataset_path: str):
        df = self._load_dataset(dataset_path)

        # Sensitivity Classification
        sensitivity_report = self.classify_sensitivity(df)

        # Privacy Risk Assessment before enforcement
        privacy_report_before = self.assess_privacy(df)

        # Apply risk-driven enforcement
        df_enforced = self.enforce_based_on_risk(df, privacy_report_before)

        # Privacy Risk Assessment after enforcement
        privacy_report_after = self.assess_privacy(df_enforced)

        # Compliance Checks on enforced dataset
        compliance_result = self.check_compliance(df_enforced, privacy_report_after["direct_identifiers"])

        # Store repository
        final_report = self.store_repository(
            compliance_result["dataset"],
            privacy_report_after,
            compliance_result["report"],
            sensitivity_report
        )

        return {
            "sensitivity": sensitivity_report,
            "privacy": privacy_report_after,
            "compliance": compliance_result["report"],
            "repository": final_report
        }
