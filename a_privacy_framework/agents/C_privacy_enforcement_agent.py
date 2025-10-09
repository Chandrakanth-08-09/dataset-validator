# agents/2_privacy_enforcement_agent.py
"""
Privacy Enforcement Agent (Generalized)

Applies deterministic non-perturbative and perturbative transformations to a dataset
flagged as risky by Layer 1. Iteratively recomputes privacy metrics using the
PrivacyRiskAssessmentAgent until thresholds are met or max iterations reached.

Design decisions (generic):
- Non-perturbative:
    - Generalize numeric columns by binning
    - Truncate long string/text columns
    - Suppress highly identifying columns (user-defined or auto-detected)
    - Data swapping among quasi-identifiers to reduce linkability
- Perturbative:
    - Laplace noise on numeric sensitive columns
    - Randomized response on categorical sensitive columns
"""

import os
import json
import copy
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from A_base_agent import BaseAgent
from B_privacy_risk_agent import PrivacyRiskAssessmentAgent

class PrivacyEnforcementAgent(BaseAgent):
    def __init__(self,
                 quasi_identifiers: Optional[List[str]] = None,
                 sensitive_attributes: Optional[List[str]] = None,
                 identifiers_to_suppress: Optional[List[str]] = None,
                 k: int = 5,
                 l: int = 3,
                 t: float = 0.2,
                 dp_epsilon: float = 1.0,
                 p_rr: float = 0.1,
                 max_iters: int = 5,
                 swap_fraction: float = 0.1):
        
        super().__init__("Privacy Enforcement Agent")
        self.quasi_identifiers = quasi_identifiers or []
        self.sensitive_attributes = sensitive_attributes or []
        self.identifiers_to_suppress = identifiers_to_suppress or []
        self.k = k
        self.l = l
        self.t = t
        self.dp_epsilon = float(dp_epsilon)
        self.p_rr = float(p_rr)
        self.max_iters = int(max_iters)
        self.swap_fraction = float(swap_fraction)

        # Risk agent for iterative checking
        self.risk_agent = PrivacyRiskAssessmentAgent(
            quasi_identifiers=self.quasi_identifiers,
            sensitive_attributes=self.sensitive_attributes,
            k=self.k, l=self.l, t=self.t
        )

    # ---------- Helpers ----------
    def _load_dataset(self, path: str) -> pd.DataFrame:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path, dtype=object)
        else:
            return pd.read_parquet(path)

    def _write_processed(self, df: pd.DataFrame, artifact: Dict[str, Any]) -> str:
        processed_dir = artifact.get("processed_dir", os.path.join("data", "2_processed"))
        os.makedirs(processed_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(artifact["dataset_path"]))[0]
        out_path = os.path.join(processed_dir, f"{base_name}_sanitized.csv")
        df.to_csv(out_path, index=False)
        return out_path

    def _save_report(self, report: Dict[str, Any], artifact: Dict[str, Any], name="enforcement_report.json") -> str:
        reports_dir = artifact.get("reports_dir", os.path.join("data", "3_reports"))
        os.makedirs(reports_dir, exist_ok=True)
        out_path = os.path.join(reports_dir, name)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=4)
        return out_path

    # ---------- Non-perturbative transformations ----------
    def generalize_numeric_bins(self, df: pd.DataFrame, n_bins: int = 5):
        df = df.copy()
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        for col in numeric_cols:
            try:
                df[col] = pd.qcut(df[col].astype(float), q=n_bins, duplicates='drop').astype(str)
            except Exception:
                continue
        return df

    def truncate_strings(self, df: pd.DataFrame, max_len: int = 8):
        df = df.copy()
        str_cols = [c for c in df.columns if df[c].dtype == object]
        for col in str_cols:
            df[col] = df[col].astype(str).str[:max_len]
        return df

    def suppress_identifiers(self, df: pd.DataFrame, identifiers: List[str]):
        df = df.copy()
        dropped = []
        for col in identifiers:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                dropped.append(col)
        return df, dropped

    def data_swap(self, df: pd.DataFrame, cols_to_swap: List[str], fraction: float = 0.1, seed: int = 42):
        df = df.copy()
        if not cols_to_swap:
            return df
        n = len(df)
        swap_n = max(1, int(n * fraction))
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        swap_idx = rng.choice(idx, size=swap_n, replace=False)
        for col in cols_to_swap:
            if col not in df.columns:
                continue
            values = df.loc[swap_idx, col].values
            rng.shuffle(values)
            df.loc[swap_idx, col] = values
        return df

    # ---------- Perturbative transformations ----------
    def apply_laplace_noise(self, df: pd.DataFrame, numeric_cols: List[str], epsilon: float):
        df = df.copy()
        for col in numeric_cols:
            if col not in df.columns:
                continue
            try:
                series = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                continue
            if series.isna().all():
                continue
            col_min, col_max = float(np.nanmin(series)), float(np.nanmax(series))
            sensitivity = max(1e-6, col_max - col_min)
            scale = sensitivity / max(1e-12, epsilon)
            noise = np.random.laplace(loc=0.0, scale=scale, size=len(series))
            df[col] = series.fillna(col_min) + noise
        return df

    def apply_randomized_response(self, df: pd.DataFrame, categorical_cols: List[str], p: float):
        df = df.copy()
        for col in categorical_cols:
            if col not in df.columns:
                continue
            vals = df[col].astype(str).fillna("NA").values
            unique_vals = list(pd.Series(vals).unique())
            if len(unique_vals) == 0:
                continue
            flips = np.random.rand(len(vals)) < p
            replacements = np.random.choice(unique_vals, size=len(vals))
            vals[flips] = replacements[flips]
            df[col] = vals
        return df

    # ---------- Main process ----------
    def process(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        if "dataset_path" not in artifact:
            raise ValueError("artifact must contain dataset_path")

        dataset_path = artifact["dataset_path"]
        df_orig = self._load_dataset(dataset_path)
        df = df_orig.copy()

        report = {
            "agent": self.name,
            "actions": [],
            "iterations": [],
            "final_processed_path": None,
            "suppressed_columns": [],
            "status": "not_completed"
        }

        # Detect numeric and categorical columns dynamically
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        # Enforcement loop
        for it in range(1, self.max_iters + 1):
            iteration_record = {"iteration": it, "applied": []}

            # Non-perturbative
            df = self.generalize_numeric_bins(df)
            iteration_record["applied"].append("generalized_numeric_bins")

            df = self.truncate_strings(df)
            iteration_record["applied"].append("truncated_string_columns")

            if self.identifiers_to_suppress:
                df, dropped = self.suppress_identifiers(df, self.identifiers_to_suppress)
                if dropped:
                    iteration_record["applied"].append(f"suppressed_columns:{dropped}")
                    report["suppressed_columns"].extend(dropped)

            df = self.data_swap(df, cols_to_swap=self.quasi_identifiers, fraction=self.swap_fraction, seed=it*42)
            iteration_record["applied"].append(f"data_swap_on_qis_fraction_{self.swap_fraction}")

            # Perturbative
            numeric_sensitive = [c for c in numeric_cols if c in self.sensitive_attributes]
            if numeric_sensitive:
                df = self.apply_laplace_noise(df, numeric_sensitive, epsilon=self.dp_epsilon)
                iteration_record["applied"].append(f"laplace_noise_on:{numeric_sensitive}_eps_{self.dp_epsilon}")

            categorical_sensitive = [c for c in categorical_cols if c in self.sensitive_attributes]
            if categorical_sensitive:
                df = self.apply_randomized_response(df, categorical_sensitive, p=self.p_rr)
                iteration_record["applied"].append(f"randomized_response_on:{categorical_sensitive}_p_{self.p_rr}")

            processed_path = self._write_processed(df, artifact)
            iteration_record["processed_snapshot"] = processed_path

            # Recompute privacy risk
            tmp_artifact = {"dataset_path": processed_path,
                            "reports_dir": artifact.get("reports_dir", os.path.join("data", "3_reports"))}
            risk_result_artifact = self.risk_agent.process(tmp_artifact)
            try:
                with open(risk_result_artifact.get("last_report_path", ""), "r") as f:
                    risk_report = json.load(f)
            except Exception:
                risk_report = None

            iteration_record["risk_report_snapshot"] = risk_report
            report["iterations"].append(iteration_record)

            # Check thresholds
            k_ok = risk_report.get("k_report", {}).get("is_satisfied", False) if risk_report else False
            l_ok = risk_report.get("l_report", {}).get("is_satisfied", False) if risk_report else False
            t_ok = True
            tr_results = risk_report.get("t_report", {}).get("results", {}) if risk_report else {}
            for sens, val in tr_results.items():
                if not val.get("is_satisfied", False):
                    t_ok = False
                    break
            if k_ok and l_ok and t_ok:
                final_path = processed_path
                report["final_processed_path"] = final_path
                report["status"] = "succeeded"
                artifact["processed_dataset_path"] = final_path
                artifact["enforcement_report_path"] = self._save_report(report, artifact)
                return artifact

        # Max iterations reached
        final_path = self._write_processed(df, artifact)
        report["final_processed_path"] = final_path
        report["status"] = "max_iters_reached_not_satisfied"
        artifact["processed_dataset_path"] = final_path
        artifact["enforcement_report_path"] = self._save_report(report, artifact)
        return artifact
