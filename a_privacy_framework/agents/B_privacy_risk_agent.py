import os
import json
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent

class PrivacyRiskAssessmentAgent(BaseAgent):
    def __init__(self, k=3, l=2, t=0.2, weights=None):
        """
        Dataset-agnostic Privacy Risk Assessment Agent.
        Automatically detects quasi-identifiers, sensitive attributes, and direct identifiers.
        
        k: k-anonymity threshold
        l: l-diversity threshold
        t: t-closeness threshold
        weights: dict for combining risk scores {"k": float, "l": float, "t": float}
        """
        super().__init__("Privacy Risk Assessment Agent")
        self.k = k
        self.l = l
        self.t = t
        self.weights = weights if weights else {"k": 0.4, "l": 0.3, "t": 0.3}

    def _load_dataset(self, path: str) -> pd.DataFrame:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        else:
            return pd.read_parquet(path)

    def _detect_columns(self, df: pd.DataFrame):
        """
        Automatically detect:
        - direct_identifiers: mostly unique string columns
        - quasi_identifiers: low-cardinality categorical/string or numeric columns
        - sensitive_attributes: numeric/text columns that are neither QIs nor IDs
        """
        direct_identifiers = []
        quasi_identifiers = []
        sensitive_attributes = []

        n_rows = len(df)
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            dtype = df[col].dtype

            # Direct identifiers: mostly unique string columns
            if dtype == object and nunique / n_rows > 0.7:
                direct_identifiers.append(col)
            # Quasi-identifiers: categorical-ish low-cardinality columns
            elif dtype == object or (dtype in [int, float] and nunique / n_rows < 0.1):
                quasi_identifiers.append(col)
            # Sensitive: everything else numeric or textual
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
            "is_satisfied": min_group_size >= self.k
        }

    def check_l_diversity(self, df: pd.DataFrame, quasi_identifiers, sensitive_attributes):
        if not sensitive_attributes or not quasi_identifiers:
            return {"metric": "l-diversity", "status": "No sensitive attributes or QIs detected"}
        min_diversity = {}
        violating_counts = {}
        for sens in sensitive_attributes:
            counts = []
            for _, group in df.groupby(quasi_identifiers):
                unique_count = group[sens].nunique(dropna=True)
                counts.append(unique_count)
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
                tv = 0.5 * np.abs(g - h).sum()  # total variation distance
                max_tv = max(max_tv, tv)
            results[sens] = {"max_distance": float(max_tv), "t_required": self.t, "is_satisfied": max_tv <= self.t}
        return {"metric": "t-closeness", "results": results}

    def compute_combined_risk_score(self, k_report, l_report, t_report):
        k_risk = max(0.0, (self.k - k_report.get("min_group_size", 0)) / self.k)
        l_risks = [max(0.0, (self.l - v) / self.l) for v in l_report.get("min_diversity", {}).values()] if "min_diversity" in l_report else [1.0]
        l_risk = max(l_risks) if l_risks else 1.0
        t_vals = [v["max_distance"] for v in t_report.get("results", {}).values()] if "results" in t_report else [1.0]
        t_risk = max(t_vals) if t_vals else 1.0
        combined = self.weights["k"] * k_risk + self.weights["l"] * l_risk + self.weights["t"] * t_risk
        return {"k_risk": k_risk, "l_risk": l_risk, "t_risk": t_risk, "combined_score": combined}

    def process(self, artifact):
        dataset_path = artifact.get("dataset_path")
        df = self._load_dataset(dataset_path)

        direct_ids, quasi_identifiers, sensitive_attributes = self._detect_columns(df)

        k_report = self.check_k_anonymity(df, quasi_identifiers)
        l_report = self.check_l_diversity(df, quasi_identifiers, sensitive_attributes)
        t_report = self.check_t_closeness(df, quasi_identifiers, sensitive_attributes)
        combined = self.compute_combined_risk_score(k_report, l_report, t_report)

        report = {
            "agent": self.name,
            "direct_identifiers": direct_ids,
            "quasi_identifiers": quasi_identifiers,
            "sensitive_attributes": sensitive_attributes,
            "k_report": k_report,
            "l_report": l_report,
            "t_report": t_report,
            "combined_risk": combined
        }

        artifact.setdefault("reports", []).append(report)
        reports_dir = artifact.get("reports_dir", "data/3_reports")
        os.makedirs(reports_dir, exist_ok=True)
        out_path = os.path.join(reports_dir, "risk_assessment_report.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=4)
        artifact["last_report_path"] = out_path
        return artifact
