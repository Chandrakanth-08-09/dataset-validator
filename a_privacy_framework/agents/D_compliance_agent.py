import pandas as pd
from datetime import datetime
from A_base_agent import BaseAgent
from typing import List, Dict, Any, Optional

class ComplianceAgent(BaseAgent):
    def __init__(self, consent_expiry_days: int = 365, sensitive_columns: Optional[List[str]] = None):
        super().__init__("Compliance Agent")
        self.consent_expiry_days = consent_expiry_days
        self.sensitive_columns = sensitive_columns or []

    def enforce_direct_identifier_masking(self, df: pd.DataFrame, direct_identifiers: List[str]):
        cleaned = df.copy()
        dropped = []
        for col in direct_identifiers:
            if col in cleaned.columns:
                cleaned.drop(columns=[col], inplace=True)
                dropped.append(col)
        return cleaned, dropped

    def enforce_consent_rules(self, df: pd.DataFrame):
        cleaned = df.copy()
        issues = []

        # Check consent_flag-like columns
        for col in df.columns:
            if col.lower() == "consent_flag":
                invalid = cleaned[cleaned[col].astype(str).str.lower() != "yes"]
                if not invalid.empty:
                    issues.append(f"{len(invalid)} records without valid consent")

        # Check consent_date-like columns
        for col in df.columns:
            if col.lower() == "consent_date":
                try:
                    cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                    now = datetime.now()
                    expired = cleaned[(now - cleaned[col]).dt.days > self.consent_expiry_days]
                    if not expired.empty:
                        issues.append(f"{len(expired)} records with expired consent")
                except Exception as e:
                    issues.append(f"Consent date parsing error: {e}")

        # Generic right-to-erasure-like columns
        for col in df.columns:
            if col.lower() in ["right_to_erasure", "delete_flag"]:
                to_delete = cleaned[cleaned[col].astype(str).str.lower() == "yes"]
                if not to_delete.empty:
                    cleaned = cleaned[cleaned[col].astype(str).str.lower() != "yes"]
                    issues.append(f"{len(to_delete)} records deleted due to erasure request")

        return cleaned, issues

    def process(self, df: pd.DataFrame, direct_identifiers: Optional[List[str]] = None):
        """
        Main compliance pipeline:
        1. Drop/mask direct identifiers
        2. Apply generic consent and erasure rules
        3. Produce compliance report
        """
        report: Dict[str, Any] = {
            "agent": self.name,
            "compliance_checklist": {},
            "final_status": "Compliant"
        }

        # 1. Direct identifier masking
        direct_identifiers = direct_identifiers or self.sensitive_columns
        df_cleaned, dropped = self.enforce_direct_identifier_masking(df, direct_identifiers)
        report["compliance_checklist"]["Direct_Identifiers"] = f"Dropped columns: {dropped}" if dropped else "No columns dropped"

        # 2. Consent rules
        df_cleaned, consent_issues = self.enforce_consent_rules(df_cleaned)
        if consent_issues:
            report["compliance_checklist"]["Consent"] = consent_issues
            report["final_status"] = "Non-Compliant"
        else:
            report["compliance_checklist"]["Consent"] = "All consent rules satisfied"

        # 3. Sectoral rules placeholder
        report["compliance_checklist"]["Sectoral"] = "No sectoral rules applied"

        return {"dataset": df_cleaned, "report": report}
