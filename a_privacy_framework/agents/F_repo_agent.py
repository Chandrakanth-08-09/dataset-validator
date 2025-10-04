import os
import json
import hashlib
import pandas as pd
from cryptography.fernet import Fernet
from agents.A_base_agent import BaseAgent

class RepositoryAgent(BaseAgent):
    def __init__(self, before_path="data/1_raw", after_path="data/2_processed", reports_path="data/3_reports"):
        """
        Repository Agent for secure dataset storage and tamper-proof validation reports.
        """
        super().__init__("Repository Agent")
        self.before_path = before_path
        self.after_path = after_path
        self.reports_path = reports_path

        # Encryption key for 'before repository' storage
        self.key_file = os.path.join(self.before_path, "encryption.key")
        self.key = self._load_or_create_key()

    def _load_or_create_key(self):
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            os.makedirs(self.before_path, exist_ok=True)
            with open(self.key_file, "wb") as f:
                f.write(key)
            return key

    def _encrypt_dataset(self, dataset: pd.DataFrame, filename: str):
        fernet = Fernet(self.key)
        csv_bytes = dataset.to_csv(index=False).encode()
        encrypted = fernet.encrypt(csv_bytes)
        filepath = os.path.join(self.before_path, filename)
        with open(filepath + ".enc", "wb") as f:
            f.write(encrypted)
        return filepath + ".enc"

    def _save_dataset(self, dataset: pd.DataFrame, filename: str, path: str):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        dataset.to_csv(filepath, index=False)
        return filepath

    def _hash_report(self, report: dict):
        """
        Generate SHA-256 hash for tamper-proofing
        """
        report_bytes = json.dumps(report, sort_keys=True).encode()
        return hashlib.sha256(report_bytes).hexdigest()

    def process(self, dataset: pd.DataFrame, privacy_report: dict, compliance_report: dict, sensitivity_report: dict, original_filename="dataset.csv"):
        """
        Stores datasets securely and generates tamper-proof validation report.
        """
        # 1. Store raw dataset encrypted (Before Repository)
        raw_encrypted_path = self._encrypt_dataset(dataset, original_filename)

        # 2. Store anonymized / ML-ready dataset (After Repository)
        processed_path = self._save_dataset(dataset, original_filename, self.after_path)

        # 3. Generate validation report
        validation_report = {
            "privacy_preserved": all([
                privacy_report.get("results", [])[0].get("is_satisfied", False),
                privacy_report.get("results", [])[1].get("is_satisfied", False),
                all(s["is_satisfied"] for s in privacy_report.get("results", [2]).get("results", {}).values())
            ]),
            "privacy_report": privacy_report,
            "compliance_report": compliance_report.get("report", {}),
            "sensitivity_report": sensitivity_report.get("results", {}),
            "before_repository_path": raw_encrypted_path,
            "after_repository_path": processed_path
        }

        # 4. Tamper-proof hash
        validation_report["tamper_proof_hash"] = self._hash_report(validation_report)

        # Save report
        os.makedirs(self.reports_path, exist_ok=True)
        report_path = os.path.join(self.reports_path, original_filename.replace(".csv", "_validation_report.json"))
        with open(report_path, "w") as f:
            json.dump(validation_report, f, indent=4)

        return {
            "dataset_path": processed_path,
            "validation_report_path": report_path,
            "report_hash": validation_report["tamper_proof_hash"]
        }
