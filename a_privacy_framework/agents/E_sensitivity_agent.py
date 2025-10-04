import pandas as pd
from agents.base_agent import BaseAgent

class SensitivityClassificationAgent(BaseAgent):
    def __init__(self, custom_keywords=None, custom_weights=None):
        """
        Dynamically assigns sensitivity scores to dataset attributes.
        
        Parameters:
        - custom_keywords: dict mapping category -> list of keywords to detect columns
        - custom_weights: dict mapping category -> weight for sensitivity calculation
        """
        super().__init__("Sensitivity Classification Agent")
        
        # Default weights
        self.attribute_weights = custom_weights or {
            "direct_identifier": 1.0,
            "quasi_identifier": 0.7,
            "sensitive_attribute": 0.9,
            "other": 0.4
        }

        # Default keywords
        self.keywords = {
            "direct_identifier": ["name", "email", "phone", "ssn", "address", "contact"],
            "quasi_identifier": ["age", "dob", "zip", "postcode", "ethnicity", "gender"],
            "sensitive_attribute": [],  # optional, user can provide domain-specific sensitive columns
            "other": []
        }

        # Merge custom keywords if provided
        if custom_keywords:
            for cat, kws in custom_keywords.items():
                if cat in self.keywords:
                    self.keywords[cat].extend(kws)
                else:
                    self.keywords[cat] = kws

    def _detect_category(self, col_name: str):
        """Detect category of a column based on keywords."""
        col_lower = col_name.lower()
        for category, kw_list in self.keywords.items():
            if any(kw in col_lower for kw in kw_list):
                return category
        return "other"  # fallback for unknown columns

    def classify_dataset(self, dataset: pd.DataFrame):
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
            "sensitivity_score": round(sensitivity_score, 3),
            "classification": classification
        }

    def process(self, dataset: pd.DataFrame):
        return {
            "agent": self.name,
            "results": self.classify_dataset(dataset)
        }
