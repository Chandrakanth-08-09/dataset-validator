# agents/base_agent.py

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def process(self, dataset):
        """
        Process an artifact and return updated artifact (may add reports, paths).
        artifact is a dictionary with keys like:
            - dataset_path: "C:\Users\abina\Documents\Documents\sem7\project_Work\dataset-validator\a_privacy_framework\data\1_raw\synthetic_healthcare_dataset.csv"
            - metadata: dict
            - reports: list
        """
        raise NotImplementedError("Each agent must implement the process() method.")
