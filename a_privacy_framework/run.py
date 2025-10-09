from data_validator import DataValidator
import json

validator = DataValidator()
results = validator.validate_dataset("data/1_raw/synthetic_healthcare_dataset.csv")

print(json.dumps(results, indent=4))
