import pytest
import os
from dataset_validator.classify_attributes import classify_attributes

@pytest.mark.integration
def test_classify_attributes():
    """
    This test makes a REAL call to Gemini API.
    Requires GEMINI_API_KEY to be set in environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set, skipping real API test")

    columns = ["Name", "Age", "Email", "Salary"]

    result = classify_attributes(columns)

    # Basic checks (don’t assert exact categories, Gemini may vary)
    assert isinstance(result, dict)
    assert "Direct Identifiers" in result
    assert "Quasi Identifiers" in result
    assert "Sensitive Attributes" in result

    print("\nGemini classification result:", result)