# tests/test_gdpr_compliance.py
import pytest
from dataset_validator.gdpr_compliance import gdpr_compliance_check

@pytest.mark.unit
def test_gdpr_compliant_case():
    # No direct identifiers → should pass
    categorized_json = {
        "Direct Identifiers": [],
        "Quasi Identifiers": ["Age", "Gender"],
        "Sensitive Attributes": ["Salary"]
    }

    report = gdpr_compliance_check(categorized_json)

    assert isinstance(report, dict)
    assert report["Compliant"] is True
    assert report["Direct Identifiers Found"] == []
    assert "PASS" in report["Details"]

@pytest.mark.unit
def test_gdpr_non_compliant_case():
    # Direct identifiers present → should fail
    categorized_json = {
        "Direct Identifiers": ["Name", "Email"],
        "Quasi Identifiers": ["Age"],
        "Sensitive Attributes": ["Salary"]
    }

    report = gdpr_compliance_check(categorized_json)

    assert isinstance(report, dict)
    assert report["Compliant"] is False
    assert report["Direct Identifiers Found"] == ["Name", "Email"]
    assert "FAIL" in report["Details"]
