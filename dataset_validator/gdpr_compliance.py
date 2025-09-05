def gdpr_compliance_check(categorized_json):
    """
    Minimal GDPR compliance check based on already categorized columns.
    
    Args:
        categorized_json (dict): {
            "Direct Identifiers": [...],
            "Quasi Identifiers": [...],
            "Sensitive Attributes": [...]
        }
    
    Returns:
        dict: Compliance report
    """

    direct_ids = categorized_json.get("Direct Identifiers", [])

    compliant = len(direct_ids) == 0
    report = {
        "Rule": "GDPR Recital 26 + Article 4(1) - Direct Identifiers",
        "Direct Identifiers Found": direct_ids,
        "Compliant": compliant,
        "Details": (
            "PASS - No direct identifiers present"
            if compliant else f"FAIL - Direct identifiers present: {direct_ids}"
        )
    }
    return report