import json
import google.generativeai as genai
import os 

# Configure Gemini API key from environment
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def classify_attributes(columns):
    """
    Classify dataset columns into Direct Identifiers, Quasi Identifiers,
    Sensitive Attributes, and Safe Attributes using Gemini API.
    
    Args:
        columns (list[str]): List of column names
    
    Returns:
        dict: JSON with classified columns
    """

    MODEL = "gemini-1.5-flash" 
    prompt = f"""
    Classify the following dataset columns into one of these categories:
    1. Direct identifiers (Direct identifiers are pieces of information that can be used to directly identify an individual like name, email, id or phone number, etc.).
    2. Quasi-identifiers (Quasi-identifiers are pieces of information that are not of themselves unique identifiers, but are sufficiently well correlated with an entity that they can be combined with other quasi-identifiers to create a unique identifier like age, gender, or location, etc.).
    3. Sensitive Attributes (Sensitive Attributes refer to data that must be protected like medical history, financial information, or personal information).

    Columns: {columns}

    Return the result in strict JSON format with keys:
    Direct Identifiers, Quasi Identifiers, Sensitive Attributes.
    """

    # Call Gemini
    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)

    # Parse Gemini response safely into dict
    try:
        result = json.loads(response.text)
    except Exception:
        # If model response isn't strict JSON, clean up
        cleaned = response.text.strip("`\n ")  
        if cleaned.startswith("json"):
            cleaned = cleaned.replace("json", "", 1).strip()
        result = json.loads(cleaned)
    
    return result