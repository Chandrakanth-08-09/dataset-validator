# dataset-validator
Privacy-Preserving Dataset Validation Framework


### Development Setup

1. **Clone the repository:**
git clone https://github.com/Chandrakanth-08-09/dataset-validator.git
cd dataset-validator

2. **Install the library in editable mode with dev dependencies:**
pip install -e .[dev]

> This installs the library along with development tools like pytest, black, and flake8 (specified in pyproject.toml).

3. **Run tests to verify setup:**
pytest -m 


This library uses the **Google Gemini API** in one of its modules.  
To use it, you must obtain your own API key and configure it securely.

### Step 1. Get a Gemini API Key

Follow the instructions on Google AI Studio to create your Gemini API key.

### Step 2. Set the Key as an Environment Variable

You must set the key in your environment so the library can read it.

#### On Linux / macOS (bash/zsh)
export GEMINI_API_KEY="your_real_api_key"

#### On Windows (PowerShell)
$env:GEMINI_API_KEY="your_real_api_key"

### On Windows (Windows CMD)
set GEMINI_API_KEY=your_api_key_here

# The code part
### Base agent 
**Purpose**
A lightweight parent class that standardizes how agents behave in the framework.
All other agents inherit from BaseAgent to ensure a consistent interface (process() method) and metadata logging.

**Key Features:**

Stores agent name and metadata.

Provides base structure for initialization and logging.

Enforces that all child agents implement a process() method.

### 1. Privacy Risk Assessment agent 
**Purpose**
This agent evaluates privacy risks of a dataset before it is published or shared.
It checks whether the dataset satisfies k-anonymity, ℓ-diversity, and t-closeness thresholds.

**Techniques Used:**

k-anonymity: Ensures each quasi-identifier group appears at least k times.

ℓ-diversity: Ensures each sensitive attribute in a group has at least ℓ distinct values.

t-closeness: Ensures the distribution of sensitive attributes in each group is close to the overall distribution.

**Process:**

Load dataset from data/1_raw/.

Define quasi-identifiers and sensitive attributes.

Compute metrics (k, ℓ, t) on dataset.

Generate a structured JSON report with results.

### 2. Privacy Enforcement Agent

**Purpose**:  
This agent applies privacy-preserving transformations when the dataset fails the risk assessment in Layer 1.  
It enforces **k-anonymity**, **ℓ-diversity**, and **t-closeness** thresholds through both **non-perturbative** and **perturbative** techniques.  

**Techniques used**:
- *Non-perturbative*:
  - **Generalization**: DOB → Year, ZIP → first 3 digits.
  - **Suppression**: Drops highly identifying columns flagged as identifiers.
  - **Data Swapping**: Randomly shuffles values across records for quasi-identifiers.
- *Perturbative*:
  - **Differential Privacy (Laplace Noise)**: Adds calibrated noise to numeric sensitive attributes.
  - **Randomized Response**: Flips categorical values with probability `p`.

**Process**:
1. Load dataset flagged as risky.  
2. Apply transformations iteratively.  
3. After each round, recompute k, ℓ, t using the **PrivacyRiskAssessmentAgent**.  
4. Stop once thresholds are satisfied (or after `max_iters`).  
5. Save the sanitized dataset under `data/2_processed/` and an enforcement report under `data/3_reports/`.

### 3. Regulatory Compliance Agent

**Objective**:  
Ensure the dataset is legally compliant with privacy regulations such as HIPAA, GDPR, and sector-specific standards. This step validates that privacy-preserved data (from Layer 2) can be legally stored, processed, or shared.

**Key Features:**

HIPAA Safe Harbor (US Healthcare):

Removes 18 types of direct identifiers (e.g., Name, Phone, SSN, Email, Address).

Generalizes Age > 89 into a single “90+” category.

GDPR (European Union):

Validates that Consent_Flag = Yes for identifiable records.

Ensures Consent_Date is still valid (not expired).

Enforces Right to Erasure → deletes records flagged for removal.

Final Status: "Compliant" or "Non-Compliant".



