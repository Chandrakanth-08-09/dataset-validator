# dataset-validator
Privacy-Preserving Dataset Validation Framework


### Development Setup

1. **Clone the repository:**
git clone https://github.com/Chandrakanth-08-09/dataset-validator.git
cd dataset-validator

2. **Install the library in editable mode with dev dependencies:**
pip install -e .[dev]

> This installs the library along with development tools like pytest, black, and flake8.

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

