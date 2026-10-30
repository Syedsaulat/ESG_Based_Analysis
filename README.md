# ESG-Based Analysis

An ESG analysis dashboard for extracting, scoring, and visualizing Environment,
Social, and Governance metrics from BRSR (Business Responsibility and
Sustainability Report) text files.

## Features

- Upload one or more ESG report `.txt` files through a Streamlit dashboard.
- Clean and normalize OCR/PDF-extracted text.
- Extract ESG metrics and relevant report sections.
- Generate ESG scores and risk-oriented summaries.
- Use Groq-powered inference with local GPU inference support for fallback
  handling.
- Cache extraction results to reduce repeated processing.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard |
| `pipeline.py` | End-to-end processing pipeline |
| `enhanced_preprocessor.py` | Text normalization, OCR cleanup, and metric extraction |
| `extraction.py` | ESG report extraction helpers |
| `compute_score.py` | Score calculation utilities |
| `hybrid_inference.py` | Groq/local inference integration |
| `local_gpu_inference.py` | Local model inference support |
| `raw_input_txt/` | Uploaded source reports |
| `cleaned_input_txt/` | Cleaned report text |
| `extracted/` | Extracted metrics and content |
| `scores/` | Generated ESG score files |
| `weights/` | Model or scoring weights |

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_api_key_here
```

Keep `.env` private. It is excluded from Git by `.gitignore`.

## Run the dashboard

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit, upload `.txt` ESG reports, and run
the analysis.

## Input and output

The dashboard accepts plain-text ESG/BRSR reports. Processing results are
written to the project folders listed above. Generated cache files are stored
in `cache/` and are excluded from version control.

## Notes

- The project is designed for text extracted from ESG/BRSR documents.
- GPU-related packages are included to support local inference where available.
- Never commit API keys, credentials, or other sensitive report data.
