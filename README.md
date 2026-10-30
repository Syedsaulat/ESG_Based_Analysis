# ESG-Based Analysis

An end-to-end ESG analysis application for extracting, scoring, and visualizing
Environment, Social, and Governance information from BRSR (Business
Responsibility and Sustainability Report) text files.

The project combines deterministic text preprocessing, regex-based metric
discovery, structured LLM extraction, weighted scoring, and a Streamlit
dashboard.

## What the application does

1. Accepts one or more plain-text ESG/BRSR reports.
2. Normalizes Unicode characters and cleans common OCR/PDF extraction errors.
3. Detects report sections, tables, numeric values, and ESG-related metrics.
4. Extracts structured Environment, Social, and Governance data.
5. Validates and post-processes extracted numeric values.
6. Calculates weighted pillar and total ESG scores using the files in `weights/`.
7. Converts the total ESG score into a Sustainalytics-style risk score and
   risk category.
8. Saves JSON results and presents company comparisons in an interactive
   dashboard.

## Main features

- Streamlit dashboard for uploading and processing reports.
- Single-file and parallel batch processing.
- Enhanced Unicode normalization and OCR correction.
- Regex pre-extraction used as hints for structured extraction.
- Groq inference through the configured API key.
- Local GPU inference support and fallback handling when CUDA is available.
- In-memory and disk-based extraction caching.
- Environment, Social, Governance, total-score, and risk visualizations.
- Persistent JSON outputs that can be inspected or consumed by other tools.

## Repository layout

| Path | Description |
| --- | --- |
| `app.py` | Streamlit user interface and dashboard visualizations |
| `pipeline.py` | End-to-end cleaning, extraction, scoring, and risk pipeline |
| `enhanced_preprocessor.py` | Unicode/OCR cleanup, table handling, section detection, and metric extraction |
| `process_txt.py` | Basic text-cleaning helpers |
| `extraction.py` | Groq-based BRSR extraction utilities |
| `hybrid_inference.py` | Groq client with local GPU fallback and validation |
| `local_gpu_inference.py` | Local transformer/GPU inference utilities |
| `compute_score.py` | Standalone weighted-score utility |
| `raw_input_txt/` | Source `.txt` reports |
| `cleaned_input_txt/` | Cleaned reports generated during processing |
| `extracted/` | Structured extraction JSON files |
| `scores/` | Final per-company score JSON files |
| `weights/` | Pillar weights, field weights, and benchmark definitions |
| `cache/` | Cached extraction results; ignored by Git |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.10 or newer is recommended.
- A Groq API key for cloud-assisted extraction.
- CUDA-compatible GPU and compatible PyTorch installation for local GPU
  inference. CPU-only environments may not support all local inference paths.

## Installation

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency list includes Streamlit, pandas, Plotly, python-dotenv, Groq,
PyTorch, Transformers, and Accelerate.

## Configuration

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_api_key_here
```

The extraction module loads this value with `python-dotenv`. Never commit real
API keys or credentials. The repository `.gitignore` excludes `.env`,
`__pycache__/`, compiled Python files, `.vscode/`, and `cache/`.

## Run the dashboard

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually
`http://localhost:8501`.

In the sidebar:

1. Upload one or more `.txt` ESG reports.
2. Click **Run pipeline for uploaded files**, or place files in
   `raw_input_txt/` and click **Run pipeline for all files in `raw_input_txt/`**.
3. Review the generated scores, E/S/G comparison charts, company details, and
   files-on-disk information.

Uploaded files are saved using their original filenames. The filename stem is
used as the company identifier and as the prefix for generated output files.

## Run the pipeline from Python

Process one report:

```powershell
python -c "from pipeline import run_pipeline; run_pipeline('raw_input_txt\\company_report.txt')"
```

Process several reports in parallel:

```powershell
python -c "from pipeline import run_pipeline_batch; print(run_pipeline_batch(['raw_input_txt\\company_a.txt', 'raw_input_txt\\company_b.txt'], max_workers=2))"
```

The module also contains an example entry point:

```powershell
python pipeline.py
```

Update the example input path in `pipeline.py` before using that command for a
different report.

## Processing workflow

The standard `run_pipeline()` flow is:

```text
raw_input_txt/*.txt
        |
        v
cleaned_input_txt/*.txt
        |
        v
extracted/*_metrics.json
        |
        v
scores/*_scores.json
```

Internally, the pipeline:

1. Cleans the source text.
2. Extracts structured BRSR metrics.
3. Loads `weights/pillar_weights.json`,
   `weights/pharma_weights.json`, and `weights/benchmarks.json`.
4. Computes Environment, Social, Governance, and total scores on a 0-100
   scale, where a higher ESG score is better.
5. Derives a risk score and category.
6. Writes the final score JSON to `scores/<company>_scores.json`.

Repeated extraction of identical text can use the disk cache in `cache/`.

## Output format

A score file contains the pillar scores and derived risk information. A
typical output has this shape:

```json
{
  "Environment": 72.4,
  "Social": 68.1,
  "Governance": 75.0,
  "Total": 71.8,
  "Sustainalytics_Risk": 14.1,
  "Risk_Category": "Low"
}
```

Exact values and category thresholds are determined by the scoring logic and
the configured benchmark files. Missing or unrecognized values may reduce the
calculated score, so generated results should be reviewed against the source
report.

## Scoring configuration

The scoring model is data-driven:

- `weights/pillar_weights.json` defines the relative Environment, Social, and
  Governance pillar weights.
- `weights/pharma_weights.json` defines field-level scoring weights used by the
  pipeline.
- `weights/benchmarks.json` defines normalization ranges and subgroup weights.

Changing these files changes future score calculations; previously generated
JSON files are not automatically recalculated.

## Supported input

The dashboard currently accepts `.txt` files. Reports should contain readable
text, ideally produced by a PDF extraction step before upload. Scanned PDFs
should be OCR-processed first. Clear section headings, units, and table labels
improve extraction quality.

## Troubleshooting

### `GROQ_API_KEY` errors

Confirm that `.env` exists in the project root, contains the key without
extra quotes or spaces, and that the virtual environment has
`python-dotenv` and `groq` installed.

### No GPU detected

The dashboard reports CUDA availability in the sidebar. Install a
CUDA-compatible PyTorch build and the appropriate NVIDIA drivers if local GPU
inference is required. Cloud-assisted extraction may still require a valid
Groq configuration.

### No files appear in the dashboard

Confirm that files end in `.txt` and are located in `raw_input_txt/`, or upload
them through the sidebar. The dashboard creates its working directories when
it starts.

### Results look incomplete

Inspect the cleaned report in `cleaned_input_txt/` and the extracted JSON in
`extracted/`. Verify that the source text contains the relevant report sections
and that benchmark ranges in `weights/benchmarks.json` match the intended
metrics.

## Security and data handling

- Do not commit `.env`, API keys, credentials, or confidential reports.
- Review files before adding them to `raw_input_txt/`.
- Generated outputs can contain extracted company information; handle them
  according to the sensitivity of the source reports.
- ESG scores are analytical outputs and should be validated against source
  disclosures before being used for decisions.

## License

No license is currently specified for this repository. Add a `LICENSE` file
before distributing or reusing the project under formal license terms.
