import os
import re
import json
import time
import hashlib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import threading

# Import enhanced preprocessor
try:
    from enhanced_preprocessor import (
        preprocess_for_extraction,
        enhanced_clean_brsr,
        extract_metrics_regex,
        detect_sections,
    )
    ENHANCED_PREPROCESSING = True
    print("✅ Enhanced preprocessing enabled")
except ImportError:
    ENHANCED_PREPROCESSING = False
    print("⚠️ Enhanced preprocessor not available, using basic cleaning")

# ---------- CACHING LAYER ----------
_extraction_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_text_hash(text: str) -> str:
    """Generate hash for text content to use as cache key."""
    return hashlib.md5(text.encode()).hexdigest()[:16]

def get_cached_result(cache_key: str) -> Optional[Dict]:
    """Retrieve cached extraction result."""
    # Check memory cache first
    with _cache_lock:
        if cache_key in _extraction_cache:
            return _extraction_cache[cache_key]
    
    # Check disk cache
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                result = json.load(f)
            with _cache_lock:
                _extraction_cache[cache_key] = result
            return result
        except:
            pass
    return None

def set_cached_result(cache_key: str, result: Dict):
    """Store extraction result in cache."""
    with _cache_lock:
        _extraction_cache[cache_key] = result
    
    # Also persist to disk
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(result, f)
    except:
        pass

# Import LOCAL GPU inference (no cloud API)
try:
    from local_gpu_inference import (
        LocalGPUClient,
        validate_extracted_data,
        check_data_consistency,
        CUDA_AVAILABLE,
        get_gpu_info,
    )
    LOCAL_GPU_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Local GPU inference not available: {e}")
    LOCAL_GPU_AVAILABLE = False
    CUDA_AVAILABLE = False

# ------------------------------
# Setup - LOCAL GPU ONLY (no Groq)
# ------------------------------
if LOCAL_GPU_AVAILABLE and CUDA_AVAILABLE:
    client = LocalGPUClient()
    gpu_info = get_gpu_info()
    print(f"✅ Local GPU inference enabled: {gpu_info.get('gpu_name', 'Unknown GPU')}")
    print(f"   VRAM: {gpu_info.get('gpu_memory_total_gb', '?')} GB")
else:
    client = None
    print("❌ No GPU available - cannot run extraction")
    print("   Please ensure CUDA is installed and a GPU is available")

# ------------------------------
# Global token counter
# ------------------------------
token_counter = {"prompt": 0, "completion": 0, "total": 0}

# ------------------------------
# Schema (73 fields total)
# ------------------------------
SCHEMA = {
    "Environment": {
        "scope1_tco2e": None, "scope2_tco2e": None, "scope3_tco2e": None,
        "ghg_total_tco2e": None, "ghg_intensity_tco2e_per_revenue": None,
        "energy_consumption_total_mwh": None, "energy_from_renewables_mwh": None,
        "renewable_energy_percent": None, "water_withdrawal_kl": None,
        "water_consumption_kl": None, "water_discharge_kl": None,
        "water_recycled_percent": None, "water_intensity_kl_per_unit": None,
        "carbon_emissions": {"baseline_year": None,"target_year": None,
                             "target_reduction_percent": None,"achieved_reduction_percent": None},
        "water_consumption_reduction": {"baseline_year": None,"target_year": None,
                                        "target_reduction_percent": None,"achieved_reduction_percent": None},
        "waste_hazardous_generated_tonnes": None,"waste_hazardous_recycled_percent": None,
        "waste_hazardous_disposed_percent": None,"waste_non_hazardous_generated_tonnes": None,
        "waste_non_hazardous_recycled_percent": None,"waste_co_processed_percent": None,
        "nox_tonnes": None,"sox_tonnes": None,"pm_tonnes": None,
        "effluent_treated_percent": None,"sites_in_protected_areas_count": None,
    },
    "Social": {
        "total_employees": None,"permanent_employees": None,"contract_workers": None,
        "male_percent": None,"female_percent": None,"women_in_board_percent": None,
        "women_in_senior_mgmt_percent": None,"pwd_employees_percent": None,"new_hires_count": None,
        "new_hires_female_percent": None,"attrition_rate_percent": None,"attrition_rate_female_percent": None,
        "training_hours_total": None,"training_hours_per_employee": None,"training_coverage_percent": None,
        "fatalities_employees_count": None,"fatalities_contractors_count": None,
        "ltifr_employees": None,"trir_employees": None,"lost_days_count": None,
        "min_wage_coverage_percent": None,"avg_salary_ratio_male_to_female": None,
        "parental_leave_return_rate_percent": None,"csr_spend_million_inr": None,
        "csr_beneficiaries_count": None,"community_investment_percent_of_profit": None,
    },
    "Governance": {
        "board_size_count": None,"independent_directors_count": None,"independent_directors_percent": None,
        "women_directors_count": None,"women_directors_percent": None,"esg_board_committee_exists": None,
        "audit_committee_exists": None,"risk_committee_exists": None,"board_meetings_count": None,
        "whistleblower_complaints_filed": None,"whistleblower_complaints_pending": None,
        "whistleblower_complaints_resolved": None,"anti_corruption_training_coverage_percent": None,
        "data_privacy_breaches_count": None,"regulatory_penalties_count": None,
        "regulatory_penalties_amount_in_inr": None,"political_contributions_in_inr": None,
        "standards_and_certifications": [],
    },
}

# ------------------------------
# Utility: Load JSON
# ------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------
# Step 1: Process raw TXT file (ENHANCED)
# ------------------------------
def process_text(raw_path, cleaned_folder="cleaned_input_txt"):
    """Clean raw text with enhanced preprocessing."""
    os.makedirs(cleaned_folder, exist_ok=True)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Use enhanced preprocessing if available
    if ENHANCED_PREPROCESSING:
        result = preprocess_for_extraction(raw_text)
        text = result["cleaned_text"]
        pre_extracted = result.get("pre_extracted", {})
        stats = result.get("stats", {})
        print(f"   📊 Pre-extracted {len(pre_extracted)} metrics via regex patterns")
        
        # Save pre-extracted data for merging later
        company = os.path.splitext(os.path.basename(raw_path))[0]
        preextract_path = os.path.join(cleaned_folder, f"{company}_preextract.json")
        with open(preextract_path, "w", encoding="utf-8") as f:
            json.dump({"pre_extracted": pre_extracted, "stats": stats}, f, indent=2)
    else:
        text = re.sub(r"\s+", " ", raw_text).strip()

    company = os.path.splitext(os.path.basename(raw_path))[0]
    cleaned_path = os.path.join(cleaned_folder, f"{company}.txt")

    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"   ✨ Cleaned text saved at {cleaned_path}")
    return cleaned_path

# ------------------------------
# Step 2: Extract BRSR metrics (ENHANCED)
# ------------------------------
def extract_brsr_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    company = os.path.splitext(os.path.basename(path))[0]
    print(f"\n➡️ Processing {company}")
    
    # Try to load pre-extracted data
    pre_extracted = {}
    preextract_path = path.replace(".txt", "_preextract.json")
    if os.path.exists(preextract_path):
        try:
            with open(preextract_path, "r") as f:
                pre_data = json.load(f)
                pre_extracted = pre_data.get("pre_extracted", {})
                print(f"   📋 Loaded {len(pre_extracted)} pre-extracted metrics")
        except:
            pass
    
    data = extract_brsr(text, pre_extracted=pre_extracted)
    
    # Merge pre-extracted values (regex is reliable for exact patterns)
    if pre_extracted:
        for key, value in pre_extracted.items():
            # Find which pillar this key belongs to
            for pillar in ["Environment", "Social", "Governance"]:
                if key in SCHEMA.get(pillar, {}):
                    if data[pillar].get(key) is None:
                        data[pillar][key] = value

    os.makedirs("extracted", exist_ok=True)
    out_path = os.path.join("extracted", f"{company}_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"   💾 Saved {out_path}")
    return out_path

# ------------------------------
# Text Cleaning helper (not used directly in run_pipeline, but kept)
# ------------------------------
def clean_brsr_text(raw_text: str) -> str:
    return re.sub(r"\s+", " ", raw_text).strip()

# ------------------------------
# Utilities
# ------------------------------
def chunk_text(text, max_chars=4500):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def deep_merge(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_merge(base[k], v)
        else:
            if v not in (None, "", []):
                base[k] = v
    return base

def postprocess_numbers(data):
    """Enhanced number postprocessing with better recognition."""
    if isinstance(data, dict):
        return {k: postprocess_numbers(v) for k, v in data.items()}
    if isinstance(data, list):
        return [postprocess_numbers(x) for x in data]
    if isinstance(data, str):
        v = data.strip().lower()
        
        # Boolean conversions
        if v in {"yes", "true", "y", "available", "exists"}: 
            return 1
        if v in {"no", "false", "n", "not available", "na", "n/a", "nil"}: 
            return 0
        
        # Clean up the string
        v = v.replace(",", "")  # Remove thousands separator
        v = v.replace("%", "")  # Remove percentage
        v = v.replace(" ", "")  # Remove spaces
        v = re.sub(r'[^\d.\-]', '', v)  # Keep only digits, decimal, minus
        
        # Handle Indian number formats
        lakh_match = re.search(r'(\d+\.?\d*)\s*lakh', data.lower())
        crore_match = re.search(r'(\d+\.?\d*)\s*crore', data.lower())
        million_match = re.search(r'(\d+\.?\d*)\s*million', data.lower())
        
        if crore_match:
            return float(crore_match.group(1)) * 10_000_000
        if lakh_match:
            return float(lakh_match.group(1)) * 100_000
        if million_match:
            return float(million_match.group(1)) * 1_000_000
        
        # Standard number parsing
        try:
            num = float(v)
            return int(num) if num.is_integer() else num
        except:
            return data
    return data

NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

# Max realistic values for ESG metrics (to cap LLM hallucinations)
MAX_VALUES = {
    "_tco2e": 50_000_000,  # 50M tonnes CO2 max
    "_mwh": 100_000_000,  # 100M MWh max
    "_kl": 1_000_000_000,  # 1B kL water max
    "_percent": 100,  # Max 100%
    "_count": 1_000_000,  # 1M count max
    "_inr": 100_000_000_000,  # 100B INR max
    "_tonnes": 10_000_000,  # 10M tonnes max
}

def cap_value(key: str, value: float) -> float:
    """Cap values to realistic maximums to prevent LLM hallucinations."""
    if value is None:
        return None
    for suffix, max_val in MAX_VALUES.items():
        if suffix in key.lower():
            if abs(value) > max_val:
                return None  # Return null for unrealistic values
    return value

def sanitize_json(data, schema, parent_key=""):
    """Strict enforcement: only numbers/null according to schema with value capping."""
    if isinstance(data, dict) and isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            if k in data:
                cleaned[k] = sanitize_json(data[k], v, k)
            else:
                cleaned[k] = None
        return cleaned
    if isinstance(data, list):
        return [sanitize_json(x, schema[0] if schema else None, parent_key) for x in data]
    if data is None: 
        return None
    if isinstance(data, (int, float)):
        return cap_value(parent_key, data)
    if isinstance(data, str):
        val = data.strip().replace(",", "").replace("%", "")
        if NUMERIC_RE.match(val):
            num = float(val) if "." in val else int(val)
            return cap_value(parent_key, num)
    return None

# ------------------------------
# Extraction (LLM) - LOCAL GPU ONLY with IMPROVED PROMPTS
# ------------------------------
SECTION_SYSTEM = (
    "Extract ESG metrics as JSON. Output ONLY {\"field\": value} format. "
    "Use numbers without units. Use null if not found. No explanations."
)

# Field descriptions to help the small model
FIELD_HINTS = {
    "Environment": {
        "scope1_tco2e": "Direct GHG emissions (Scope 1) in tonnes CO2",
        "scope2_tco2e": "Indirect GHG emissions from electricity (Scope 2)",
        "scope3_tco2e": "Other indirect emissions (Scope 3)",
        "energy_consumption_total_mwh": "Total energy consumed in MWh or GJ",
        "renewable_energy_percent": "Percentage of renewable energy",
        "water_withdrawal_kl": "Total water withdrawn in kilolitres",
        "water_recycled_percent": "Percentage of water recycled",
        "waste_hazardous_generated_tonnes": "Hazardous waste generated in tonnes",
    },
    "Social": {
        "total_employees": "Total number of employees",
        "permanent_employees": "Number of permanent employees",
        "women_in_board_percent": "Percentage of women on board",
        "attrition_rate_percent": "Employee turnover/attrition rate",
        "training_hours_per_employee": "Average training hours per employee",
        "ltifr_employees": "Lost Time Injury Frequency Rate",
        "fatalities_employees_count": "Number of employee fatalities",
        "csr_spend_million_inr": "CSR spending in million INR",
    },
    "Governance": {
        "board_size_count": "Total number of board members",
        "independent_directors_percent": "Percentage of independent directors",
        "women_directors_percent": "Percentage of women directors",
        "whistleblower_complaints_filed": "Number of whistleblower complaints",
        "regulatory_penalties_count": "Number of regulatory penalties",
        "board_meetings_count": "Number of board meetings held",
    },
}

def call_llm(messages, max_tokens=400):
    """Call local GPU LLM for inference (optimized for speed)."""
    global LOCAL_GPU_AVAILABLE
    
    if not LOCAL_GPU_AVAILABLE or client is None:
        raise RuntimeError("Local GPU not available for inference")
    
    # Use local GPU client with reduced tokens for speed
    result = client.call(messages, max_tokens=max_tokens, temperature=0.05)
    
    # Create a mock response object for compatibility
    class MockResponse:
        def __init__(self, content, usage):
            self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})()})]
            if usage:
                self.usage = type('obj', (object,), {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                })()
            else:
                self.usage = type('obj', (object,), {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                })()
    
    return MockResponse(result["content"], result.get("usage"))

def extract_section_json(section_name, section_text, sub_schema, pre_extracted=None):
    global token_counter
    
    # Get field hints for better prompts
    hints = FIELD_HINTS.get(section_name, {})
    
    # Build focused schema with hints
    schema_keys = list(sub_schema.keys())[:12]  # Limit to key fields
    schema_with_hints = {}
    for k in schema_keys:
        if k in hints:
            schema_with_hints[k] = f"<{hints[k]}>"
        else:
            schema_with_hints[k] = "<number or null>"
    
    schema_text = json.dumps(schema_with_hints, indent=1)
    
    # If we have pre-extracted values, include them as examples
    pre_hint = ""
    if pre_extracted:
        relevant_pre = {k: v for k, v in pre_extracted.items() if k in sub_schema}
        if relevant_pre:
            pre_hint = f"\nAlready found: {json.dumps(relevant_pre)}\nVerify or add missing values.\n"

    prompt = (
        f"Extract {section_name} data from text below.\n"
        f"Output JSON: {{{', '.join([f'\"{k}\": number' for k in schema_keys[:8]])}, ...}}\n"
        f"{pre_hint}"
        f"Text: {section_text[:2000]}\n"
        f"JSON:"
    )

    resp = call_llm([
        {"role": "system", "content": SECTION_SYSTEM},
        {"role": "user", "content": prompt}
    ])

    if hasattr(resp, "usage"):
        token_counter["prompt"] += resp.usage.prompt_tokens
        token_counter["completion"] += resp.usage.completion_tokens
        token_counter["total"] += resp.usage.total_tokens

    raw = resp.choices[0].message.content
    
    # Handle empty response
    if not raw or not raw.strip():
        return deepcopy(sub_schema)
    
    try:
        # Try multiple JSON extraction patterns
        parsed = None
        
        # Pattern 1: Find JSON object with nested content
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except:
                pass
        
        # Pattern 2: Try full response as JSON
        if parsed is None:
            try:
                parsed = json.loads(raw.strip())
            except:
                pass
        
        # Pattern 3: Extract key-value pairs manually
        if parsed is None:
            parsed = {}
            for key in sub_schema.keys():
                # Look for "key": number pattern
                match = re.search(rf'"{key}"\s*:\s*([\d.]+|null)', raw, re.IGNORECASE)
                if match:
                    val = match.group(1)
                    parsed[key] = None if val == 'null' else float(val)
        
        if parsed:
            parsed = sanitize_json(parsed, sub_schema)
            return parsed
        return deepcopy(sub_schema)
        
    except Exception as e:
        print(f"   ⚠️ JSON parse error in {section_name}: {e}")
        return deepcopy(sub_schema)

def extract_brsr(text, use_cache: bool = True, pre_extracted: Dict = None):
    global token_counter
    token_counter = {"prompt": 0, "completion": 0, "total": 0}
    result = deepcopy(SCHEMA)
    
    # Check cache first
    if use_cache:
        cache_key = get_text_hash(text)
        cached = get_cached_result(cache_key)
        if cached:
            print("   ⚡ Using cached extraction result!")
            return cached

    # Larger chunks (8KB) = fewer LLM calls = faster overall
    chunks = chunk_text(text, max_chars=8000)
    num_chunks = len(chunks)
    
    # Limit to 6 most relevant chunks per section to reduce processing time
    if num_chunks > 6:
        # Keep first 3 and last 3 chunks (intro + main content + summary)
        chunks = chunks[:3] + chunks[-3:]
        num_chunks = len(chunks)
        print(f"   📉 Using {num_chunks} key chunks (reduced from {len(chunk_text(text, max_chars=8000))} for speed)")
    
    # Load pre-extracted data if not provided
    if pre_extracted is None:
        pre_extracted = {}
        if ENHANCED_PREPROCESSING:
            pre_extracted = extract_metrics_regex(text)
    
    def process_chunk(args):
        """Process a single chunk for a section"""
        section, chunk_idx, chunk, schema = args
        try:
            return section, chunk_idx, extract_section_json(section, chunk, schema, pre_extracted)
        except Exception as e:
            print(f"⚠️ Error in {section} chunk {chunk_idx}: {e}")
            return section, chunk_idx, deepcopy(schema)
    
    def process_section_sequential(section, text_chunks):
        """Process chunks sequentially (safer for local GPU memory)"""
        print(f"   🔹 Extracting {section} ({len(text_chunks)} chunks)")
        section_result = deepcopy(SCHEMA[section])
        
        for i, chunk in enumerate(text_chunks):
            try:
                chunk_result = extract_section_json(section, chunk, SCHEMA[section], pre_extracted)
                section_result = deep_merge(section_result, chunk_result)
            except Exception as e:
                print(f"⚠️ Error in {section} chunk {i}: {e}")
        
        return section, section_result

    # Process sections sequentially (more stable for local GPU)
    print(f"   ⚡ Processing E/S/G sections ({num_chunks} chunks total)...")
    print(f"   🖥️ Using local GPU with 4-bit quantization")
    if pre_extracted:
        print(f"   📊 Using {len(pre_extracted)} pre-extracted values as hints")
    
    start_time = time.time()
    
    for section in ["Environment", "Social", "Governance"]:
        section, section_result = process_section_sequential(section, chunks)
        result[section] = section_result
        print(f"   ✅ {section} complete")
    
    elapsed = time.time() - start_time
    print(f"   ⏱️ Extraction completed in {elapsed:.1f}s")

    result = postprocess_numbers(result)

    # ---------- Validation & Error Checking ----------
    if LOCAL_GPU_AVAILABLE:
        print("   🔍 Validating extracted data...")
        
        # Validate against schema
        validation = validate_extracted_data(result, SCHEMA)
        
        if validation["warnings"]:
            print(f"   ⚠️ {len(validation['warnings'])} warnings:")
            for w in validation["warnings"][:5]:  # Show first 5
                print(f"      - {w}")
        
        # Check logical consistency
        consistency_issues = check_data_consistency(result)
        if consistency_issues:
            print(f"   ⚠️ {len(consistency_issues)} consistency issues:")
            for issue in consistency_issues:
                print(f"      - {issue}")
        
        if validation["suspicious_values"]:
            print(f"   🔎 {len(validation['suspicious_values'])} suspicious values found")
        
        # Print GPU stats
        if hasattr(client, 'get_stats'):
            stats = client.get_stats()
            print(f"   📊 Local GPU inference: {stats['calls']} calls, ~{stats['tokens_generated']} tokens generated")

    s = json.dumps(result)
    nulls = s.count(": null")
    total = s.count(":")
    print(f"    Coverage: {total-nulls}/{total} fields filled ({100*(total-nulls)/total:.1f}%)")
    print(
        f"    Tokens used — "
        f"Prompt: {token_counter['prompt']}, "
        f"Completion: {token_counter['completion']}, "
        f"Total: {token_counter['total']}"
    )
    
    # Cache the result
    if use_cache:
        set_cached_result(cache_key, result)
        print("   💾 Result cached for future runs")
    
    return result


# ------------------------------
# BATCH PROCESSING - Process multiple files in parallel
# ------------------------------
def run_pipeline_batch(raw_txt_paths: List[str], max_workers: int = 2):
    """
    Process multiple ESG reports in parallel.
    Use max_workers=2 to balance speed vs API rate limits.
    """
    print(f"\n🚀 Batch processing {len(raw_txt_paths)} files with {max_workers} workers...")
    start_time = time.time()
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(run_pipeline, path): path for path in raw_txt_paths}
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            company = os.path.splitext(os.path.basename(path))[0]
            try:
                future.result()
                results[company] = "Success"
                print(f"   ✅ {company} completed")
            except Exception as e:
                results[company] = f"Error: {e}"
                print(f"   ❌ {company} failed: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Batch completed in {elapsed:.1f}s ({elapsed/len(raw_txt_paths):.1f}s per file avg)")
    return results

# ------------------------------
# Step 3: Weighted Scoring
# ------------------------------
def compute_weighted_score(company_data, pillar_weights, field_weights, benchmarks):
    """
    Compute weighted ESG score with subgroup normalization.
    """
    results = {}
    total_score = 0.0

    for pillar, p_weight in pillar_weights["Pharma"].items():
        pillar_score = 0.0
        pillar_bench = benchmarks.get(pillar, {})

        for subgroup, subgroup_data in pillar_bench.items():
            subgroup_weight = subgroup_data.get("weight", 0)
            subgroup_score = 0.0

            for field, limits in subgroup_data["fields"].items():
                value = company_data.get(pillar, {}).get(field, None)
                norm_value = 0.0

                if value is not None:
                    if isinstance(value, (int, float)):
                        num = value
                    elif isinstance(value, str):
                        try:
                            num = float(value.replace(",", "").replace("%", ""))
                        except:
                            num = None
                    else:
                        num = None

                    if num is not None:
                        vmin, vmax = limits["min"], limits["max"]
                        if vmax > vmin:
                            norm_value = (num - vmin) / (vmax - vmin)
                            norm_value = max(0, min(1, norm_value))

                subgroup_score += norm_value

            if subgroup_data["fields"]:
                subgroup_score /= len(subgroup_data["fields"])

            pillar_score += subgroup_score * subgroup_weight

        results[pillar] = round(pillar_score * p_weight * 100, 2)
        total_score += results[pillar]

    results["Total"] = round(total_score, 2)
    return results

# ------------------------------
# NEW: Convert ESG score to Sustainalytics-style risk
# ------------------------------
def convert_to_sustainalytics_risk(total_esg_score: float) -> float:
    """
    Our ESG Total is 0–100, higher = better.
    Sustainalytics risk is 0–50, higher = worse.

    So:
      1) reverse: 100 - total
      2) scale to 0–50
    """
    if total_esg_score is None:
        return None
    reversed_score = 100.0 - float(total_esg_score)
    sustain_risk = (reversed_score / 100.0) * 50.0
    return round(sustain_risk, 2)

def classify_risk_category(risk_score: float) -> str:
    """
    0–10 Negligible
    10–20 Low
    20–30 Medium
    30–40 High
    40+  Severe
    """
    if risk_score is None:
        return "Unknown"
    if risk_score < 10:
        return "Negligible"
    elif risk_score < 20:
        return "Low"
    elif risk_score < 30:
        return "Medium"
    elif risk_score < 40:
        return "High"
    else:
        return "Severe"

# ------------------------------
# Pipeline Driver
# ------------------------------
def run_pipeline(raw_txt_path):
    # Step 1: Clean text
    cleaned_path = process_text(raw_txt_path)

    # Step 2: Extract metrics JSON
    metrics_path = extract_brsr_from_file(cleaned_path)

    # Step 3: Load extracted data + weights
    data = load_json(metrics_path)
    weights = load_json("weights/pillar_weights.json")
    field_weights = load_json("weights/pharma_weights.json")
    benchmarks = load_json("weights/benchmarks.json")

    # Step 4: Compute ESG score (0–100, higher = better)
    scores = compute_weighted_score(data, weights, field_weights, benchmarks)

    # Step 5: Derive Sustainalytics-style risk score + category
    total_esg = scores.get("Total")
    risk_score = convert_to_sustainalytics_risk(total_esg)
    scores["Sustainalytics_Risk"] = risk_score
    scores["Risk_Category"] = classify_risk_category(risk_score)

    # Save scores
    company = os.path.splitext(os.path.basename(raw_txt_path))[0]
    os.makedirs("scores", exist_ok=True)
    out_path = os.path.join("scores", f"{company}_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"   ✅ Final scores saved at {out_path}")

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    run_pipeline("raw_input_txt/CIPLA_2024-25_ESG.txt")
