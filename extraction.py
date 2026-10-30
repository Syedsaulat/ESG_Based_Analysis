import os
import re
import json
import time
from copy import deepcopy
from dotenv import load_dotenv
from groq import Groq

# ------------------------------
# Setup
# ------------------------------
load_dotenv()
MODEL_NAME = "deepseek-r1-distill-llama-70b"   # or "deepseek-r1-distill-llama-70b"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ------------------------------
# Schema (shortened example schema, extend as needed)
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
# Utilities
# ------------------------------
def chunk_text(text, max_chars=3000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def deep_merge(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_merge(base[k], v)
        else:
            if v not in (None, "", []):
                base[k] = v
    return base

def count_nulls(data):
    return json.dumps(data).count(": null")

SECTION_SYSTEM = (
    "You are an ESG/BRSR data extractor. "
    "Output ONLY valid JSON following the schema. "
    "All values must be numbers or null. "
    "If unsure, output null."
)

def call_llm(messages, max_tokens=1800):
    while True:
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if "rate limit" in str(e).lower():
                print("⚠️ Rate limit hit, waiting 60s...")
                time.sleep(60)
                continue
            raise e

def extract_section_json(section_name, section_text, sub_schema):
    schema_text = json.dumps(sub_schema, indent=2)
    prompt = f"Extract numeric ESG/BRSR values for {section_name}.\nSchema:\n{schema_text}\n\nText:\n{section_text}"

    resp = call_llm([
        {"role": "system", "content": SECTION_SYSTEM},
        {"role": "user", "content": prompt}
    ])

    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except:
        return deepcopy(sub_schema)

# ------------------------------
# Extraction with 1 re-iteration
# ------------------------------
def extract_brsr(text):
    result = deepcopy(SCHEMA)

    # Pass 1
    for section in ["Environment", "Social", "Governance"]:
        print(f"   🔹 Extracting {section}")
        section_result = deepcopy(SCHEMA[section])
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            print(f"      ➡️ {section} chunk {i+1}/{len(chunks)}")
            try:
                chunk_result = extract_section_json(section, chunk, SCHEMA[section])
                section_result = deep_merge(section_result, chunk_result)
            except Exception as e:
                print(f"⚠️ Error in {section} chunk {i+1}: {e}")
        result[section] = section_result

    # Count nulls after pass 1
    prev_nulls = count_nulls(result)
    print(f"   📊 Pass 1 coverage: {prev_nulls} null fields")

    # Pass 2 (one re-iteration only if nulls exist)
    if prev_nulls > 0:
        print("   🔄 Iteration 1: re-extracting null fields...")
        for section in ["Environment", "Social", "Governance"]:
            section_result = deepcopy(result[section])
            for i, chunk in enumerate(chunk_text(text)):
                chunk_result = extract_section_json(section, chunk, SCHEMA[section])
                section_result = deep_merge(section_result, chunk_result)
            result[section] = section_result

        nulls_after = count_nulls(result)
        print(f"   ✅ Coverage improved: {prev_nulls - nulls_after} fields filled, {nulls_after} still null")

    return result

# ------------------------------
# Main
# ------------------------------
def extract_brsr_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    company = os.path.splitext(os.path.basename(path))[0]
    print(f"\n➡️ Processing {company}")
    data = extract_brsr(text)

    os.makedirs("extracted_test", exist_ok=True)
    out_path = os.path.join("extracted_test", f"{company}_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"   💾 Saved {out_path}")
    return out_path

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    extract_brsr_from_file("cleaned_input_txt/CIPLA_2024-25_ESG.txt")
