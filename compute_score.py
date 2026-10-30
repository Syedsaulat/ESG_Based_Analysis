import os
import json

# ------------------------------
# Utility
# ------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------
# Compute Weighted Score
# ------------------------------
def compute_weighted_score(company_data, pillar_weights, benchmarks):
    """
    Compute weighted ESG score with subgroup normalization.
    - pillar_weights: e.g. {"Pharma": {"Environment":0.4, "Social":0.3, "Governance":0.3}}
    - benchmarks: nested dict with pillar -> subgroup -> {weight, fields}
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
                    # ensure numeric
                    try:
                        num = float(str(value).replace(",", "").replace("%", ""))
                    except:
                        num = None

                    # normalize with min/max
                    if num is not None:
                        vmin, vmax = limits["min"], limits["max"]
                        if vmax > vmin:
                            norm_value = (num - vmin) / (vmax - vmin)
                            norm_value = max(0, min(1, norm_value))  # clamp 0–1

                subgroup_score += norm_value

            # average subgroup score
            if subgroup_data["fields"]:
                subgroup_score /= len(subgroup_data["fields"])

            # apply subgroup weight
            pillar_score += subgroup_score * subgroup_weight

        # apply pillar weight (scale 0–100)
        results[pillar] = round(pillar_score * p_weight * 100, 2)
        total_score += results[pillar]

    results["Total"] = round(total_score, 2)
    return results

# ------------------------------
# Main Test
# ------------------------------
if __name__ == "__main__":
    # Example: test on already extracted JSON
    company_data = load_json("extracted_test/CIPLA_2024-25_ESG_metrics.json")
    pillar_weights = load_json("weights/pillar_weights.json")
    benchmarks = load_json("weights/benchmarks.json")

    scores = compute_weighted_score(company_data, pillar_weights, benchmarks)

    print("\n✅ Final Scores:")
    print(json.dumps(scores, indent=2))
