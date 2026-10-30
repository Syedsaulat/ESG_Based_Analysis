# app.py
import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pipeline import run_pipeline, run_pipeline_batch  # backend functions

# ---------- CONFIG ----------
RAW_FOLDER = "raw_input_txt"
CLEANED_FOLDER = "cleaned_input_txt"
EXTRACTED_FOLDER = "extracted"
SCORES_FOLDER = "scores"

for d in [RAW_FOLDER, CLEANED_FOLDER, EXTRACTED_FOLDER, SCORES_FOLDER]:
    os.makedirs(d, exist_ok=True)

st.set_page_config(
    page_title="ESG Scoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌱 ESG Scoring Dashboard")
st.markdown(
    "Upload raw ESG report `.txt` files, run your Groq-powered pipeline, "
    "and explore **Environment, Social, Governance** scores interactively."
)

# ---------- HELPERS ----------

def save_uploaded_files(files):
    """Save uploaded txt files into raw_input_txt."""
    saved_paths = []
    for f in files:
        dest = os.path.join(RAW_FOLDER, f.name)
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        saved_paths.append(dest)
    return saved_paths


def load_scores_for_company(company: str):
    """Load scores JSON for a company if it exists."""
    p = os.path.join(SCORES_FOLDER, f"{company}_scores.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def compute_risk_score(total_esg: float | None) -> float | None:
    """
    Map Total ESG (higher = better) to a Sustainalytics-style risk score (lower = better).

    Handles both 0–50 and 0–100 Total ESG scales:
      - If Total ESG <= 50: assume 0–50 scale → risk = 50 - total_esg
      - If Total ESG > 50: assume 0–100 scale → normalize to 0–50 first
            normalized_total = total_esg / 2
            risk = 50 - normalized_total

    Result is clamped to [0, 50]. Returns None if input is invalid.
    """
    if total_esg is None:
        return None
    try:
        total_val = float(total_esg)
    except Exception:
        return None

    # Normalize to 0–50 scale if needed
    if total_val > 50:
        total_val = total_val / 2.0

    risk = 50.0 - total_val
    risk = max(0.0, min(50.0, risk))
    return round(risk, 2)


def map_risk_category(risk_score: float | None) -> str:
    """Map numeric risk score to category like Sustainalytics."""
    if risk_score is None:
        return "Unknown"
    if 0 <= risk_score < 10:
        return "Negligible"
    if 10 <= risk_score < 20:
        return "Low"
    if 20 <= risk_score < 30:
        return "Medium"
    if 30 <= risk_score < 40:
        return "High"
    return "Severe"


def plot_risk_gauge(risk_score: float):
    """
    Single horizontal stacked bar 0–50 with colored segments and an arrow at risk_score.
    """
    categories = [
        ("Negligible", 0, 10, "#22c55e"),  # green
        ("Low",        10, 20, "#a3e635"),
        ("Medium",     20, 30, "#facc15"),
        ("High",       30, 40, "#fb923c"),
        ("Severe",     40, 50, "#ef4444"),
    ]

    fig = go.Figure()

    # One row: “ESG risk”, stacked horizontally from 0 → 50
    for label, x0, x1, color in categories:
        width = x1 - x0
        fig.add_trace(
            go.Bar(
                x=[width],
                y=["ESG risk"],
                base=x0,
                orientation="h",
                marker_color=color,
                name=label,
                hovertemplate=f"{label}: {x0}–{x1}<extra></extra>",
                showlegend=True,
            )
        )

    fig.update_layout(
        barmode="stack",
        showlegend=True,
        xaxis=dict(range=[0, 50], title="Risk score (0 = best, 50 = worst)"),
        yaxis=dict(title=""),
        height=260,
        margin=dict(l=60, r=60, t=30, b=40),
    )

    # ▶ arrow on the bar at the exact score
    fig.add_annotation(
        x=risk_score,
        y="ESG risk",
        text="▼",
        showarrow=False,
        font=dict(size=16, color="black"),
        xshift=0,
        yshift=12,
    )

    return fig


# ---------- SIDEBAR: UPLOAD & RUN ----------
st.sidebar.header("Upload / Pipeline")

# Pipeline status indicator
st.sidebar.markdown("---")
st.sidebar.markdown("**🖥️ Local GPU Inference**")

# Check for local GPU (lazy import to avoid watcher conflict)
try:
    import subprocess
    result = subprocess.run(
        ["python", "-c", "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"],
        capture_output=True, text=True, timeout=10
    )
    lines = result.stdout.strip().split('\n')
    cuda_available = lines[0] == "True" if lines else False
    gpu_name = lines[1] if len(lines) > 1 else "Unknown"
    
    if cuda_available:
        st.sidebar.success(f"🟢 GPU: {gpu_name}")
        st.sidebar.caption("100% Local - No cloud API needed")
    else:
        st.sidebar.error("❌ No GPU detected")
        st.sidebar.caption("CUDA required for inference")
except Exception as e:
    st.sidebar.warning("⚠️ Could not detect GPU")

st.sidebar.caption("TinyLlama 1.1B model on GPU")
st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more raw `.txt` ESG reports",
    type=["txt"],
    accept_multiple_files=True,
)

run_uploaded_btn = st.sidebar.button("🚀 Run pipeline for uploaded files")
run_existing_btn = st.sidebar.button("⚙️ Run pipeline for all files in `raw_input_txt/`")

# ---------- RUN PIPELINE ON UPLOADED ----------
if run_uploaded_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one `.txt` file.")
    else:
        saved_paths = save_uploaded_files(uploaded_files)
        st.sidebar.success(f"Saved {len(saved_paths)} file(s) to `{RAW_FOLDER}/`.")
        progress = st.sidebar.progress(0)
        for i, raw_path in enumerate(saved_paths):
            company = Path(raw_path).stem
            st.sidebar.info(f"Processing {company} ...")
            try:
                run_pipeline(raw_path)
                st.sidebar.success(f"{company}: ✅ Done")
            except Exception as e:
                st.sidebar.error(f"{company}: ❌ Error — {e}")
            progress.progress(int((i + 1) / len(saved_paths) * 100))
        progress.empty()
        st.sidebar.success("All uploaded files processed.")

# ---------- RUN PIPELINE ON ALL RAW FILES ----------
if run_existing_btn:
    raw_files = sorted(
        f for f in os.listdir(RAW_FOLDER)
        if f.endswith(".txt")
    )
    if not raw_files:
        st.sidebar.warning(f"No `.txt` files found in `{RAW_FOLDER}/`.")
    else:
        raw_paths = [os.path.join(RAW_FOLDER, fname) for fname in raw_files]
        
        # Use batch processing for speed when multiple files
        if len(raw_files) > 1:
            st.sidebar.info(f"⚡ Batch processing {len(raw_files)} files in parallel...")
            progress = st.sidebar.progress(0)
            
            import time
            start = time.time()
            
            try:
                results = run_pipeline_batch(raw_paths, max_workers=2)
                elapsed = time.time() - start
                
                success = sum(1 for r in results.values() if r == "Success")
                st.sidebar.success(f"✅ {success}/{len(raw_files)} files processed in {elapsed:.1f}s")
                
                for company, status in results.items():
                    if status == "Success":
                        st.sidebar.success(f"{company}: ✅ Done")
                    else:
                        st.sidebar.error(f"{company}: ❌ {status}")
                        
            except Exception as e:
                st.sidebar.error(f"Batch error: {e}")
            finally:
                progress.progress(100)
                progress.empty()
        else:
            # Single file - use regular processing
            progress = st.sidebar.progress(0)
            raw_path = raw_paths[0]
            company = Path(raw_path).stem
            st.sidebar.info(f"Processing existing file: {company} ...")
            try:
                run_pipeline(raw_path)
                st.sidebar.success(f"{company}: ✅ Done")
            except Exception as e:
                st.sidebar.error(f"{company}: ❌ Error — {e}")
            progress.progress(100)
            progress.empty()

# ---------- BUILD SUMMARY DATAFRAME ----------
score_files = sorted(
    f for f in os.listdir(SCORES_FOLDER)
    if f.endswith("_scores.json")
)

rows = []
for sf in score_files:
    company = Path(sf).stem.replace("_scores", "")
    with open(os.path.join(SCORES_FOLDER, sf), "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    # handle both shapes:
    # 1) {"Environment":..,"Social":..,"Governance":..,"Total":..}
    # 2) {"pillar_scores":{...}, "total_esg":..}
    env = data.get("Environment", data.get("pillar_scores", {}).get("Environment"))
    soc = data.get("Social", data.get("pillar_scores", {}).get("Social"))
    gov = data.get("Governance", data.get("pillar_scores", {}).get("Governance"))
    total = data.get("Total",
                     data.get("total_esg",
                              data.get("Total_ESG")))

    risk_score = compute_risk_score(total)
    risk_cat = map_risk_category(risk_score)

    rows.append({
        "Company": company,
        "Environment": env,
        "Social": soc,
        "Governance": gov,
        "Total": total,
        "RiskScore": risk_score,
        "RiskCategory": risk_cat,
    })

if rows:
    df = pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)
else:
    df = pd.DataFrame(columns=["Company", "Environment", "Social", "Governance", "Total", "RiskScore", "RiskCategory"])

# ---------- TABS LAYOUT ----------
tab_overview, tab_details, tab_raw = st.tabs(
    ["📊 Overview", "🏢 Company details", "📂 Files"]
)

# ====== TAB 1: OVERVIEW ======
with tab_overview:
    st.subheader("Processed Company Scores")

    if df.empty:
        st.info("No scores found yet. Upload reports and run the pipeline.")
    else:
        # SAFETY: cast numeric cols so None won't break "{:.2f}"
        numeric_cols = ["Environment", "Social", "Governance", "Total", "RiskScore"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        st.dataframe(
            df.style.format(
                {
                    "Environment": "{:.2f}",
                    "Social": "{:.2f}",
                    "Governance": "{:.2f}",
                    "Total": "{:.2f}",
                    "RiskScore": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        st.subheader("E / S / G comparison")

        molten = df.melt(
            id_vars=["Company", "Total"],
            value_vars=["Environment", "Social", "Governance"],
            var_name="Pillar",
            value_name="Score",
        )
        fig = px.bar(
            molten,
            x="Company",
            y="Score",
            color="Pillar",
            barmode="group",
            title="Pillar scores by company",
        )
        st.plotly_chart(fig, use_container_width=True)

# ====== TAB 2: COMPANY DETAILS ======
with tab_details:
    st.subheader("Company details")

    if df.empty:
        st.info("Run the pipeline first to see company details.")
    else:
        companies = df["Company"].tolist()
        selected_company = st.selectbox("Select a company", companies)

        if selected_company:
            scores = load_scores_for_company(selected_company)
            if scores is None:
                st.warning("Scores file not found for this company.")
            else:
                total = scores.get(
                    "Total",
                    scores.get("Total_ESG", scores.get("total_esg", None)),
                )
                env = scores.get("Environment",
                                 scores.get("pillar_scores", {}).get("Environment", 0))
                soc = scores.get("Social",
                                 scores.get("pillar_scores", {}).get("Social", 0))
                gov = scores.get("Governance",
                                 scores.get("pillar_scores", {}).get("Governance", 0))

                risk_score = compute_risk_score(total)
                risk_cat = map_risk_category(risk_score)

                # Top metrics area: Total ESG + Risk score
                st.metric("⭐ Total ESG Score", f"{total:.2f}" if total is not None else "N/A")

                # Clear label: Sustainalytics-style Risk Score (0–50, lower = better)
                st.markdown("### ESG Risk (Sustainalytics-style)")
                if risk_score is not None:
                    # show metric and category side-by-side
                    rcol1, rcol2 = st.columns([2, 3])
                    with rcol1:
                        st.metric("Sustainalytics-style Risk Score (0–50, lower = better)",
                                  f"{risk_score:.2f}")
                    with rcol2:
                        # coloured category text using simple inline HTML for contrast
                        color_map = {
                            "Negligible": "#16a34a",  # green
                            "Low": "#65a30d",
                            "Medium": "#d97706",
                            "High": "#ea580c",
                            "Severe": "#dc2626",
                            "Unknown": "#6b7280",
                        }
                        cat_color = color_map.get(risk_cat, "#6b7280")
                        # Using unsafe_allow_html to allow colored span — Streamlit permits this
                        st.markdown(
                            f"<div style='font-size:18px;padding:6px 10px;border-radius:8px;"
                            f"background:#f3f4f6;display:inline-block;'>"
                            f"<strong>Category:</strong> "
                            f"<span style='color:{cat_color};font-weight:700'>{risk_cat}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Sustainalytics-style Risk Score not available for this company.")

                # Optional short legend so users understand the mapping (0 best → 50 worst)
                st.caption("**Legend:** 0 = Negligible risk (best), 50 = Severe risk (worst). Lower is better.")

                col1, col2 = st.columns([2, 1])

                with col1:
                    # pillar bar chart
                    pillar_df = pd.DataFrame({
                        "Pillar": ["Environment", "Social", "Governance"],
                        "Score": [env or 0, soc or 0, gov or 0],
                    })
                    fig2 = px.bar(
                        pillar_df,
                        x="Pillar",
                        y="Score",
                        title=f"Pillar scores for {selected_company}",
                        text="Score",
                        range_y=[0, 100],
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    # radar chart
                    radar_df = pillar_df.copy()
                    fig_radar = px.line_polar(
                        radar_df,
                        r="Score",
                        theta="Pillar",
                        line_close=True,
                        title="ESG profile (radar)",
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col2:
                    if risk_score is not None:
                        st.markdown("#### ESG Risk (Sustainalytics-style scale)")
                        fig_risk = plot_risk_gauge(risk_score)
                        st.plotly_chart(fig_risk, use_container_width=True)

                    st.markdown("#### Downloads")
                    # scores JSON
                    st.download_button(
                        "Download scores JSON",
                        data=json.dumps(scores, indent=2),
                        file_name=f"{selected_company}_scores.json",
                        mime="application/json",
                    )

                    # extracted metrics JSON
                    extracted_path = os.path.join(
                        EXTRACTED_FOLDER,
                        f"{selected_company}_metrics.json",
                    )
                    if os.path.exists(extracted_path):
                        with open(extracted_path, "r", encoding="utf-8", errors="ignore") as f:
                            st.download_button(
                                "Download extracted metrics JSON",
                                data=f.read(),
                                file_name=f"{selected_company}_metrics.json",
                                mime="application/json",
                            )

                    # cleaned text
                    cleaned_path = os.path.join(
                        CLEANED_FOLDER,
                        f"{selected_company}.txt",
                    )
                    if os.path.exists(cleaned_path):
                        with open(cleaned_path, "r", encoding="utf-8", errors="ignore") as f:
                            st.download_button(
                                "Download cleaned text",
                                data=f.read(),
                                file_name=f"{selected_company}.txt",
                                mime="text/plain",
                            )

# ====== TAB 3: RAW & FILES ======
with tab_raw:
    st.subheader("Files on disk")

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("**Raw input (`raw_input_txt/`)**")
        raw_files = sorted(f for f in os.listdir(RAW_FOLDER) if f.endswith(".txt"))
        if raw_files:
            st.write(raw_files)
        else:
            st.caption("No raw input files found.")

    with colB:
        st.markdown("**Extracted metrics (`extracted/`)**")
        extr_files = sorted(f for f in os.listdir(EXTRACTED_FOLDER) if f.endswith(".json"))
        if extr_files:
            st.write(extr_files)
        else:
            st.caption("No extracted metrics yet.")

    with colC:
        st.markdown("**Scores (`scores/`)**")
        score_files_only = sorted(f for f in os.listdir(SCORES_FOLDER) if f.endswith(".json"))
        if score_files_only:
            st.write(score_files_only)
        else:
            st.caption("No scores saved yet.")
