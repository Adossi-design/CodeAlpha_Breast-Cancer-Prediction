"""
FredCare AI: Breast Cancer Clinical Screening Tool
Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import streamlit as st

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FredCare AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Palette ──────────────────────────────────────────────────────────────────
BLUE_DARK   = "#0D3B7A"
BLUE_MID    = "#1558A8"
BLUE_LIGHT  = "#2577D4"
BLUE_SOFT   = "#E8F0FB"
BLUE_PALE   = "#F0F5FF"
ACCENT_CYAN = "#0EA5E9"
TEXT        = "#0F172A"
MUTED       = "#475569"
BORDER      = "#BFCFEA"
WHITE       = "#FFFFFF"
GREEN       = "#15803D"
GREEN_SOFT  = "#F0FDF4"
RED         = "#B91C1C"
RED_SOFT    = "#FFF1F2"

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, [class*="css"] {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: {TEXT};
  }}

  /* ── Page background ── */
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  section[data-testid="stMain"] {{
    background-color: {BLUE_PALE} !important;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"],
  [data-testid="stSidebar"] > div {{
    background: linear-gradient(180deg, {BLUE_DARK} 0%, {BLUE_MID} 100%) !important;
    border-right: none !important;
  }}
  [data-testid="stSidebar"] label {{
    color: #CBD8F0 !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
  }}
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span {{
    color: #CBD8F0 !important;
  }}
  [data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.15) !important;
  }}

  /* ── Number Inputs ── */
  [data-testid="stNumberInput"] input,
  [data-testid="stNumberInput"] input[type="number"],
  [data-testid="stNumberInput"] > div input {{
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    border: 1.5px solid rgba(255,255,255,0.50) !important;
    border-radius: 7px !important;
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    opacity: 1 !important;
  }}
  [data-testid="stNumberInput"] input::placeholder {{
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
  }}
  [data-testid="stNumberInput"] input:focus,
  [data-testid="stNumberInput"] input[type="number"]:focus {{
    border-color: {ACCENT_CYAN} !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.30) !important;
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
  }}
  [data-testid="stNumberInput"] button {{
    background: rgba(255,255,255,0.20) !important;
    color: white !important;
    border-color: rgba(255,255,255,0.25) !important;
  }}

  /* ── Run Analysis button ── */
  div[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, #0EA5E9 0%, #38BDF8 100%) !important;
    color: {BLUE_DARK} !important;
    font-weight: 800 !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 18px rgba(14,165,233,0.40) !important;
  }}
  div[data-testid="stButton"] > button:hover {{
    background: linear-gradient(135deg, #38BDF8 0%, #7DD3FC 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(14,165,233,0.50) !important;
  }}

  /* ── Section labels ── */
  .section-label {{
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {BLUE_MID};
    border-bottom: 2px solid {BLUE_LIGHT};
    padding-bottom: 7px;
    margin-bottom: 18px;
    margin-top: 6px;
  }}

  /* ── Sidebar group headers ── */
  .feat-group {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {ACCENT_CYAN};
    margin: 18px 0 10px 0;
  }}

  /* ── Diagnosis cards ── */
  .dx-malignant {{
    background: linear-gradient(135deg, #7F1D1D 0%, {RED} 100%);
    border-radius: 14px;
    padding: 26px 24px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 24px rgba(185,28,28,0.30);
  }}
  .dx-benign {{
    background: linear-gradient(135deg, #14532D 0%, {GREEN} 100%);
    border-radius: 14px;
    padding: 26px 24px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 24px rgba(21,128,61,0.30);
  }}

  /* ── Info cards ── */
  .info-card {{
    background: {WHITE};
    border: 1.5px solid {BORDER};
    border-top: 4px solid {BLUE_LIGHT};
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(21,88,168,0.08);
    height: 100%;
  }}
  .card-title {{
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: {BLUE_LIGHT};
    margin-bottom: 14px;
  }}

  /* ── Confidence number ── */
  .conf-num {{
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.0;
    margin-bottom: 10px;
    text-align: center;
  }}

  /* ── Progress bars ── */
  [data-testid="stProgressBar"] > div {{
    background: #DBEAFE !important;
    border-radius: 6px !important;
    height: 10px !important;
  }}
  [data-testid="stProgressBar"] > div > div {{
    border-radius: 6px !important;
  }}

  /* ── Clinical note ── */
  .clinical-note {{
    background: {WHITE};
    border-left: 5px solid {BLUE_LIGHT};
    border-radius: 0 14px 14px 0;
    padding: 24px 28px;
    font-size: 0.91rem;
    line-height: 1.82;
    color: {TEXT};
    box-shadow: 0 3px 14px rgba(21,88,168,0.09);
    margin-bottom: 8px;
  }}

  /* ── Sidebar warning ── */
  [data-testid="stWarning"] {{
    background: rgba(251,191,36,0.15) !important;
    border: 1px solid rgba(251,191,36,0.50) !important;
    border-radius: 8px !important;
    font-size: 0.80rem !important;
    color: #FEF3C7 !important;
  }}

  /* ── Welcome card ── */
  .welcome-card {{
    background: {WHITE};
    border-radius: 20px;
    overflow: hidden;
    max-width: 680px;
    margin: 56px auto 0;
    box-shadow: 0 6px 32px rgba(21,88,168,0.12);
  }}
  .welcome-header {{
    background: linear-gradient(135deg, {BLUE_DARK} 0%, {BLUE_LIGHT} 100%);
    padding: 40px 44px 32px;
    text-align: center;
    color: white;
  }}
  .welcome-header h2 {{
    font-size: 1.7rem;
    font-weight: 900;
    margin: 14px 0 8px;
    color: white;
  }}
  .welcome-header p {{
    font-size: 0.80rem;
    color: rgba(255,255,255,0.75);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
  }}
  .welcome-body {{
    padding: 30px 44px 36px;
    text-align: center;
  }}
  .welcome-body p {{
    color: {MUTED};
    font-size: 0.92rem;
    line-height: 1.76;
    margin: 0;
  }}
  .welcome-step {{
    display: flex;
    align-items: center;
    gap: 14px;
    background: {BLUE_SOFT};
    border-radius: 10px;
    padding: 12px 18px;
    margin-top: 16px;
    text-align: left;
  }}
  .step-num {{
    background: {BLUE_LIGHT};
    color: white;
    font-weight: 800;
    font-size: 0.85rem;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }}
  .step-txt {{
    color: {TEXT};
    font-size: 0.86rem;
    line-height: 1.45;
  }}

  hr {{ border-color: {BORDER} !important; margin: 22px 0; }}
  footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ─── Feature Specification ────────────────────────────────────────────────────
SECTION_1 = [
    ("mean radius",    6.98,    28.11,  0.01,  "%.2f"),
    ("mean perimeter", 43.79,  188.50,  0.1,   "%.1f"),
    ("mean area",     143.50, 2501.0,   1.0,   "%.1f"),
    ("mean texture",    9.71,   39.28,  0.01,  "%.2f"),
]
SECTION_2 = [
    ("mean smoothness",        0.053, 0.163, 0.001, "%.3f"),
    ("mean compactness",       0.019, 0.345, 0.001, "%.3f"),
    ("mean symmetry",          0.106, 0.304, 0.001, "%.3f"),
    ("mean fractal dimension", 0.050, 0.097, 0.001, "%.3f"),
]
SECTION_3 = [
    ("mean concavity",      0.0, 0.427, 0.001, "%.3f"),
    ("mean concave points", 0.0, 0.201, 0.001, "%.3f"),
]
ALL_FEATURES = [r[0] for r in SECTION_1 + SECTION_2 + SECTION_3]


# ─── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FredCare AI model…")
def build_model():
    data = load_breast_cancer()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    y    = data.target
    X    = df[ALL_FEATURES]

    medians = X.median()
    q05     = X.quantile(0.05)
    q95     = X.quantile(0.95)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    scaler_eval = StandardScaler()
    X_tr_s = scaler_eval.fit_transform(X_tr)

    model_eval = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    model_eval.fit(X_tr_s, y_tr)

    scaler_prod = StandardScaler()
    X_prod_s    = scaler_prod.fit_transform(X)
    model_prod  = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    model_prod.fit(X_prod_s, y)

    return {
        "model":   model_prod,
        "scaler":  scaler_prod,
        "medians": medians,
        "q05":     q05,
        "q95":     q95,
    }


# ─── Clinical Note ────────────────────────────────────────────────────────────
def clinical_note(is_mal: bool, conf: float) -> str:
    if is_mal:
        if conf >= 0.90:
            return (
                "<b>High confidence malignant prediction.</b> The submitted measurements "
                "display strong patterns associated with malignant tissue across multiple "
                "dimensions. Model confidence exceeds 90%.<br><br>"
                "<b>Recommended action:</b> Immediate referral to an oncologist is advised. "
                "Confirmatory biopsy and advanced imaging, including mammography, ultrasound, or MRI, "
                "should be prioritised without delay. Early stage intervention is critical "
                "to treatment outcome."
            )
        elif conf >= 0.70:
            return (
                "<b>Moderate confidence malignant prediction.</b> Several measurements fall "
                "within ranges frequently associated with malignant tissue, though model "
                "confidence is in the moderate range (70–90%).<br><br>"
                "<b>Recommended action:</b> Prompt specialist consultation is warranted. "
                "Confirmatory diagnostic tests such as biopsy or imaging are strongly recommended "
                "before any clinical decision is finalised."
            )
        else:
            return (
                "<b>Low confidence malignant prediction.</b> The model leans malignant but "
                "with uncertainty below 70%, suggesting borderline or atypical values.<br><br>"
                "<b>Recommended action:</b> Exercise clinical caution. This result must be "
                "interpreted alongside the full clinical picture. Additional diagnostics and "
                "expert review are necessary before drawing any conclusion."
            )
    else:
        if conf >= 0.90:
            return (
                "<b>High confidence benign prediction.</b> The submitted measurements are "
                "strongly consistent with benign tissue characteristics. Model confidence "
                "exceeds 90%.<br><br>"
                "<b>Recommended action:</b> This result is reassuring. Routine annual "
                "follow up imaging is advised as standard precautionary practice. "
                "This result does not replace a comprehensive clinical evaluation."
            )
        elif conf >= 0.70:
            return (
                "<b>Moderate confidence benign prediction.</b> Measurements align "
                "predominantly with benign patterns, though model confidence is in the "
                "moderate range (70–90%).<br><br>"
                "<b>Recommended action:</b> A professional clinical review is recommended "
                "to confirm this assessment. Continued monitoring over time is advisable."
            )
        else:
            return (
                "<b>Low confidence benign prediction.</b> The model leans benign but "
                "with limited confidence, indicating borderline or atypical values.<br><br>"
                "<b>Recommended action:</b> Do not treat this as a definitive benign "
                "finding. Expert evaluation and supplementary diagnostic tests are essential "
                "before any clinical decision is made."
            )


# ─── Sidebar Input Builder ────────────────────────────────────────────────────
def sidebar_inputs(section, medians, q05, q95, warnings_out):
    inputs = {}
    for feat, fmin, fmax, step, fmt in section:
        raw     = float(medians[feat])
        default = round(fmin + round((raw - fmin) / step) * step, 8)
        default = min(max(default, fmin), fmax)

        val = st.sidebar.number_input(
            label=feat.title(),
            min_value=float(fmin),
            max_value=float(fmax),
            value=float(default),
            step=float(step),
            format=fmt,
            key=f"ni_{feat}",
        )
        inputs[feat] = val
        if val < float(q05[feat]) * 0.80 or val > float(q95[feat]) * 1.20:
            warnings_out.append(feat.title())
    return inputs


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    art = build_model()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        f"<div style='text-align:center;padding:24px 0 20px;'>"
        f"<div style='font-size:2.8rem;'>🩺</div>"
        f"<div style='font-size:1.50rem;font-weight:900;color:#FFFFFF;"
        f"letter-spacing:0.5px;margin-top:6px;'>FredCare AI</div>"
        f"<div style='font-size:0.68rem;color:rgba(255,255,255,0.60);letter-spacing:2.5px;"
        f"text-transform:uppercase;margin-top:5px;'>Clinical Screening Tool</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    warnings_list = []

    st.sidebar.markdown(
        "<div class='feat-group'>📐 Size Measurements</div>",
        unsafe_allow_html=True,
    )
    inputs = sidebar_inputs(SECTION_1, art["medians"], art["q05"], art["q95"], warnings_list)

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "<div class='feat-group'>🔷 Shape &amp; Texture</div>",
        unsafe_allow_html=True,
    )
    inputs.update(sidebar_inputs(SECTION_2, art["medians"], art["q05"], art["q95"], warnings_list))

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "<div class='feat-group'>🌀 Concavity</div>",
        unsafe_allow_html=True,
    )
    inputs.update(sidebar_inputs(SECTION_3, art["medians"], art["q05"], art["q95"], warnings_list))

    st.sidebar.markdown("---")

    if warnings_list:
        st.sidebar.warning(
            f"⚠ {len(warnings_list)} value(s) outside the expected clinical range:\n"
            + ",  ".join(warnings_list)
        )

    run = st.sidebar.button("🔬  Run Analysis", use_container_width=True)

    st.sidebar.markdown(
        "<p style='color:rgba(255,255,255,0.40);font-size:0.67rem;text-align:center;"
        "margin-top:20px;line-height:1.65;'>"
        "Wisconsin Breast Cancer Dataset<br>"
        "569 clinical records · 10 features</p>",
        unsafe_allow_html=True,
    )

    # ── Main Panel ────────────────────────────────────────────────────────────
    if not run:
        st.markdown("""
        <div class="welcome-card">
          <div class="welcome-header">
            <div style="font-size:3.2rem;">🩺</div>
            <h2>Welcome to FredCare AI</h2>
            <p>Breast Cancer Clinical Screening Tool</p>
          </div>
          <div class="welcome-body">
            <p>
              A clinical decision support tool powered by Logistic Regression,
              trained on the Wisconsin Breast Cancer dataset. Enter the patient's
              tumor measurements and receive an instant malignancy assessment
              with confidence scoring.
            </p>
            <div class="welcome-step">
              <div class="step-num">1</div>
              <div class="step-txt">
                <b>Enter patient measurements:</b> fill in the 10 tumor fields
                in the sidebar on the left.
              </div>
            </div>
            <div class="welcome-step">
              <div class="step-num">2</div>
              <div class="step-txt">
                <b>Run the analysis:</b> click the
                <em>Run Analysis</em> button at the bottom of the sidebar.
              </div>
            </div>
            <div class="welcome-step">
              <div class="step-num">3</div>
              <div class="step-txt">
                <b>Review the result:</b> the diagnosis, confidence level,
                and clinical recommendation will appear here.
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Prediction ────────────────────────────────────────────────────────────
    X_in  = pd.DataFrame([inputs], columns=ALL_FEATURES)
    X_sc  = art["scaler"].transform(X_in)
    pred  = art["model"].predict(X_sc)[0]
    proba = art["model"].predict_proba(X_sc)[0]

    prob_mal = float(proba[0])
    prob_ben = float(proba[1])
    is_mal   = (pred == 0)
    conf     = max(prob_mal, prob_ben)
    conf_color = RED if is_mal else GREEN

    # ── Header banner ─────────────────────────────────────────────────────────
    label  = "MALIGNANT" if is_mal else "BENIGN"
    icon   = "🔴" if is_mal else "🟢"
    banner_bg  = "linear-gradient(135deg, #7F1D1D 0%, #B91C1C 100%)" if is_mal \
                 else "linear-gradient(135deg, #14532D 0%, #15803D 100%)"
    st.markdown(
        f"<div style='background:{banner_bg};border-radius:16px;padding:28px 36px;"
        f"display:flex;align-items:center;gap:20px;margin-bottom:28px;"
        f"box-shadow:0 6px 28px rgba(0,0,0,0.15);'>"
        f"<div style='font-size:3rem;'>{icon}</div>"
        f"<div>"
        f"<div style='font-size:0.72rem;color:rgba(255,255,255,0.70);letter-spacing:3px;"
        f"text-transform:uppercase;margin-bottom:4px;'>Screening Result</div>"
        f"<div style='font-size:2.0rem;font-weight:900;color:#FFFFFF;"
        f"letter-spacing:1px;'>{label}</div>"
        f"</div>"
        f"<div style='margin-left:auto;text-align:right;'>"
        f"<div style='font-size:0.72rem;color:rgba(255,255,255,0.70);letter-spacing:2px;"
        f"text-transform:uppercase;margin-bottom:4px;'>Confidence</div>"
        f"<div style='font-size:2.2rem;font-weight:900;color:#FFFFFF;'>{conf*100:.1f}%</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Confidence + Probability row ──────────────────────────────────────────
    st.markdown(
        "<div class='section-label'>Patient Screening Details</div>",
        unsafe_allow_html=True,
    )

    col_conf, col_prob = st.columns(2, gap="large")

    with col_conf:
        st.markdown(
            f"<div class='info-card'>"
            f"<div class='card-title'>Confidence Score</div>"
            f"<div class='conf-num' style='color:{conf_color};'>{conf*100:.1f}%</div>",
            unsafe_allow_html=True,
        )
        st.progress(float(conf))
        st.markdown(
            f"<div style='color:{MUTED};font-size:0.78rem;margin-top:10px;'>"
            f"The model assigns a <b>{conf*100:.1f}%</b> probability to the "
            f"<b>{'Malignant' if is_mal else 'Benign'}</b> class for this patient's "
            f"measurements.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_prob:
        st.markdown(
            f"<div class='info-card'>"
            f"<div class='card-title'>Class Probability Breakdown</div>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:5px;'>"
            f"<span style='color:{GREEN};font-size:0.85rem;font-weight:700;'>🟢 Benign</span>"
            f"<span style='color:{TEXT};font-size:1.0rem;font-weight:800;'>{prob_ben*100:.2f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(float(prob_ben))
        st.markdown(
            f"<div style='height:12px;'></div>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:5px;'>"
            f"<span style='color:{RED};font-size:0.85rem;font-weight:700;'>🔴 Malignant</span>"
            f"<span style='color:{TEXT};font-size:1.0rem;font-weight:800;'>{prob_mal*100:.2f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(float(prob_mal))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Clinical Assessment ───────────────────────────────────────────────────
    st.markdown(
        "<div class='section-label'>Clinical Assessment &amp; Recommendation</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='clinical-note'>{clinical_note(is_mal, conf)}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.70rem;text-align:center;'>"
        f"FredCare AI &nbsp;·&nbsp; For educational and research purposes only &nbsp;·&nbsp; "
        f"Not a substitute for professional medical diagnosis or clinical judgement</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
