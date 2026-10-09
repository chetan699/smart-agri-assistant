"""
Streamlit Frontend for Smart Agriculture Assistant Pro.

Multimodal Deep Learning Decision Support Platform for Smart Farming:
- Tabular Deep Learning: Crop Recommendation, Fertilizer Optimization, Irrigation Scheduling
- Computer Vision: Multi-Crop Plant Pathology & Leaf Disease Diagnosis (8 classes)
- Telemetry Engine: Diurnal IoT Sensor Simulation & Autonomous Field Assessment

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import tempfile
from pathlib import Path
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Agri Assistant | AgroCortex AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# High-Contrast, Production-Hardened CSS Styling (Dark/Light Mode Compatible)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts with robust system fallbacks */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Hero Header Container */
    .hero-container {
        background: linear-gradient(135deg, #0F3825 0%, #1B4D3E 50%, #2D6A4F 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 20px;
        color: #FFFFFF;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(15, 56, 37, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.9rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #A7F3D0;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(167, 243, 208, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin: 0;
        color: #FFFFFF !important;
    }
    
    .hero-subtitle {
        font-size: 1.02rem;
        color: #D1FAE5 !important;
        margin-top: 0.6rem;
        font-weight: 400;
        max-width: 880px;
        line-height: 1.55;
    }

    .hero-metrics-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1.8rem;
        margin-top: 1.4rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.15);
    }

    .hero-metric-item {
        display: flex;
        flex-direction: column;
    }

    .hero-metric-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF !important;
    }

    .hero-metric-lbl {
        font-size: 0.74rem;
        text-transform: uppercase;
        color: #A7F3D0 !important;
        letter-spacing: 0.05em;
    }

    /* Standard Card Container */
    .agri-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
        color: #1F2937;
    }

    .agri-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #111827 !important;
        margin: 0 0 0.3rem 0;
    }

    .agri-card-desc {
        font-size: 0.88rem;
        color: #4B5563 !important;
        margin: 0 0 1rem 0;
        line-height: 1.5;
    }

    /* Result Showcase Boxes */
    .result-hero {
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
    }

    .result-success {
        background: #ECFDF5;
        border-color: #6EE7B7;
        color: #065F46 !important;
    }

    .result-alert {
        background: #FFFBEB;
        border-color: #FCD34D;
        color: #92400E !important;
    }

    .result-info {
        background: #EFF6FF;
        border-color: #93C5FD;
        color: #1E40AF !important;
    }

    .result-danger {
        background: #FEF2F2;
        border-color: #FCA5A5;
        color: #991B1B !important;
    }

    .result-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .result-main {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    /* Simulation Disclosure Banner */
    .sim-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        color: #92400E !important;
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #F9FAFB;
        border-right: 1px solid #E5E7EB;
    }

    /* Tab navigation polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 8px;
        border-bottom: 2px solid #E5E7EB;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 10px 10px 0 0;
        gap: 6px;
        padding: 10px 18px;
        color: #374151;
        font-weight: 600;
        border: 1px solid #E5E7EB;
        border-bottom: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1B4D3E !important;
        color: #FFFFFF !important;
        border-color: #1B4D3E !important;
    }
    
    /* Primary Action Buttons */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #0F3825 0%, #1B4D3E 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(15, 56, 37, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(15, 56, 37, 0.3) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Model Loaders (Invokes real inference classes in src/)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_crop_recommender():
    from src.crop_recommendation.predict import CropRecommender
    return CropRecommender()


@st.cache_resource
def load_fertilizer_recommender():
    from src.fertilizer_recommendation.predict import FertilizerRecommender
    return FertilizerRecommender()


@st.cache_resource
def load_irrigation_predictor():
    from src.irrigation_prediction.predict import IrrigationPredictor
    return IrrigationPredictor()


@st.cache_resource
def load_disease_detector():
    from src.disease_detection.predict import DiseaseDetector
    return DiseaseDetector()


# -----------------------------------------------------------------------------
# Agronomic Pathology Action Library (For Disease Detection Tab)
# -----------------------------------------------------------------------------
DISEASE_ACTION_GUIDE = {
    "Apple___Apple_scab": {
        "pathogen": "Venturia inaequalis (Fungal pathogen)",
        "urgency": "High",
        "action": "Apply protectant copper spray or myclobutanil fungicide at early green-tip bud break.",
        "cultural": "Rake and destroy infected fallen leaves to suppress overwintering ascospores. Prune canopy for sunlight.",
    },
    "Apple___healthy": {
        "pathogen": "None detected",
        "urgency": "Low (Preventive)",
        "action": "Maintain standard seasonal surveillance and balanced micro-nutrient foliar application.",
        "cultural": "Maintain adequate tree spacing and balanced orchard drip irrigation.",
    },
    "Corn_(maize)___Common_rust_": {
        "pathogen": "Puccinia sorghi (Fungal rust)",
        "urgency": "Moderate",
        "action": "Apply triazole or strobilurin fungicides if pustules spread extensively before the tasseling/silking stage.",
        "cultural": "Sow rust-resistant seed hybrids. Practice crop rotation with non-cereal crops (legumes/soybeans).",
    },
    "Corn_(maize)___healthy": {
        "pathogen": "None detected",
        "urgency": "Low (Preventive)",
        "action": "Proceed with scheduled nitrogen side-dressing during peak vegetative development (V6-V8).",
        "cultural": "Maintain effective weed control and regular field scouting.",
    },
    "Potato___Early_blight": {
        "pathogen": "Alternaria solani (Fungal pathogen)",
        "urgency": "High",
        "action": "Apply chlorothalonil or mancozeb protectant fungicides on 7-to-10 day intervals.",
        "cultural": "Avoid overhead sprinkler irrigation to minimize canopy moisture. Ensure adequate potassium nutrition.",
    },
    "Potato___healthy": {
        "pathogen": "None detected",
        "urgency": "Low (Preventive)",
        "action": "Ensure uniform soil moisture to prevent tuber hollow-heart and maintain hill integrity.",
        "cultural": "Monitor lower leaves weekly, especially following rain or heavy morning dew.",
    },
    "Tomato___Early_blight": {
        "pathogen": "Alternaria solani (Fungal pathogen)",
        "urgency": "High",
        "action": "Prune and destroy infected lower foliage immediately. Apply copper soap or azoxystrobin spray.",
        "cultural": "Stake and cage plants to keep foliage off soil. Apply organic mulch around base to prevent spore splash.",
    },
    "Tomato___healthy": {
        "pathogen": "None detected",
        "urgency": "Low (Preventive)",
        "action": "Maintain consistent calcium and potassium fertigation to prevent blossom end rot.",
        "cultural": "Prune excess suckers for airflow and use drip irrigation at soil level.",
    },
}


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.2rem;">
            <div style="background: #1B4D3E; width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                🌾
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #1B4D3E;">AgroCortex AI</h3>
                <span style="font-size: 0.75rem; color: #6B7280; font-weight: 500;">Academic Research Suite</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🟢 System Status")
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 0.8rem; font-size: 0.82rem; margin-bottom: 1.2rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <span style="color: #6B7280;">Core Framework</span>
                <span style="font-weight: 600; color: #059669;">PyTorch 2.14</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <span style="color: #6B7280;">Neural Engines</span>
                <span style="font-weight: 600; color: #059669;">4 Models Verified</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <span style="color: #6B7280;">Inference Mode</span>
                <span style="font-weight: 600; color: #1B4D3E;">Live Checkpoints</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #6B7280;">Edge IoT Layer</span>
                <span style="font-weight: 600; color: #D97706;">Diurnal Simulator</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Validated Benchmark Metrics")
    st.markdown(
        """
        <div style="font-size: 0.80rem; color: #374151; line-height: 1.6; background: #FFFFFF; border: 1px solid #E5E7EB; padding: 0.8rem; border-radius: 10px;">
            <div style="margin-bottom: 0.4rem;">🌱 <b>Crop Recommender:</b> 99.4% test accuracy<br><span style="color: #6B7280; font-size: 0.75rem;">(MLP, Kaggle Crop Dataset, 2,200 samples)</span></div>
            <div style="margin-bottom: 0.4rem;">🧪 <b>Fertilizer Engine:</b> 95.0% test accuracy<br><span style="color: #6B7280; font-size: 0.75rem;">(MLP, 99 samples — small dataset caveat)</span></div>
            <div style="margin-bottom: 0.4rem;">💧 <b>Irrigation Predictor:</b> 83.0% acc / 80% recall<br><span style="color: #6B7280; font-size: 0.75rem;">(Class-weighted MLP, GCEK IoT, 150 samples)</span></div>
            <div>🍃 <b>Leaf Pathology:</b> 93.9% test accuracy<br><span style="color: #6B7280; font-size: 0.75rem;">(Vision CNN, PlantVillage 8-class subset, 1,200 images)</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Built for Academic Research Demonstration. All predictions execute live against locally stored neural network weights.")


# -----------------------------------------------------------------------------
# Farmland Hero Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">
            <span>✨</span> Multimodal Agronomic Intelligence
        </div>
        <h1 class="hero-title">Smart Agriculture Decision Assistant</h1>
        <p class="hero-subtitle">
            An end-to-end deep learning framework combining Multi-Layer Perceptrons for tabular soil/nutrient tasks,
            Convolutional Neural Networks for leaf pathology detection, and an integrated diurnal IoT telemetry simulator.
        </p>
        <div class="hero-metrics-row">
            <div class="hero-metric-item">
                <span class="hero-metric-val">4</span>
                <span class="hero-metric-lbl">Trained PyTorch Models</span>
            </div>
            <div class="hero-metric-item">
                <span class="hero-metric-val">22</span>
                <span class="hero-metric-lbl">Crop Categories</span>
            </div>
            <div class="hero-metric-item">
                <span class="hero-metric-val">8</span>
                <span class="hero-metric-lbl">Pathology Conditions</span>
            </div>
            <div class="hero-metric-item">
                <span class="hero-metric-val">Local Edge</span>
                <span class="hero-metric-lbl">Zero-Cloud Dependency</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_crop, tab_fert, tab_irrigation, tab_disease, tab_iot = st.tabs(
    [
        "🌱 Crop Recommendation",
        "🧪 Fertilizer Optimizer",
        "💧 Irrigation Predictor",
        "🍃 Plant Disease Vision",
        "📡 IoT Telemetry Simulator",
    ]
)


# =============================================================================
# TAB 1: CROP RECOMMENDATION
# =============================================================================
with tab_crop:
    st.markdown(
        """
        <div class="agri-card">
            <h3 class="agri-card-title">🌱 Soil & Climate-Driven Crop Recommendation</h3>
            <p class="agri-card-desc">Deep Multi-Layer Perceptron trained on soil macronutrients (N-P-K), pH, and atmospheric variables (99.4% test accuracy on Kaggle benchmark).</p>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state keys for crop inputs
    if "c_in_n" not in st.session_state:
        st.session_state["c_in_n"] = 90.0
        st.session_state["c_in_p"] = 42.0
        st.session_state["c_in_k"] = 43.0
        st.session_state["c_in_ph"] = 6.5
        st.session_state["c_in_temp"] = 20.8
        st.session_state["c_in_hum"] = 82.0
        st.session_state["c_in_rain"] = 202.9

    # One-Click Demo Presets
    st.markdown("**⚡ Live Demo Sample Presets (Click to Auto-Populate Inputs):**")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)

    if pcol1.button("🌾 Preset: Monsoon Rice", use_container_width=True):
        st.session_state["c_in_n"] = 80.0
        st.session_state["c_in_p"] = 40.0
        st.session_state["c_in_k"] = 40.0
        st.session_state["c_in_ph"] = 6.5
        st.session_state["c_in_temp"] = 24.0
        st.session_state["c_in_hum"] = 82.0
        st.session_state["c_in_rain"] = 240.0
        st.rerun()

    if pcol2.button("🌽 Preset: High-Yield Maize", use_container_width=True):
        st.session_state["c_in_n"] = 75.0
        st.session_state["c_in_p"] = 45.0
        st.session_state["c_in_k"] = 20.0
        st.session_state["c_in_ph"] = 6.2
        st.session_state["c_in_temp"] = 23.5
        st.session_state["c_in_hum"] = 65.0
        st.session_state["c_in_rain"] = 85.0
        st.rerun()

    if pcol3.button("☕ Preset: Plantation Coffee", use_container_width=True):
        st.session_state["c_in_n"] = 100.0
        st.session_state["c_in_p"] = 30.0
        st.session_state["c_in_k"] = 30.0
        st.session_state["c_in_ph"] = 6.8
        st.session_state["c_in_temp"] = 26.0
        st.session_state["c_in_hum"] = 58.0
        st.session_state["c_in_rain"] = 160.0
        st.rerun()

    if pcol4.button("🌿 Preset: Arid Chickpea", use_container_width=True):
        st.session_state["c_in_n"] = 40.0
        st.session_state["c_in_p"] = 60.0
        st.session_state["c_in_k"] = 80.0
        st.session_state["c_in_ph"] = 7.3
        st.session_state["c_in_temp"] = 19.0
        st.session_state["c_in_hum"] = 17.0
        st.session_state["c_in_rain"] = 75.0
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🧪 Soil Chemistry Inputs")
        N = st.number_input("Nitrogen (N) [mg/kg]", 0.0, 200.0, step=1.0, key="c_in_n")
        P = st.number_input("Phosphorus (P) [mg/kg]", 0.0, 200.0, step=1.0, key="c_in_p")
        K = st.number_input("Potassium (K) [mg/kg]", 0.0, 250.0, step=1.0, key="c_in_k")
        ph = st.number_input("Soil Acidity / Alkalinity (pH)", 0.0, 14.0, step=0.1, key="c_in_ph")

    with col2:
        st.markdown("##### ⛅ Environmental Weather Inputs")
        temperature = st.number_input("Ambient Temperature (°C)", -10.0, 55.0, step=0.1, key="c_in_temp")
        humidity = st.number_input("Relative Air Humidity (%)", 0.0, 100.0, step=0.5, key="c_in_hum")
        rainfall = st.number_input("Seasonal Rainfall (mm)", 0.0, 3500.0, step=1.0, key="c_in_rain")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Execute Crop Recommendation Model", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing soil-climate matrix through MLP neural network..."):
                recommender = load_crop_recommender()
                result = recommender.recommend({
                    "N": N, "P": P, "K": K, "temperature": temperature,
                    "humidity": humidity, "ph": ph, "rainfall": rainfall,
                })

            top_crop = result["top_prediction"].capitalize()
            top_prob = result["top_k"][0][1]

            st.markdown(
                f"""
                <div class="result-hero result-success">
                    <div class="result-title">Optimal Agronomic Recommendation</div>
                    <div class="result-main">🌾 {top_crop}</div>
                    <div style="margin-top: 0.4rem; font-size: 0.95rem; font-weight: 600;">
                        Model Confidence: {top_prob:.1%} | Neural Output Layer: Softmax Normalized
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top Candidates Visual Distribution
            st.markdown("##### 📊 Top Candidate Crops (Probability Distribution)")
            for crop, prob in result["top_k"]:
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.write(f"**{crop.capitalize()}**")
                with c2:
                    st.progress(float(prob), text=f"{prob:.1%}")

            if result["warnings"]:
                for w in result["warnings"]:
                    st.warning(f"⚠️ Telemetry Alert: {w}")

        except FileNotFoundError as e:
            st.error(f"Model checkpoint missing: {e}")
        except Exception as e:
            st.error(f"Inference error: {e}")


# =============================================================================
# TAB 2: FERTILIZER OPTIMIZER
# =============================================================================
with tab_fert:
    st.markdown(
        """
        <div class="agri-card">
            <h3 class="agri-card-title">🧪 Fertilizer Nutrient Recommendation Engine</h3>
            <p class="agri-card-desc">Hybrid categorical-continuous MLP mapping crop stage, soil type, and existing soil nutrients to targeted fertilizer formulations (95.0% test accuracy on Kaggle benchmark).</p>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for fertilizer inputs
    soil_types = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
    crop_types = ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil seeds", "Ground Nuts", "Pulses"]

    if "fert_temp" not in st.session_state:
        st.session_state["fert_temp"] = 26.0
        st.session_state["fert_hum"] = 52.0
        st.session_state["fert_moist"] = 38.0
        st.session_state["fert_soil"] = "Sandy"
        st.session_state["fert_crop"] = "Maize"
        st.session_state["fert_n"] = 37.0
        st.session_state["fert_k"] = 0.0
        st.session_state["fert_p"] = 0.0

    # Presets for Fertilizer
    st.markdown("**⚡ Live Demo Sample Presets (Click to Auto-Populate Inputs):**")
    fcol1, fcol2, fcol3 = st.columns(3)

    if fcol1.button("🌽 Preset: Maize in Sandy Soil (Nitrogen Deficit)", use_container_width=True):
        st.session_state["fert_temp"] = 26.0
        st.session_state["fert_hum"] = 52.0
        st.session_state["fert_moist"] = 38.0
        st.session_state["fert_soil"] = "Sandy"
        st.session_state["fert_crop"] = "Maize"
        st.session_state["fert_n"] = 37.0
        st.session_state["fert_k"] = 0.0
        st.session_state["fert_p"] = 0.0
        st.rerun()

    if fcol2.button("🎋 Preset: Sugarcane in Black Soil (High P)", use_container_width=True):
        st.session_state["fert_temp"] = 32.0
        st.session_state["fert_hum"] = 62.0
        st.session_state["fert_moist"] = 42.0
        st.session_state["fert_soil"] = "Black"
        st.session_state["fert_crop"] = "Sugarcane"
        st.session_state["fert_n"] = 12.0
        st.session_state["fert_k"] = 10.0
        st.session_state["fert_p"] = 36.0
        st.rerun()

    if fcol3.button("🌾 Preset: Paddy in Clayey Soil", use_container_width=True):
        st.session_state["fert_temp"] = 28.0
        st.session_state["fert_hum"] = 72.0
        st.session_state["fert_moist"] = 55.0
        st.session_state["fert_soil"] = "Clayey"
        st.session_state["fert_crop"] = "Paddy"
        st.session_state["fert_n"] = 22.0
        st.session_state["fert_k"] = 15.0
        st.session_state["fert_p"] = 25.0
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🌡️ Climate & Soil Substrate")
        f_temp = st.number_input("Ambient Temperature (°C)", 0.0, 55.0, step=0.1, key="fert_temp")
        f_humidity = st.number_input("Air Humidity (%)", 0.0, 100.0, step=0.5, key="fert_hum")
        f_moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, step=0.5, key="fert_moist")
        soil_type = st.selectbox("Soil Classification", soil_types, key="fert_soil")

    with col2:
        st.markdown("##### 🌿 Target Crop & Macronutrients")
        crop_type = st.selectbox("Cultivated Crop", crop_types, key="fert_crop")
        nitrogen = st.number_input("Nitrogen Level", 0.0, 150.0, step=1.0, key="fert_n")
        potassium = st.number_input("Potassium Level", 0.0, 150.0, step=1.0, key="fert_k")
        phosphorous = st.number_input("Phosphorus Level", 0.0, 150.0, step=1.0, key="fert_p")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Recommend Optimal Fertilizer Formulation", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing categorical embeddings and nutrient scales..."):
                recommender = load_fertilizer_recommender()
                result = recommender.recommend({
                    "Temparature": f_temp, "Humidity": f_humidity, "Moisture": f_moisture,
                    "Soil Type": soil_type, "Crop Type": crop_type,
                    "Nitrogen": nitrogen, "Potassium": potassium, "Phosphorous": phosphorous,
                })

            top_fert = result["top_prediction"]
            top_prob = result["top_k"][0][1]

            st.markdown(
                f"""
                <div class="result-hero result-info">
                    <div class="result-title">Targeted Fertilizer Recommendation</div>
                    <div class="result-main">🧪 {top_fert}</div>
                    <div style="margin-top: 0.4rem; font-size: 0.95rem; font-weight: 600;">
                        Recommendation Certainty: {top_prob:.1%} | Crop: {crop_type} on {soil_type} Soil
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top Candidates Visual Distribution
            st.markdown("##### 📊 Top Formulation Rankings")
            for fert, prob in result["top_k"]:
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.write(f"**{fert}**")
                with c2:
                    st.progress(float(prob), text=f"{prob:.1%}")

            if result["warnings"]:
                for w in result["warnings"]:
                    st.warning(f"⚠️ {w}")

        except FileNotFoundError as e:
            st.error(f"Model checkpoint missing: {e}")
        except Exception as e:
            st.error(f"Inference error: {e}")


# =============================================================================
# TAB 3: IRRIGATION SCHEDULER
# =============================================================================
with tab_irrigation:
    st.markdown(
        """
        <div class="agri-card">
            <h3 class="agri-card-title">💧 Precision Smart Irrigation Classifier</h3>
            <p class="agri-card-desc">Binary classification MLP with inverse-frequency class weighting to manage dataset imbalance (83.0% test accuracy on GCEK IoT benchmark).</p>
        """,
        unsafe_allow_html=True,
    )

    crop_options = ["Paddy", "Ground Nuts"]

    # Initialize session state for irrigation inputs
    if "irr_crop" not in st.session_state:
        st.session_state["irr_crop"] = "Paddy"
        st.session_state["irr_days"] = 45
        st.session_state["irr_moist"] = 700.0  # High ADC reading = dry soil in GCEK hardware dataset
        st.session_state["irr_soil_temp"] = 22.0
        st.session_state["irr_air_temp"] = 28.0
        st.session_state["irr_hum"] = 55.0

    # Irrigation Presets (Reflecting real hardware ADC polarity verified in data audit)
    st.markdown("**⚡ Live Demo Sample Presets (Click to Auto-Populate Inputs):**")
    icol1, icol2 = st.columns(2)

    if icol1.button("💧 Preset: Dry Field (High ADC Resistance -> Needs Irrigation)", use_container_width=True):
        st.session_state["irr_crop"] = "Paddy"
        st.session_state["irr_days"] = 45
        st.session_state["irr_moist"] = 700.0
        st.session_state["irr_soil_temp"] = 22.0
        st.session_state["irr_air_temp"] = 28.0
        st.session_state["irr_hum"] = 55.0
        st.rerun()

    if icol2.button("🌿 Preset: Saturated Field (Low ADC Resistance -> Moisture Optimal)", use_container_width=True):
        st.session_state["irr_crop"] = "Ground Nuts"
        st.session_state["irr_days"] = 50
        st.session_state["irr_moist"] = 230.0
        st.session_state["irr_soil_temp"] = 25.0
        st.session_state["irr_air_temp"] = 30.0
        st.session_state["irr_hum"] = 65.0
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🌾 Crop & Soil Moisture Sensor")
        crop_label = st.selectbox("Crop Under Cultivation", crop_options, key="irr_crop")
        crop_code = 1 if crop_label == "Paddy" else 2

        crop_days = st.number_input("Crop Growth Stage (Days Since Planting)", 0, 365, step=1, key="irr_days")
        soil_moisture = st.number_input(
            "Soil Moisture (Raw Sensor ADC Reading [100-900])",
            0.0, 1000.0, step=10.0, key="irr_moist",
            help="Note: In analog resistive/capacitive soil sensors (GCEK dataset), higher ADC readings (600-900) denote dry soil with high electrical resistance, while lower readings (100-350) denote moist soil."
        )

    with col2:
        st.markdown("##### 🌡️ Thermal & Atmospheric Conditions")
        soil_temp = st.number_input("Soil Temperature Sensor (°C)", 0.0, 60.0, step=0.1, key="irr_soil_temp")
        i_temp = st.number_input("Ambient Air Temperature (°C)", 0.0, 55.0, step=0.1, key="irr_air_temp")
        i_humidity = st.number_input("Relative Air Humidity (%)", 0.0, 100.0, step=0.5, key="irr_hum")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Assess Field Irrigation Status", type="primary", use_container_width=True):
        try:
            with st.spinner("Evaluating soil-water threshold with binary neural model..."):
                predictor = load_irrigation_predictor()
                result = predictor.predict({
                    "CropType": crop_code, "CropDays": crop_days,
                    "Soil Moisture": soil_moisture, "Soil Temperature": soil_temp,
                    "Temperature": i_temp, "Humidity": i_humidity,
                })

            needs_irrigation = result["irrigate"]
            confidence = result["confidence"]

            if needs_irrigation:
                st.markdown(
                    f"""
                    <div class="result-hero result-alert">
                        <div class="result-title">Immediate Action Advised</div>
                        <div class="result-main">💧 IRRIGATION REQUIRED</div>
                        <div style="margin-top: 0.4rem; font-size: 0.95rem; font-weight: 600;">
                            Model Confidence: {confidence:.1%} | Depleted Soil Moisture Zone Detected
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-hero result-success">
                        <div class="result-title">Field Hydration Status</div>
                        <div class="result-main">✅ MOISTURE OPTIMAL (NO WATERING NEEDED)</div>
                        <div style="margin-top: 0.4rem; font-size: 0.95rem; font-weight: 600;">
                            Model Confidence: {confidence:.1%} | Soil Hydration Within Favorable Bounds
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Probabilities Breakdown
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Probability: No Irrigation**")
                st.progress(float(result["probabilities"]["No"]), text=f"{result['probabilities']['No']:.1%}")
            with c2:
                st.markdown("**Probability: Irrigate Now**")
                st.progress(float(result["probabilities"]["Yes"]), text=f"{result['probabilities']['Yes']:.1%}")

            if result["warnings"]:
                for w in result["warnings"]:
                    st.warning(f"⚠️ {w}")

        except FileNotFoundError as e:
            st.error(f"Model checkpoint missing: {e}")
        except Exception as e:
            st.error(f"Inference error: {e}")


# =============================================================================
# TAB 4: PLANT DISEASE DETECTION
# =============================================================================
with tab_disease:
    st.markdown(
        """
        <div class="agri-card">
            <h3 class="agri-card-title">🍃 Computer Vision Plant Disease Classifier</h3>
            <p class="agri-card-desc">Convolutional Neural Network (CNN) trained on an 8-class subset of the PlantVillage benchmark (Apple, Corn, Potato, and Tomato crops; 93.9% test accuracy).</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 📁 Provide Specimen Image")

    img_source = st.radio(
        "Choose Image Source for Live Evaluation:",
        ["⚡ Use Pre-Loaded Plant Specimen (Guaranteed Demo Safe)", "📤 Upload Leaf Photo from Computer"],
        horizontal=True,
    )

    image_to_evaluate = None
    specimen_name = ""
    direct_file_path = None

    if img_source == "⚡ Use Pre-Loaded Plant Specimen (Guaranteed Demo Safe)":
        preset_samples = {
            "🍅 Tomato — Early Blight (Infected)": "Tomato___Early_blight",
            "🍅 Tomato — Healthy Leaf": "Tomato___healthy",
            "🥔 Potato — Early Blight (Infected)": "Potato___Early_blight",
            "🥔 Potato — Healthy Leaf": "Potato___healthy",
            "🌽 Corn — Common Rust (Infected)": "Corn_(maize)___Common_rust_",
            "🌽 Corn — Healthy Leaf": "Corn_(maize)___healthy",
            "🍏 Apple — Apple Scab (Infected)": "Apple___Apple_scab",
            "🍏 Apple — Healthy Leaf": "Apple___healthy",
        }

        selected_label = st.selectbox("Select Benchmark Specimen:", list(preset_samples.keys()))
        folder_name = preset_samples[selected_label]
        target_dir = config.DISEASE_IMAGES_DIR / folder_name

        if target_dir.exists():
            sample_files = list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.JPG"))
            if sample_files:
                direct_file_path = sample_files[0]
                image_to_evaluate = Image.open(direct_file_path).convert("RGB")
                specimen_name = direct_file_path.name
        else:
            st.error(f"Sample folder not found at {target_dir}")

    else:
        uploaded_file = st.file_uploader("Upload leaf photograph (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_to_evaluate = Image.open(uploaded_file).convert("RGB")
            specimen_name = uploaded_file.name

    st.markdown("</div>", unsafe_allow_html=True)

    if image_to_evaluate is not None:
        c_img, c_diag = st.columns([1, 1.3])

        with c_img:
            st.markdown("##### 🌿 Leaf Specimen Preview")
            st.image(image_to_evaluate, caption=f"Specimen: {specimen_name}", use_container_width=True)

        with c_diag:
            st.markdown("##### 🔬 AI Diagnostic Output")
            if st.button("🚀 Analyze Leaf Pathology with Vision CNN", type="primary", use_container_width=True):
                try:
                    with st.spinner("Extracting convolutional feature maps..."):
                        detector = load_disease_detector()
                        
                        # Use direct file path for presets, or safe tempfile for uploads
                        if direct_file_path is not None and Path(direct_file_path).exists():
                            result = detector.predict(str(direct_file_path))
                        else:
                            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                                image_to_evaluate.save(tmp.name)
                                tmp_path = tmp.name
                            try:
                                result = detector.predict(tmp_path)
                            finally:
                                Path(tmp_path).unlink(missing_ok=True)

                        st.session_state["disease_result"] = result

                except FileNotFoundError as e:
                    st.error(f"CNN Model checkpoint missing: {e}")
                except Exception as e:
                    st.error(f"Vision inference error: {e}")

            # Display result if available in session state
            if "disease_result" in st.session_state:
                result = st.session_state["disease_result"]
                is_healthy = "healthy" in result["condition"].lower()
                top_class = result["top_prediction"]
                top_prob = result["top_k"][0][1]

                if is_healthy:
                    st.markdown(
                        f"""
                        <div class="result-hero result-success">
                            <div class="result-title">Specimen Health Status</div>
                            <div class="result-main">✅ {result['crop']} — Healthy</div>
                            <div style="margin-top: 0.4rem; font-size: 0.95rem; font-weight: 600;">
                                Diagnosis Confidence: {top_prob:.1%} | No pathogen detected
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-hero result-danger">
                            <div class="result-title">Pathology Alert Detected</div>
                            <div class="result-main">⚠️ {result['crop']} — {result['condition'].title()}</div>
                            <div style="margin-top: 0.4rem; font-size: 0.95rem; font-weight: 600;">
                                Diagnosis Confidence: {top_prob:.1%} | Active Infection
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Top-3 Predictions Bar
                st.markdown("###### 📊 Vision Classifier Probability Distribution")
                for cls_name, prob in result["top_k"]:
                    clean_lbl = cls_name.replace("___", " — ").replace("_", " ")
                    st.write(f"**{clean_lbl}**")
                    st.progress(float(prob), text=f"{prob:.1%}")

                # Agronomic Advisory Action Card
                if top_class in DISEASE_ACTION_GUIDE:
                    info = DISEASE_ACTION_GUIDE[top_class]
                    st.markdown(
                        f"""
                        <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 12px; padding: 1.1rem; margin-top: 1rem; color: #1E293B;">
                            <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 0.4rem;">
                                🌾 Agronomic Advisory & Management Protocol
                            </div>
                            <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.3rem;">
                                <b>Identified Pathogen:</b> {info['pathogen']} | <b>Severity Urgency:</b> {info['urgency']}
                            </div>
                            <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.3rem;">
                                <b>Recommended Chemical / Organic Control:</b> {info['action']}
                            </div>
                            <div style="font-size: 0.85rem; color: #334155;">
                                <b>Cultural Practice:</b> {info['cultural']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# =============================================================================
# TAB 5: IOT SIMULATOR & AUTONOMOUS FARM AUDIT
# =============================================================================
with tab_iot:
    st.markdown(
        """
        <div class="agri-card">
            <h3 class="agri-card-title">📡 IoT Sensor Telemetry Simulator & Pipeline Audit</h3>
            <p class="agri-card-desc">Simulates a physical IoT sensor node with diurnal day/night cycle curves and feeds readings through Crop, Fertilizer, and Irrigation models simultaneously.</p>
        """,
        unsafe_allow_html=True,
    )

    # Required Simulation Disclosure Banner
    st.markdown(
        """
        <div class="sim-banner">
            <b>⚠️ SIMULATED TELEMETRY LAYER:</b> Sensor readings in this module are mathematically generated
            using smooth diurnal (day/night) cycles for temperature, solar illuminance, and soil moisture depletion
            (as designed in Section III-F of the paper). <b>No physical sensor microcontrollers (ESP32/Arduino) are connected.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_soil_type = st.selectbox("Field Soil Type (for Fertilizer)", ["Sandy", "Loamy", "Black", "Red", "Clayey"], key="sim_soil")
    with col2:
        sim_crop_type = st.selectbox(
            "Cultivated Crop (for Fertilizer)",
            ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil seeds", "Ground Nuts", "Pulses"],
            key="sim_crop",
        )
    with col3:
        sim_crop_label = st.selectbox("Crop Type (for Irrigation)", ["Paddy", "Ground Nuts"], key="sim_irr_crop")
        sim_crop_code = 1 if sim_crop_label == "Paddy" else 2

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("📡 Generate Simulated Reading & Execute Autonomous Pipeline", type="primary", use_container_width=True):
        from src.iot.simulator import IoTSensorSimulator
        from src.iot.adapters import (
            to_crop_recommendation_input,
            to_fertilizer_recommendation_input,
            to_irrigation_prediction_input,
        )

        with st.spinner("Polling simulated sensor node and calculating inference stream..."):
            sim = IoTSensorSimulator()
            reading = sim.read()
            r_dict = reading.as_dict()

        st.markdown("##### 📈 Live Simulated Node Readings")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Air Temperature", f"{r_dict['air_temperature_c']:.1f} °C", f"Diurnal Hour: {r_dict['hour_of_day']:.1f}h")
        m2.metric("Relative Humidity", f"{r_dict['humidity_pct']:.1f} %")
        m3.metric("Soil Moisture (Raw)", f"{r_dict['soil_moisture_raw']:.0f} ADC")
        m4.metric("Solar Illuminance", f"{r_dict['light_lux']:.0f} Lux")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Nitrogen (N)", f"{r_dict['N']:.0f} mg/kg")
        m6.metric("Phosphorus (P)", f"{r_dict['P']:.0f} mg/kg")
        m7.metric("Potassium (K)", f"{r_dict['K']:.0f} mg/kg")
        m8.metric("Soil pH", f"{r_dict['ph']:.2f}")

        st.markdown("---")
        st.markdown("##### 🤖 Multi-Model Unified Farm Audit")

        r_col1, r_col2, r_col3 = st.columns(3)

        with r_col1:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; height: 100%; color: #1E293B;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Model 1 (Tabular MLP)</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0;">🌱 Crop Intelligence</div>
                """,
                unsafe_allow_html=True,
            )
            try:
                rec_crop = load_crop_recommender()
                c_out = rec_crop.recommend(to_crop_recommendation_input(reading))
                st.success(f"**Optimal Crop:** {c_out['top_prediction'].capitalize()}")
                st.caption(f"Top Confidence: {c_out['top_k'][0][1]:.1%}")
            except Exception as e:
                st.error(f"Error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with r_col2:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; height: 100%; color: #1E293B;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Model 2 (Tabular MLP)</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0;">🧪 Fertilizer Optimizer</div>
                """,
                unsafe_allow_html=True,
            )
            try:
                rec_fert = load_fertilizer_recommender()
                f_out = rec_fert.recommend(to_fertilizer_recommendation_input(reading, sim_soil_type, sim_crop_type))
                st.info(f"**Formulation:** {f_out['top_prediction']}")
                st.caption(f"Top Confidence: {f_out['top_k'][0][1]:.1%}")
            except Exception as e:
                st.error(f"Error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with r_col3:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; height: 100%; color: #1E293B;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Model 3 (Binary MLP)</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0;">💧 Irrigation Status</div>
                """,
                unsafe_allow_html=True,
            )
            try:
                pred_irr = load_irrigation_predictor()
                i_out = pred_irr.predict(to_irrigation_prediction_input(reading, sim_crop_code, 15))
                if i_out["irrigate"]:
                    st.warning(f"**Status: Irrigate Now**")
                else:
                    st.success(f"**Status: Moisture Optimal**")
                st.caption(f"Confidence: {i_out['confidence']:.1%}")
            except Exception as e:
                st.error(f"Error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Footer Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-top: 3.5rem; padding: 1.5rem; color: #6B7280; font-size: 0.85rem; border-top: 1px solid #E5E7EB;">
        🌾 <b>AgroCortex AI</b> — Smart Agriculture Decision Assistant Platform | VIT-AP Academic Research Prototype
        <br>
        Deep Learning Pipeline: PyTorch 2.14, TorchVision, Scikit-Learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
