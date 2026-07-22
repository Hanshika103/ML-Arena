'''import streamlit as st
from ui.theme import load_theme
from config.settings import PAGE_CONFIG
from utils.logger import get_logger
from utils.session_manager import initialize_session_state

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(**PAGE_CONFIG)

# ----------------------------
# Load Custom Theme
# ----------------------------
load_theme()

# ----------------------------
# Initialize Session State
# ----------------------------
initialize_session_state()

# ----------------------------
# Initialize Logger
# ----------------------------
logger = get_logger()
logger.info("ML Arena application started.")
st.success("Logger executed from app.py")

# ----------------------------
# Session State Initialization
# ----------------------------


# ----------------------------
# Hero Section
# ----------------------------
st.markdown(
    """
<div class="hero">

<h1>🚀 ML Arena</h1>

<h3>Intelligent Machine Learning Benchmark Platform</h3>

<p>
Upload datasets, benchmark multiple ML models,
visualize performance and discover the best algorithm
—all from one elegant dashboard.
</p>

</div>
""",
    unsafe_allow_html=True,
)

st.info(
    "👈 Use the sidebar to navigate through the application."
)'''


import streamlit as st

from config.settings import PAGE_CONFIG
from ui.theme import load_theme
from utils.logger import get_logger
from utils.session_manager import initialize_session_state

# ==========================================================
# Streamlit Page Configuration
# ==========================================================
st.set_page_config(**PAGE_CONFIG)

# ==========================================================
# Initialize Logger
# ==========================================================
logger = get_logger()
logger.info("ML Arena application started.")

# ==========================================================
# Load Theme
# ==========================================================
try:
    load_theme()
except Exception as e:
    logger.exception(f"Failed to load theme: {e}")

# ==========================================================
# Initialize Session State
# ==========================================================
initialize_session_state()

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("🚀 ML Arena")
st.sidebar.caption("Intelligent Machine Learning Benchmark Platform")

# ==========================================================
# Hero Section
# ==========================================================
st.markdown(
    """
<div class="hero">

<h1>🚀 ML Arena</h1>

<h3>Intelligent Machine Learning Benchmark Platform</h3>

<p>
Upload datasets, preprocess data, train multiple machine learning models,
compare their performance, visualize results, and export everything—
all from one elegant dashboard.
</p>

</div>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# Quick Overview
# ==========================================================
st.markdown("## 📊 Platform Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Supported Models",
        value="8+"
    )

with col2:
    st.metric(
        label="ML Tasks",
        value="Classification & Regression"
    )

with col3:
    st.metric(
        label="Export Options",
        value="CSV / Model"
    )

# ==========================================================
# Welcome Section
# ==========================================================
st.markdown("## 👋 Welcome")

st.write(
    """
ML Arena is an intelligent benchmarking platform that enables users to
train, compare, evaluate, and export Machine Learning models without
writing repetitive code.

Use the sidebar to move through the complete ML workflow—from uploading
your dataset to exporting the best-performing model.
"""
)

# ==========================================================
# Feature Cards
# ==========================================================
st.markdown("## ✨ Key Features")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
<div class="feature-card">

<h3>📂 Dataset Upload</h3>

<p>
Upload CSV datasets quickly with automatic validation.
</p>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="feature-card">

<h3>⚙️ Data Preprocessing</h3>

<p>
Handle missing values, encode categorical features,
and scale your dataset effortlessly.
</p>

</div>
""",
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        """
<div class="feature-card">

<h3>🧠 Model Benchmarking</h3>

<p>
Train and compare multiple Machine Learning models
using a unified interface.
</p>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="feature-card">

<h3>📈 Results & Export</h3>

<p>
Visualize metrics, compare performance,
and export trained models and reports.
</p>

</div>
""",
        unsafe_allow_html=True,
    )

# ==========================================================
# Information
# ==========================================================
st.info(
    "👈 Use the sidebar to navigate through each stage of the Machine Learning pipeline."
)

# ==========================================================
# Footer
# ==========================================================
st.divider()

st.caption(
    "🚀 ML Arena | Intelligent Machine Learning Benchmark Platform | Built with Streamlit"
)