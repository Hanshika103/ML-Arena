import streamlit as st
from ui.theme import load_theme
from config.settings import PAGE_CONFIG

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(**PAGE_CONFIG)

# ----------------------------
# Load Custom Theme
# ----------------------------
load_theme()

# ----------------------------
# Session State Initialization
# ----------------------------
if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

if "problem_type" not in st.session_state:
    st.session_state.problem_type = None

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "best_model" not in st.session_state:
    st.session_state.best_model = None

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
)