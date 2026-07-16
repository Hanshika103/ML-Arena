import streamlit as st
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
)