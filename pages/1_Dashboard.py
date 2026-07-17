import streamlit as st
from ui.theme import load_theme

# ----------------------------------------
# Page Configuration
# ----------------------------------------

load_theme()

st.set_page_config(
    page_title="ML Arena",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------
# Hero Section
# ----------------------------------------

st.markdown("""
<div class="hero">

<h1>🧠 ML Arena</h1>

<h3>Intelligent Machine Learning Benchmark Platform</h3>

<p>
Build, train, compare, evaluate and export Machine Learning models
through a professional interactive interface.
</p>

</div>
""", unsafe_allow_html=True)

# ----------------------------------------
# Quick Statistics
# ----------------------------------------

st.subheader("📊 Project Overview")

col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric("ML Models","8")

with col2:
    st.metric("Problem Types","2")

with col3:
    st.metric("Project Modules","7")

with col4:
    st.metric("Export Formats","4")

st.divider()

# ----------------------------------------
# Features
# ----------------------------------------

st.subheader("⚡ Platform Features")

c1,c2,c3=st.columns(3)

with c1:

    st.info("""
### 📂 Upload Dataset

Upload CSV datasets securely with validation.
""")

    st.info("""
### 📊 Dataset Analysis

Interactive EDA with statistics and visualizations.
""")

with c2:

    st.info("""
### ⚙️ Data Preprocessing

Missing values, encoding and feature scaling.
""")

    st.info("""
### 🧠 Model Training

Train multiple ML algorithms with one click.
""")

with c3:

    st.info("""
### 📈 Performance Evaluation

Compare models using detailed metrics.
""")

    st.info("""
### ⬇️ Export Results

Download predictions, reports and trained models.
""")

st.divider()

# ----------------------------------------
# Workflow
# ----------------------------------------

st.subheader("🔄 Machine Learning Workflow")

st.markdown("""
✅ Upload Dataset

⬇️

✅ Analyze Dataset

⬇️

✅ Preprocess Data

⬇️

✅ Train Models

⬇️

✅ Compare Performance

⬇️

✅ Export Results
""")

st.divider()

# ----------------------------------------
# Supported Models
# ----------------------------------------

st.subheader("🤖 Supported Machine Learning Models")

left,right=st.columns(2)

with left:

    st.success("""
### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Support Vector Machine
""")

with right:

    st.success("""
### Regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
""")

st.divider()

# ----------------------------------------
# Why ML Arena
# ----------------------------------------

st.subheader("⭐ Why ML Arena?")

st.markdown("""
- 🚀 Beginner Friendly

- 📊 Interactive Visualizations

- ⚡ Fast Model Training

- 🎯 Automatic Performance Evaluation

- 📦 Export Predictions & Models

- 💻 Professional Streamlit Interface

- 📈 Multiple Machine Learning Algorithms

- 🛠 Easy-to-use Workflow
""")

st.divider()

# ----------------------------------------
# Technologies
# ----------------------------------------

st.subheader("🛠 Technology Stack")

tech1,tech2,tech3,tech4=st.columns(4)

tech1.success("Python")
tech2.success("Streamlit")
tech3.success("Scikit-Learn")
tech4.success("Plotly")

st.divider()

# ----------------------------------------
# Footer
# ----------------------------------------

st.caption(
    "ML Arena • Intelligent Machine Learning Benchmark Platform • Version 1.0"
)