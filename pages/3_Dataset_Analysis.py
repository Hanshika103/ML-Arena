import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ui.theme import load_theme
from utils.cache_utils import get_dataset_summary
from utils.logger import get_logger


load_theme()
from utils.validators import validate_dataset

if not validate_dataset():
    st.stop()

# ----------------------------
# Session State Initialization
# ----------------------------
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

if "target_column" not in st.session_state:
    st.session_state["target_column"] = None

# ----------------------------
# Page Title
# ----------------------------
st.title("📊 Dataset Analysis")



df = st.session_state["dataset"]
logger = get_logger()
logger.info("Dataset Analysis page opened.")
summary = get_dataset_summary(df)

st.success("✅ Dataset Loaded Successfully")

st.divider()

# ----------------------------
# Summary Metrics
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", summary["rows"])
col2.metric("Columns", summary["columns"])
col3.metric("Missing Values", summary["missing"])
col4.metric("Duplicate Rows", summary["duplicates"])

st.divider()

# ----------------------------
# Data Types
# ----------------------------
st.subheader("📑 Data Types")

datatype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})



st.divider()

# ----------------------------
# Missing Values
# ----------------------------
st.subheader("❌ Missing Values")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})

st.dataframe(missing_df, width = "stretch")

st.divider()

# ----------------------------
# Statistical Summary
# ----------------------------
st.subheader("📈 Statistical Summary")

st.dataframe(
    df.describe(include="all"),
    
)

st.divider()

# ----------------------------
# Correlation Heatmap
# ----------------------------
numeric_df = df.select_dtypes(include=["number"])

st.subheader("🔥 Correlation Heatmap")

if len(numeric_df.columns) > 1:

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8,6))

    heatmap = ax.imshow(corr)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    plt.colorbar(heatmap)

    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Not enough numerical columns.")

st.divider()

# ----------------------------
# Distribution Plot
# ----------------------------
st.subheader("📉 Distribution Plot")

if len(numeric_df.columns):

    selected = st.selectbox(
        "Select Numerical Column",
        numeric_df.columns
    )

    fig, ax = plt.subplots(figsize=(7,4))

    ax.hist(df[selected], bins=20)

    ax.set_title(selected)

    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ----------------------------
# Target Distribution
# ----------------------------
target = st.session_state["target_column"]

if target is not None:

    st.subheader("🎯 Target Distribution")

    counts = df[target].value_counts()

    fig, ax = plt.subplots(figsize=(7,4))

    ax.bar(counts.index.astype(str), counts.values)

    ax.set_xlabel(target)
    ax.set_ylabel("Count")

    st.pyplot(fig)
    plt.close(fig)