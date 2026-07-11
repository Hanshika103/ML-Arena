import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

# ----------------------------
# Check Dataset
# ----------------------------
if st.session_state["dataset"] is None:
    st.warning("⚠️ Please upload a dataset first from the Upload Dataset page.")
    st.stop()

df = st.session_state["dataset"]

st.success("✅ Dataset Loaded Successfully")

st.divider()

# ----------------------------
# Summary Metrics
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))
col4.metric("Duplicate Rows", int(df.duplicated().sum()))

st.divider()

# ----------------------------
# Data Types
# ----------------------------
st.subheader("📑 Data Types")

datatype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(datatype_df, use_container_width=True)

st.divider()

# ----------------------------
# Missing Values
# ----------------------------
st.subheader("❌ Missing Values")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})

st.dataframe(missing_df, use_container_width=True)

st.divider()

# ----------------------------
# Statistical Summary
# ----------------------------
st.subheader("📈 Statistical Summary")

st.dataframe(
    df.describe(include="all"),
    use_container_width=True
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