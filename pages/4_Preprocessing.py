import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from ui.theme import load_theme

load_theme()

from utils.validators import validate_dataset

if not validate_dataset():
    st.stop()

# ----------------------------
# Session State Initialization
# ----------------------------
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None

if "target_column" not in st.session_state:
    st.session_state["target_column"] = None

# ----------------------------
# Check Dataset
# ----------------------------
st.title("🧹 Data Preprocessing")

if st.session_state["dataset"] is None:
    st.warning("⚠ Please upload a dataset first.")
    st.stop()

df = st.session_state["dataset"].copy()

st.success("✅ Dataset Loaded Successfully")

st.divider()

# ============================
# Missing Values
# ============================

st.subheader("❌ Missing Value Handling")
# Show Missing Value Statistics

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if len(missing_values) > 0:

    st.info("Columns containing missing values")

    stats = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(
    stats,
    width="stretch",
    hide_index=True
)

else:
    st.success("✅ No Missing Values Found")

missing_option = st.radio(
    "Choose Method",
    [
        "Do Nothing",
        "Mean",
        "Median",
        "Mode",
        "Drop Rows"
    ]
)

# ============================
# Encoding
# ============================

st.divider()

st.subheader("🔤 Encoding")

encoding = st.checkbox("Apply Label Encoding")

# ============================
# Scaling
# ============================

st.divider()

st.subheader("📏 Feature Scaling")

scaling = st.selectbox(
    "Scaling Method",
    [
        "None",
        "StandardScaler",
        "MinMaxScaler"
    ]
)

# ============================
# Remove Duplicates
# ============================

st.divider()

remove_duplicates = st.checkbox("Remove Duplicate Rows")

# ============================
# Apply Button
# ============================

st.divider()

if st.button("🚀 Apply Preprocessing"):

    processed = df.copy()

    # -------------------------
    # Missing Values
    # -------------------------

    if missing_option == "Mean":

        numeric = processed.select_dtypes(include="number").columns

        for col in numeric:
            processed[col] = processed[col].fillna(
                processed[col].mean()
            )

    elif missing_option == "Median":

        numeric = processed.select_dtypes(include="number").columns

        for col in numeric:
            processed[col] = processed[col].fillna(
                processed[col].median()
            )

    elif missing_option == "Mode":

        for col in processed.columns:
            processed[col] = processed[col].fillna(
                processed[col].mode()[0]
            )

    elif missing_option == "Drop Rows":

        processed.dropna(inplace=True)

    # -------------------------
    # Encoding
    # -------------------------

    if encoding:

        encoder = LabelEncoder()

        categorical = processed.select_dtypes(
            include="object"
        ).columns

        for col in categorical:

            processed[col] = encoder.fit_transform(
                processed[col].astype(str)
            )

    # -------------------------
    # Scaling
    # -------------------------

    numeric = processed.select_dtypes(include="number").columns

    if scaling == "StandardScaler":

        scaler = StandardScaler()

        processed[numeric] = scaler.fit_transform(
            processed[numeric]
        )

    elif scaling == "MinMaxScaler":

        scaler = MinMaxScaler()

        processed[numeric] = scaler.fit_transform(
            processed[numeric]
        )

    # -------------------------
    # Remove Duplicates
    # -------------------------

    if remove_duplicates:

        processed.drop_duplicates(inplace=True)

    # -------------------------
    # Save
    # -------------------------

    st.session_state["processed_data"] = processed

    st.success("✅ Preprocessing Completed!")

    st.divider()

    st.subheader("📊 Processed Dataset Preview")

    st.dataframe(
    processed,
    width="stretch",
    height=400
)
# -------------------------
# Download Processed Dataset
# -------------------------

csv = processed.to_csv(index=False).encode("utf-8")


st.info(
        "➡ Next open **Model Training** from the sidebar."
    )
if st.session_state["processed_data"] is not None:

    csv = st.session_state["processed_data"].to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Processed Dataset",
        data=csv,
        file_name="processed_dataset.csv",
        mime="text/csv",
        
    )