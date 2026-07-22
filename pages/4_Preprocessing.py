import streamlit as st
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
)

from ui.theme import load_theme
from utils.validators import validate_dataset

# ==========================================================
# Load Theme
# ==========================================================

load_theme()

# ==========================================================
# Validate Dataset
# ==========================================================

if not validate_dataset():
    st.stop()

# ==========================================================
# Page Title
# ==========================================================

st.title("🧹 Data Preprocessing")

df = st.session_state["dataset"].copy()
target = st.session_state["target_column"]

st.info(
    "Configure preprocessing options and prepare your dataset for model training."
)

st.divider()

# ==========================================================
# Missing Value Statistics
# ==========================================================

st.subheader("❌ Missing Value Handling")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:

    stats = pd.DataFrame(
        {
            "Column": missing_values.index,
            "Missing Values": missing_values.values,
        }
    )

    st.dataframe(
        stats,
        width="stretch",
        hide_index=True,
    )

else:

    st.success("✅ No Missing Values Found")

# ==========================================================
# Missing Value Method
# ==========================================================

missing_option = st.radio(
    "Choose Missing Value Strategy",
    [
        "Do Nothing",
        "Mean",
        "Median",
        "Mode",
        "Drop Rows",
    ],
)

st.divider()

# ==========================================================
# Encoding
# ==========================================================

st.subheader("🔤 Encoding")

encoding = st.checkbox(
    "Apply Label Encoding to Categorical Features"
)

st.divider()

# ==========================================================
# Feature Scaling
# ==========================================================

st.subheader("📏 Feature Scaling")

scaling = st.selectbox(
    "Scaling Method",
    [
        "None",
        "StandardScaler",
        "MinMaxScaler",
    ],
)

st.divider()

# ==========================================================
# Duplicate Rows
# ==========================================================

remove_duplicates = st.checkbox(
    "Remove Duplicate Rows"
)

st.divider()
# ==========================================================
# Apply Preprocessing
# ==========================================================

if st.button("🚀 Apply Preprocessing", type="primary"):

    processed = df.copy()

    # ------------------------------------------------------
    # Missing Value Handling
    # ------------------------------------------------------

    if missing_option == "Mean":

        numeric_cols = processed.select_dtypes(
            include="number"
        ).columns

        for col in numeric_cols:
            processed[col] = processed[col].fillna(
                processed[col].mean()
            )

    elif missing_option == "Median":

        numeric_cols = processed.select_dtypes(
            include="number"
        ).columns

        for col in numeric_cols:
            processed[col] = processed[col].fillna(
                processed[col].median()
            )

    elif missing_option == "Mode":

        for col in processed.columns:

            if not processed[col].mode().empty:

                processed[col] = processed[col].fillna(
                    processed[col].mode()[0]
                )

    elif missing_option == "Drop Rows":

        processed = processed.dropna()

    # ------------------------------------------------------
    # Feature Columns
    # (Never preprocess target column)
    # ------------------------------------------------------

    feature_columns = [
        col for col in processed.columns
        if col != target
    ]

    # ------------------------------------------------------
    # Label Encoding
    # ------------------------------------------------------

    if encoding:

        encoder = LabelEncoder()

        categorical_columns = processed[
            feature_columns
        ].select_dtypes(
            include=["object", "category"]
        ).columns

        for col in categorical_columns:

            processed[col] = encoder.fit_transform(
                processed[col].astype(str)
            )

    # ------------------------------------------------------
    # Feature Scaling
    # ------------------------------------------------------

    numeric_columns = processed[
        feature_columns
    ].select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:

        if scaling == "StandardScaler":

            scaler = StandardScaler()

            processed[numeric_columns] = scaler.fit_transform(
                processed[numeric_columns]
            )

        elif scaling == "MinMaxScaler":

            scaler = MinMaxScaler()

            processed[numeric_columns] = scaler.fit_transform(
                processed[numeric_columns]
            )

    # ------------------------------------------------------
    # Remove Duplicate Rows
    # ------------------------------------------------------

    if remove_duplicates:

        processed = processed.drop_duplicates()

    # ------------------------------------------------------
    # Save Processed Dataset
    # ------------------------------------------------------

    st.session_state["processed_data"] = processed

    st.success("✅ Preprocessing Completed Successfully!")

    st.divider()

    # ======================================================
    # Preview
    # ======================================================

    st.subheader("📊 Processed Dataset Preview")

    preview_rows = min(len(processed), 100)

    st.dataframe(
        processed.head(preview_rows),
        width="stretch",
        height=400,
        hide_index=True,
    )

    st.divider()

    # ======================================================
    # Summary
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", processed.shape[0])

    with col2:
        st.metric("Columns", processed.shape[1])

    with col3:
        st.metric(
            "Missing Values",
            int(processed.isnull().sum().sum())
        )

    st.divider()

    # ======================================================
    # Download
    # ======================================================

    csv = processed.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download Processed Dataset",
        data=csv,
        file_name="processed_dataset.csv",
        mime="text/csv",
        width="stretch",
    )

    st.info(
        "➡ Next open **Model Training** from the sidebar."
    )