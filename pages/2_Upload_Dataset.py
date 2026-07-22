import streamlit as st
import pandas as pd

from ui.theme import load_theme
from utils.cache_utils import load_csv
from utils.error_handler import handle_error
from utils.logger import get_logger

# ==========================================================
# Load Theme & Logger
# ==========================================================

load_theme()
logger = get_logger()

# ==========================================================
# Page Title
# ==========================================================

st.title("📂 Upload Dataset")

st.markdown(
    """
Upload your dataset in **CSV** or **Excel** format to begin the
Machine Learning workflow.
"""
)

st.divider()

# ==========================================================
# File Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "Choose a Dataset",
    type=["csv", "xlsx"],
)

if uploaded_file is not None:

    try:

        # --------------------------------------------------
        # Read Dataset
        # --------------------------------------------------

        if uploaded_file.name.endswith(".csv"):
            df = load_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if df.empty:
            st.error("❌ The uploaded dataset is empty.")
            st.stop()

        if df.columns.duplicated().any():
            st.error("❌ Dataset contains duplicate column names.")
            st.stop()

        if uploaded_file.size > 50 * 1024 * 1024:
            st.warning(
                "⚠ Large dataset detected. Loading may take a few moments."
            )

        # --------------------------------------------------
        # Save Dataset
        # --------------------------------------------------

        st.session_state["dataset"] = df

        logger.info(
            f"Dataset uploaded: {uploaded_file.name} | "
            f"Rows={df.shape[0]} | Columns={df.shape[1]}"
        )

        st.success(
            f"✅ Dataset '{uploaded_file.name}' uploaded successfully!"
        )

        st.divider()

        # ==================================================
        # Dataset Summary
        # ==================================================

        st.subheader("📊 Dataset Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        st.divider()

        # ==================================================
        # Preview
        # ==================================================

        st.subheader("👀 Dataset Preview")

        preview_rows = min(len(df), 100)

        st.dataframe(
            df.head(preview_rows),
            width="stretch",
            height=400,
            hide_index=True,
        )

        st.divider()

        # ==================================================
        # Dataset Information
        # ==================================================

        st.subheader("📑 Dataset Information")

        col1, col2 = st.columns(2)

        with col1:

            st.write("### Column Names")

            st.dataframe(
                pd.DataFrame(
                    {"Column": df.columns}
                ),
                width="stretch",
                hide_index=True,
            )

        with col2:

            st.write("### Data Types")

            info_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Data Type": df.dtypes.astype(str),
                }
            )

            st.dataframe(
                info_df,
                width="stretch",
                hide_index=True,
            )

        st.divider()

        # ==================================================
        # Target Column
        # ==================================================

        st.subheader("🎯 Target Selection")

        target = st.selectbox(
            "Select Target Column",
            options=df.columns,
        )

        st.session_state["target_column"] = target

        st.success(
            f"✅ Target Column Selected: **{target}**"
        )

        st.info(
            "➡ Continue to **Dataset Analysis** from the sidebar."
        )

    except Exception as e:

        logger.exception(e)

        handle_error(
            e,
            "Unable to read the uploaded dataset.",
        )

        st.stop()

else:

    st.info("📁 Please upload a dataset to continue.")