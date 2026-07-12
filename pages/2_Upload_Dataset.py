import streamlit as st
import pandas as pd
from ui.theme import load_theme

load_theme()

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
st.title("📂 Upload Dataset")

st.markdown("""
Upload your dataset in **CSV** or **Excel** format to begin the
Machine Learning workflow.
""")

st.divider()

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose a Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:

        # Read File
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Save in Session State
        st.session_state["dataset"] = df

        st.success("✅ Dataset uploaded successfully!")

        st.divider()

        # Metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.divider()

        # Preview
        st.subheader("👀 Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        st.divider()

        # Dataset Info
        st.subheader("📑 Dataset Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Column Names")
            st.write(list(df.columns))

        with col2:
            st.write("### Data Types")
            st.write(df.dtypes.astype(str))

        st.divider()

        # Target Column
        target = st.selectbox(
            "🎯 Select Target Column",
            df.columns
        )

        st.session_state["target_column"] = target

        st.success(f"✅ Target Column Selected: **{target}**")

        st.info("➡ Now open **Dataset Analysis** from the sidebar.")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("📁 Please upload a dataset to continue.")