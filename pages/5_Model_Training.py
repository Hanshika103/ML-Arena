import streamlit as st
import pandas as pd

from ui.theme import load_theme

from utils.model_training import (
    split_data,
    get_model,
    train_selected_model,
    predict,
)

from utils.validators import validate_dataset
from utils.logger import get_logger

# ==========================================================
# Load Theme
# ==========================================================

load_theme()

logger = get_logger()

# ==========================================================
# Validate Dataset
# ==========================================================

if not validate_dataset():
    st.stop()

# ==========================================================
# Check Processed Dataset
# ==========================================================

if st.session_state["processed_data"] is None:

    st.warning("⚠ Please complete Data Preprocessing first.")

    st.stop()

# ==========================================================
# Load Dataset
# ==========================================================

df = st.session_state["processed_data"].copy()

# ==========================================================
# Page Title
# ==========================================================

st.title("🧠 Model Training")

st.success("✅ Processed Dataset Loaded Successfully")

logger.info("Model Training page opened.")

st.divider()

# ==========================================================
# Dataset Information
# ==========================================================

rows, cols = df.shape

col1, col2, col3 = st.columns(3)

with col1:

    st.metric("Rows", rows)

with col2:

    st.metric("Columns", cols)

with col3:

    memory = df.memory_usage(
        deep=True
    ).sum() / 1024

    st.metric(
        "Memory Usage",
        f"{memory:.1f} KB",
    )

st.divider()

# ==========================================================
# Target Column
# ==========================================================

default_target = st.session_state.get(
    "target_column",
    df.columns[-1],
)

default_index = (
    list(df.columns).index(default_target)
    if default_target in df.columns
    else 0
)

target = st.selectbox(
    "🎯 Select Target Column",
    options=df.columns,
    index=default_index,
)

st.session_state["target_column"] = target

st.divider()

# ==========================================================
# Task Selection
# ==========================================================

task = st.radio(
    "Choose Machine Learning Task",
    [
        "Classification",
        "Regression",
    ],
    horizontal=True,
)

st.divider()

# ==========================================================
# Model Selection
# ==========================================================

if task == "Classification":

    model_name = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "Support Vector Machine",
        ],
    )

else:

    model_name = st.selectbox(
        "Select Model",
        [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
        ],
    )

st.divider()

# ==========================================================
# Train/Test Split
# ==========================================================

test_size = st.slider(
    "Test Size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
)

random_state = st.number_input(
    "Random State",
    value=42,
    step=1,
)

st.divider()
# ==========================================================
# Train Model
# ==========================================================

if st.button(
    "🚀 Train Model",
    type="primary",
):

    try:

        # --------------------------------------------------
        # Prepare Features & Target
        # --------------------------------------------------

        X = df.drop(columns=[target])
        y = df[target]

        # --------------------------------------------------
        # Split Dataset
        # --------------------------------------------------

        X_train, X_test, y_train, y_test = split_data(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state),
        )

        # --------------------------------------------------
        # Train Model
        # --------------------------------------------------

        with st.spinner("Training model..."):

            model = get_model(
                task,
                model_name,
            )

            model = train_selected_model(
                model,
                X_train,
                y_train,
            )

            predictions = predict(
                model,
                X_test,
            )

        # --------------------------------------------------
        # Save Session State
        # --------------------------------------------------

        st.session_state["trained_model"] = model
        st.session_state["predictions"] = predictions

        st.session_state["task"] = task
        st.session_state["model_name"] = model_name

        st.session_state["target_column"] = target

        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test

        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        st.session_state["feature_names"] = list(X.columns)

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        logger.info(
            f"Model trained successfully | "
            f"Task={task} | "
            f"Model={model_name}"
        )

        # --------------------------------------------------
        # Success
        # --------------------------------------------------

        st.success("✅ Model Trained Successfully!")

        st.divider()

        # ==================================================
        # Training Summary
        # ==================================================

        st.subheader("📋 Training Summary")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Model",
                model_name,
            )

            st.metric(
                "Task",
                task,
            )

        with col2:

            st.metric(
                "Training Samples",
                len(X_train),
            )

            st.metric(
                "Testing Samples",
                len(X_test),
            )

        st.divider()

        # ==================================================
        # Dataset Split Overview
        # ==================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Train %",
                f"{(1-test_size)*100:.0f}%",
            )

        with col2:

            st.metric(
                "Test %",
                f"{test_size*100:.0f}%",
            )

        with col3:

            st.metric(
                "Features",
                X.shape[1],
            )

        st.info(
            "➡ Open the **Results** page from the sidebar to view evaluation metrics and visualizations."
        )

    except Exception as e:

        logger.exception(e)

        st.error(
            f"❌ Model training failed.\n\n{e}"
        )