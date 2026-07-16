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

if not validate_dataset():
    st.stop()

load_theme()

# ----------------------------
# Initialize Session State
# ----------------------------

defaults = {
    "processed_data": None,
    "trained_model": None,
    "predictions": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "task": None,
    "model_name": None,
    "target_column": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ----------------------------
# Check Dataset
# ----------------------------

if st.session_state["processed_data"] is None:
    st.warning("⚠ Please complete Data Preprocessing first.")
    st.stop()

df = st.session_state["processed_data"].copy()

# ----------------------------
# Page Title
# ----------------------------

st.title("🧠 Model Training")
st.success("✅ Processed Dataset Loaded Successfully")

# ----------------------------
# Dataset Information
# ----------------------------

rows, cols = df.shape

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Rows", rows)

with c2:
    st.metric("Columns", cols)

with c3:
    st.metric(
        "Memory Usage",
        f"{df.memory_usage(deep=True).sum()/1024:.1f} KB",
    )

st.divider()

# ----------------------------
# Target Column
# ----------------------------

target = st.selectbox(
    "🎯 Select Target Column",
    df.columns,
)

st.session_state["target_column"] = target

# ----------------------------
# Task Selection
# ----------------------------

task = st.radio(
    "Choose ML Task",
    ["Classification", "Regression"],
    horizontal=True,
)

# ----------------------------
# Model Selection
# ----------------------------

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

# ----------------------------
# Train/Test Split
# ----------------------------

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

# ----------------------------
# Train Model
# ----------------------------

if st.button("🚀 Train Model"):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
    )

    with st.spinner("Training model..."):

        model = get_model(task, model_name)

        model = train_selected_model(
            model,
            X_train,
            y_train,
        )

        predictions = predict(
            model,
            X_test,
        )
            # ----------------------------
    # Save in Session State
    # ----------------------------

    st.session_state["trained_model"] = model
    st.session_state["predictions"] = predictions
    st.session_state["task"] = task
    st.session_state["model_name"] = model_name
    st.session_state["target_column"] = target

    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

    # ----------------------------
    # Success Message
    # ----------------------------

    st.success("✅ Model Trained Successfully!")

    st.divider()

    st.subheader("📋 Training Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model", model_name)
        st.metric("Task", task)

    with col2:
        st.metric("Training Samples", len(X_train))
        st.metric("Testing Samples", len(X_test))

    st.info("➡ Open the **Results** page from the sidebar to view metrics and visualizations.")