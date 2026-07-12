import streamlit as st
import pandas as pd
from ui.theme import load_theme
from utils.model_training import (
    split_data,
    get_model,
    train_selected_model,
    predict,
)
if "X_test" not in st.session_state:
    st.session_state["X_test"] = None

if "y_test" not in st.session_state:
    st.session_state["y_test"] = None

if "task" not in st.session_state:
    st.session_state["task"] = None

if "model_name" not in st.session_state:
    st.session_state["model_name"] = None
load_theme()

# ----------------------------
# Session State
# ----------------------------

if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None

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
        f"{df.memory_usage(deep=True).sum()/1024:.1f} KB"
    )

st.divider()


# ----------------------------
# Target Column
# ----------------------------

target = st.selectbox(
    "🎯 Select Target Column",
    df.columns
)

st.session_state["target_column"] = target

# ----------------------------
# Task Selection
# ----------------------------

task = st.radio(
    "Choose ML Task",
    [
        "Classification",
        "Regression"
    ],
    horizontal=True
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
    0.1,
    0.4,
    0.2,
)

random_state = st.number_input(
    "Random State",
    value=42,
)

st.divider()

# ----------------------------
# Train Model
# ----------------------------

if st.button("🚀 Train Model", use_container_width=True):

    X = df.drop(columns=[target])
    y = df[target]
    st.write("Target Column:", target)
    st.write("Unique Target Values:", y.unique())

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
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["predictions"] = predictions
    st.session_state["task"] = task
    st.session_state["model_name"] = model_name

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

    st.info("➡ Next open **Results** from the sidebar.")

    