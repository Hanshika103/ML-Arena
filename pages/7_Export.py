import streamlit as st
import pandas as pd
import joblib
import io
from datetime import datetime

from ui.theme import load_theme

# ----------------------------
# Load Theme
# ----------------------------

load_theme()

# ----------------------------
# Session State Check
# ----------------------------

required = [
    "trained_model",
    "predictions",
    "y_test",
    "task",
    "model_name",
    "metrics",
    "X_train",
    "X_test",
    "target_column",
]

missing = []

for key in required:
    if key not in st.session_state:
        missing.append(f"{key} (not found)")
    elif st.session_state[key] is None:
        missing.append(f"{key} (None)")

if missing:
    st.error("Missing Session State Variables:")
    st.write(missing)
    st.stop()

# ----------------------------
# Load Session Data
# ----------------------------

model = st.session_state["trained_model"]
predictions = st.session_state["predictions"]
y_test = st.session_state["y_test"]
metrics = st.session_state["metrics"]

task = st.session_state["task"]
model_name = st.session_state["model_name"]

X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]

target_column = st.session_state["target_column"]

# ----------------------------
# Page Header
# ----------------------------

st.title("⬇️ Export Center")

st.success(
    "Export your trained model, predictions, and evaluation reports."
)

st.divider()

# ----------------------------
# Project Summary
# ----------------------------

st.subheader("📦 Project Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Task", task)

with c2:
    st.metric("Model", model_name)

with c3:
    st.metric("Target Column", target_column)

st.divider()

# =====================================================
# Model Information
# =====================================================

st.subheader("🤖 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.info(f"**Model Name:** {model_name}")
    st.info(f"**Task Type:** {task}")
    st.info(f"**Target Column:** {target_column}")

with col2:
    st.info(f"**Training Samples:** {len(X_train)}")
    st.info(f"**Testing Samples:** {len(X_test)}")
    st.info(f"**Features Used:** {X_train.shape[1]}")

st.divider()

# =====================================================
# Experiment Summary
# =====================================================

st.subheader("🧪 Experiment Summary")

summary = pd.DataFrame({
    "Property": [
        "Model",
        "Task",
        "Training Samples",
        "Testing Samples",
        "Features",
        "Target Column",
        "Predictions",
        "Generated On"
    ],
    "Value": [
        model_name,
        task,
        len(X_train),
        len(X_test),
        X_train.shape[1],
        target_column,
        len(predictions),
        datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)

st.divider()

# =====================================================
# Download Center
# =====================================================

st.subheader("📥 Download Center")

# ----------------------------
# Download Trained Model
# ----------------------------

model_buffer = io.BytesIO()
joblib.dump(model, model_buffer)
model_buffer.seek(0)

st.download_button(
    label="💾 Download Trained Model (.pkl)",
    data=model_buffer,
    file_name=f"{model_name.lower().replace(' ', '_')}_model.pkl",
    mime="application/octet-stream",
    use_container_width=True,
)

# ----------------------------
# Download Predictions
# ----------------------------

prediction_df = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": predictions
})

prediction_csv = prediction_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📄 Download Predictions (.csv)",
    data=prediction_csv,
    file_name="predictions.csv",
    mime="text/csv",
    use_container_width=True,
)

# ----------------------------
# Download Metrics
# ----------------------------

metrics_df = pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Value"]
)

metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📊 Download Metrics (.csv)",
    data=metrics_csv,
    file_name="metrics.csv",
    mime="text/csv",
    use_container_width=True,
)

st.success("✅ All export files are ready for download!")

# =====================================================
# Export Statistics
# =====================================================

st.subheader("📊 Export Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Files Ready", "3")

with col2:
    st.metric("Predictions", len(predictions))

with col3:
    st.metric("Metrics", len(metrics))

st.divider()

# =====================================================
# Metrics Preview
# =====================================================

st.subheader("📈 Metrics Preview")

metrics_df = pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Value"]
)

st.dataframe(
    metrics_df,
    use_container_width=True,
    hide_index=True,
)

st.divider()

# =====================================================
# Predictions Preview
# =====================================================

st.subheader("📄 Prediction Preview")

prediction_df = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": predictions
})

st.dataframe(
    prediction_df.head(10),
    use_container_width=True,
    hide_index=True,
)