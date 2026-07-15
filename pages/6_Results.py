import streamlit as st
import pandas as pd
from utils.comparison import compare_models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ui.theme import load_theme
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np



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
]

for item in required:

    if item not in st.session_state:

        st.warning("⚠ Please train a model first.")

        st.stop()

if st.session_state["trained_model"] is None:

    st.warning("⚠ Please train a model first.")

    st.stop()

# ----------------------------
# Load Data
# ----------------------------

predictions = st.session_state["predictions"]

y_test = st.session_state["y_test"]

task = st.session_state["task"]

model_name = st.session_state["model_name"]

st.title("📈 Model Results")

st.success("✅ Results Generated Successfully")
st.write("Task:", task)
st.write("Model:", model_name)
st.write("Predictions Length:", len(predictions))
st.write("Y Test Length:", len(y_test))

# ----------------------------
# Performance Metrics
# ----------------------------

st.divider()
st.subheader("📊 Performance Metrics")

if task == "Classification":

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    # Save Metrics
    st.session_state["metrics"] = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
    }

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")

    with c2:
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")

else:

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    # Save Metrics
    st.session_state["metrics"] = {
    "MAE": round(mae, 4),
    "MSE": round(mse, 4),
    "RMSE": round(rmse, 4),
    "R2 Score": round(r2, 4),
    }

    c1, c2 = st.columns(2)

    with c1:
        st.metric("MAE", f"{mae:.4f}")
        st.metric("MSE", f"{mse:.4f}")

    with c2:
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("R² Score", f"{r2:.4f}")
# ----------------------------
# Predictions Preview
# ----------------------------

# ----------------------------
# Predictions Preview
# ----------------------------

st.divider()

st.subheader("📋 Predictions Preview")

results_df = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": predictions
})

st.dataframe(
    results_df.head(20),
    use_container_width=True,
    height=400,
)
# ----------------------------
# Download Predictions
# ----------------------------

csv = results_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Predictions",
    data=csv,
    file_name="predictions.csv",
    mime="text/csv",
    use_container_width=True,
)




if task == "Classification":
# ----------------------------
# Confusion Matrix
# ----------------------------

    st.divider()

    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(cm)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    classes = sorted(y_test.unique())

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                    cm[i, j],
                ha="center",
                va="center",
                fontsize=12,
            )

    plt.colorbar(image)

    st.pyplot(fig)

# ----------------------------
# Classification Report
# ----------------------------

    st.divider()

    st.subheader("📄 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True,
    )

else: 
# ----------------------------
# Actual vs Predicted Plot
# ----------------------------

    st.divider()

    st.subheader("📈 Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        y_test,
        predictions,
        alpha=0.7,
    )

# Perfect Prediction Line
    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))

    ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
)

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")

    st.pyplot(fig)

# ----------------------------
# Residual Plot
# ----------------------------

    st.divider()    

    st.subheader("📉 Residual Plot")    

    residuals = y_test - predictions    

    fig, ax = plt.subplots(figsize=(6, 5))  

    ax.scatter(
        predictions,
        residuals,
        alpha=0.7,
    )

    ax.axhline(
        y=0,
        linestyle="--",
    )

    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")

    st.pyplot(fig)

    # ----------------------------
# Model Comparison
# ----------------------------
if not all(
    key in st.session_state
    for key in ["X_train", "X_test", "y_train", "y_test"]
):
    st.warning("⚠ Please train the model first.")
    st.stop()
st.divider()

st.subheader("🏆 Model Comparison")
if st.button("Compare All Models", use_container_width=True):

    comparison_df = compare_models(
    st.session_state["X_train"],
    st.session_state["X_test"],
    st.session_state["y_train"],
    st.session_state["y_test"],
    st.session_state["task"],
)

    st.dataframe(
    comparison_df,
    use_container_width=True,
)

    best_model = comparison_df.iloc[0]["Model"]

    st.success(f"🏆 Best Model: {best_model}")

    csv = comparison_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Comparison Report",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )