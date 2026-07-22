import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    roc_curve,
    auc,
    precision_recall_curve,
)

from ui.theme import load_theme
from utils.validators import validate_model
from utils.comparison import compare_models
from utils.logger import get_logger

# ==========================================================
# Load Theme
# ==========================================================

load_theme()
logger = get_logger()


# ==========================================================
# Validate Model
# ==========================================================

if not validate_model():
    st.stop()

# ==========================================================
# Check Session State
# ==========================================================

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

# ==========================================================
# Load Data
# ==========================================================

model = st.session_state["trained_model"]
predictions = st.session_state["predictions"]

y_test = st.session_state["y_test"]
X_test = st.session_state["X_test"]

task = st.session_state["task"]
model_name = st.session_state["model_name"]

logger.info(
    f"Results page opened | Task={task} | Model={model_name}"
)

# ==========================================================
# Page Title
# ==========================================================

st.title("📈 Model Results")

st.success("✅ Results Generated Successfully")

st.divider()

# ==========================================================
# Summary
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Task",
        task,
    )

with col2:

    st.metric(
        "Model",
        model_name,
    )

st.divider()

# ==========================================================
# Performance Metrics
# ==========================================================

st.subheader("📊 Performance Metrics")

if task == "Classification":

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

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

    st.session_state["metrics"] = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
    }

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Accuracy",
            f"{accuracy:.4f}",
        )

        st.metric(
            "Precision",
            f"{precision:.4f}",
        )

    with c2:

        st.metric(
            "Recall",
            f"{recall:.4f}",
        )

        st.metric(
            "F1 Score",
            f"{f1:.4f}",
        )

else:

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    st.session_state["metrics"] = {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4),
    }

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "MAE",
            f"{mae:.4f}",
        )

        st.metric(
            "MSE",
            f"{mse:.4f}",
        )

    with c2:

        st.metric(
            "RMSE",
            f"{rmse:.4f}",
        )

        st.metric(
            "R² Score",
            f"{r2:.4f}",
        )

st.divider()
# ==========================================================
# Predictions Preview
# ==========================================================

st.subheader("📋 Predictions Preview")

results_df = pd.DataFrame(
    {
        "Actual": y_test.reset_index(drop=True),
        "Predicted": predictions,
    }
)

st.dataframe(
    results_df.head(20),
    width="stretch",
    height=400,
    hide_index=True,
)

st.divider()

# ==========================================================
# Download Predictions
# ==========================================================

csv = results_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇ Download Predictions",
    data=csv,
    file_name="predictions.csv",
    mime="text/csv",
    width="stretch",
    key="prediction_download_1",
)

# ==========================================================
# Classification Visualizations
# ==========================================================

if task == "Classification":

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    st.divider()

    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    classes = sorted(
        y_test.unique()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(cm)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    ax.set_xticks(
        range(len(classes))
    )

    ax.set_yticks(
        range(len(classes))
    )

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

    plt.close(fig)

    # ------------------------------------------------------
    # Classification Report
    # ------------------------------------------------------

    st.divider()

    st.subheader("📄 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .round(4)
    )

    st.dataframe(
        report_df,
        width="stretch",
    )

    st.divider()
    # ==========================================================
# Predictions Preview
# ==========================================================

st.subheader("📋 Predictions Preview")

results_df = pd.DataFrame(
    {
        "Actual": y_test.reset_index(drop=True),
        "Predicted": predictions,
    }
)

st.dataframe(
    results_df.head(20),
    width="stretch",
    height=400,
    hide_index=True,
)

st.divider()

# ==========================================================
# Download Predictions
# ==========================================================

csv = results_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇ Download Predictions",
    data=csv,
    file_name="predictions.csv",
    mime="text/csv",
    width="stretch",
    key="prediction_download_2",
)

# ==========================================================
# Classification Visualizations
# ==========================================================

if task == "Classification":

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    st.divider()

    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    classes = sorted(
        y_test.unique()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(cm)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    ax.set_xticks(
        range(len(classes))
    )

    ax.set_yticks(
        range(len(classes))
    )

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

    plt.close(fig)

    # ------------------------------------------------------
    # Classification Report
    # ------------------------------------------------------

    st.divider()

    st.subheader("📄 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .round(4)
    )

    st.dataframe(
        report_df,
        width="stretch",
    )

    st.divider()
    # ==========================================================
# Model Comparison
# ==========================================================

st.subheader("🏆 Model Comparison")

required_keys = [
    "X_train",
    "X_test",
    "y_train",
    "y_test",
]

if not all(
    key in st.session_state and st.session_state[key] is not None
    for key in required_keys
):
    st.warning("⚠ Please train a model first.")
    st.stop()

if st.button(
    "🔍 Compare All Models",
    type="primary",
):

    try:

        with st.spinner("Comparing models..."):

            comparison_df = compare_models(
                st.session_state["X_train"],
                st.session_state["X_test"],
                st.session_state["y_train"],
                st.session_state["y_test"],
                st.session_state["task"],
            )

        if comparison_df.empty:

            st.warning("No comparison results available.")

        else:

            st.success("✅ Model comparison completed successfully!")

            st.dataframe(
                comparison_df,
                width="stretch",
                hide_index=True,
            )

            best_model = comparison_df.iloc[0]["Model"]

            st.session_state["best_model"] = best_model

            st.success(
                f"🏆 Recommended Model: **{best_model}**"
            )

            st.divider()

            st.subheader("📌 Recommendation")

            st.info(
                f"""
**Recommended Model:** {best_model}

This model achieved the best overall performance on your dataset
based on the evaluation metrics.

You can export this report from the Export page.
"""
            )

            csv = comparison_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇ Download Comparison Report",
                data=csv,
                file_name="model_comparison.csv",
                mime="text/csv",
                width="stretch",
                key="comparison_download",
            )

    except Exception as e:

        logger.exception(e)

        st.error(
            f"❌ Model comparison failed.\n\n{e}"
        )

st.divider()

# ==========================================================
# Next Step
# ==========================================================

st.success(
    "🎉 Model evaluation completed successfully!"
)

st.info(
    "➡ Open the **Export** page to download the trained model, predictions, metrics, and reports."
)