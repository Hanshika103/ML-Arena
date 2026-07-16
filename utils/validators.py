import streamlit as st


def validate_dataset():
    """Check if dataset is available."""
    if st.session_state.get("dataset") is None:
        st.warning("⚠️ Please upload a dataset first.")
        return False
    return True


def validate_processed_data():
    """Check if preprocessing has been completed."""
    if st.session_state.get("processed_data") is None:
        st.warning("⚠️ Please preprocess the dataset first.")
        return False
    return True


def validate_target():
    """Check if target column is selected."""
    if st.session_state.get("target_column") is None:
        st.warning("⚠️ Please select a target column.")
        return False
    return True


def validate_model():
    """Check if model has been trained."""
    if st.session_state.get("trained_model") is None:
        st.warning("⚠️ Please train a model first.")
        return False
    return True


def validate_predictions():
    """Check if predictions are available."""
    if st.session_state.get("predictions") is None:
        st.warning("⚠️ Please generate predictions first.")
        return False
    return True