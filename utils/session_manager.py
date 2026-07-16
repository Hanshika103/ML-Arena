import streamlit as st


DEFAULT_SESSION_STATE = {
    "dataset": None,
    "processed_data": None,
    "target_column": None,
    "problem_type": None,
    "trained_model": None,
    "trained_models": {},
    "best_model": None,
    "predictions": None,
    "metrics": None,
    "task": None,
    "model_name": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "feature_names": None,
}


def initialize_session_state():
    """
    Initialize all required session state variables.
    """
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value