import streamlit as st
from copy import deepcopy

# ==========================================================
# Default Session State
# ==========================================================

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


# ==========================================================
# Initialize Session State
# ==========================================================

def initialize_session_state() -> None:
    """
    Initialize all required Streamlit session state variables.
    """

    for key, value in DEFAULT_SESSION_STATE.items():

        if key not in st.session_state:

            # Prevent shared mutable objects
            st.session_state[key] = deepcopy(value)


# ==========================================================
# Reset Session State
# ==========================================================

def reset_session_state() -> None:
    """
    Reset the complete ML workflow.
    """

    for key, value in DEFAULT_SESSION_STATE.items():

        st.session_state[key] = deepcopy(value)