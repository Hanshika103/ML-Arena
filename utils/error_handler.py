import streamlit as st
from utils.logger import get_logger

logger = get_logger()


def handle_error(error: Exception, user_message: str = "Something went wrong."):
    """
    Display a user-friendly error message and log the actual exception.
    """

    logger.exception(error)

    st.error(f"❌ {user_message}")

    with st.expander("🔍 Technical Details"):
        st.code(str(error))