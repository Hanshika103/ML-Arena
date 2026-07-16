import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_csv(file):
    """
    Load CSV file with caching.
    """
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def get_dataset_summary(df):
    """
    Return basic dataset summary.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


@st.cache_data(show_spinner=False)
def correlation_matrix(df):
    """
    Compute correlation matrix for numeric columns.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return None

    return numeric_df.corr()


@st.cache_data(show_spinner=False)
def get_numeric_columns(df):
    """
    Return numeric column names.
    """
    return df.select_dtypes(include="number").columns.tolist()


@st.cache_data(show_spinner=False)
def get_categorical_columns(df):
    """
    Return categorical column names.
    """
    return df.select_dtypes(exclude="number").columns.tolist()