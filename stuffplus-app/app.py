import streamlit as st
import pandas as pd

st.title("Arsenal summary test")

ARSENAL_SUMMARY_URL = "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/arsenal_summary.parquet?download=true"

@st.cache_data
def load_arsenal_summary():
    return pd.read_parquet(ARSENAL_SUMMARY_URL)

st.write("About to load arsenal_summary")
arsenal_summary = load_arsenal_summary()
st.success(f"Loaded arsenal_summary: {arsenal_summary.shape}")