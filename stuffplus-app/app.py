import streamlit as st
import pandas as pd
import traceback

st.title("Data load test")

DF_SCORED_URL = "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/df_scored.parquet?download=true"

@st.cache_data
def load_df_scored():
    return pd.read_parquet(DF_SCORED_URL)

st.write("About to load df_scored")

try:
    df_scored = load_df_scored()
    st.success(f"Loaded df_scored: {df_scored.shape}")
except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
    st.code(traceback.format_exc())