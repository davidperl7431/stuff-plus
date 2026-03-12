import streamlit as st
import pandas as pd

st.title("Data load test")

DF_SCORED_URL = "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/df_scored.parquet?download=true"
ARSENAL_SUMMARY_URL = "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/arsenal_summary.parquet?download=true"

@st.cache_data
def load_df_scored():
    return pd.read_parquet(DF_SCORED_URL)

@st.cache_data
def load_arsenal_summary():
    return pd.read_parquet(ARSENAL_SUMMARY_URL)

st.write("About to load df_scored")
df_scored = load_df_scored()
st.write("Loaded df_scored", df_scored.shape)

st.write("About to load arsenal_summary")
arsenal_summary = load_arsenal_summary()
st.write("Loaded arsenal_summary", arsenal_summary.shape)