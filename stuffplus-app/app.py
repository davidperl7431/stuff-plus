import streamlit as st
import pandas as pd
import requests
import io

st.title("df_scored download test")

URL = "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/df_scored.parquet?download=true"

st.write("Downloading file...")

r = requests.get(URL, timeout=120)

st.write("Download complete")
st.write("Bytes:", len(r.content))

st.write("Parsing parquet...")

df = pd.read_parquet(io.BytesIO(r.content))

st.success("Loaded!")
st.write(df.shape)