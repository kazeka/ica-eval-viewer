"""ICA evaluation viewer, public build: the Model 1 page only (written by scripts/build_hf_space.py, do not edit).

Run:  streamlit run viewer/app.py
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="ICA Model 1 evaluation viewer", layout="wide")
st.markdown("""<style>
div[data-testid="stMetric"] {background:#EEF1F5;border:1px solid #C9D1DB;padding:8px 12px;}
.block-container {padding-top:1.6rem;}
</style>""", unsafe_allow_html=True)
st.navigation([st.Page(str(HERE / "m1_trigger.py"), title="Model 1 · objection trigger", default=True)],
              position="hidden").run()
