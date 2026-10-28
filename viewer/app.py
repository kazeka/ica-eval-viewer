"""ICA evaluation viewer (entry point). One page per evaluation family.

Run:  uv run streamlit run viewer/app.py
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="ICA evaluation viewer", layout="wide")
st.markdown("""<style>
div[data-testid="stMetric"] {background:#EEF1F5;border:1px solid #C9D1DB;padding:8px 12px;}
.block-container {padding-top:1.6rem;}
</style>""", unsafe_allow_html=True)

# The default page is served at "/"; every other page has its own address, so links can be shared.
pages = [st.Page(str(HERE / "overview.py"), title="Overview", default=True),
         st.Page(str(HERE / "m1_trigger.py"), title="Model 1 · objection trigger", url_path="model1")]
if (HERE.parent / "data" / "sandbox_m2" / "m2_replies.parquet").exists():
    pages.append(st.Page(str(HERE / "m2_replies.py"), title="Model 2 · replies (sandbox)", url_path="model2"))
if (HERE.parent / "data" / "synth_b0" / "meta.json").exists():
    pages.append(st.Page(str(HERE / "b0_report.py"), title="Model 2 · B0 sensitivity (synthetic)", url_path="b0"))
if (HERE.parent / "data" / "sandbox_validation" / "validation_long.parquet").exists():
    pages.append(st.Page(str(HERE / "validation_page.py"), title="Scoring checks · T1–T8 (sandbox)", url_path="checks"))
st.navigation(pages).run()
