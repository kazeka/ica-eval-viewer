"""ICA evaluation viewer, overview page: the evaluation pipeline on one screen and how far each part is.

Live facts (models, runs, check results) are read from the data folders; statuses are maintained here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"

DONE, PART, TODO, NEEDS = "Built and tested", "Partly built", "Not built yet", "Needs AmEx"
STYLE = {DONE: ("#DCEBDD", "#2E6B3A"), PART: ("#F7EDD0", "#7A5B12"), TODO: ("#ECEEF2", "#5B6575"), NEEDS: ("#E3ECF7", "#1F3A5F")}


def facts():
    f = {}
    p = D / "public_m1" / "m1_points.parquet"
    if p.exists():
        f["m1_models"] = sorted(pd.read_parquet(p, columns=["candidate"]).candidate.unique())
    p = D / "sandbox_m2" / "m2_runs.json"
    if p.exists():
        r = pd.read_parquet(D / "sandbox_m2" / "m2_replies.parquet", columns=["run", "policy"]).drop_duplicates()
        f["m2_runs"] = {k: sorted(g.policy) for k, g in r.groupby("run")}
    p = D / "sandbox_validation" / "validation_status.csv"
    if p.exists():
        f["checks"] = pd.read_csv(p)
    return f


F = facts()
st.title("ICA evaluation · overview")
st.caption("Phase 1 evaluation framework. Everything below runs on public data (Model 1) and on the team's synthetic "
           "sandbox (Model 2); AmEx data only needs an adapter. Other pages hold the details.")

STEPS = [
    ("Data", "Calls, turns, outcomes, objection types", NEEDS,
     "Public objection data (Model 1) and the synthetic sandbox (Model 2) map to the same tables."),
    ("Model 1 metrics", "Recall, false alarms, calibration, latency", DONE,
     f"{len(F.get('m1_models', []))} models compared on public data, by objection type and context length."),
    ("Model 2 metrics", "Text metrics, rule checks, failures", PART,
     "Text and rule metrics built; embedding, entailment and judge metrics need approved self-hosted models."),
    ("Scoring checks T1–T8", "Can a score be trusted?", DONE,
     "Staged pipeline; reproduces the sandbox results. T4 uses simulated SMEs until real ratings exist."),
    ("Reward gate R1–R4", "Is the reward safe for GRPO?", TODO, "Reward formula and gate checks are next."),
    ("Training runs", "SFT → DPO → GRPO, tracked", PART,
     "Runs end to end locally with MLflow tracking; full run reached DPO round 2."),
    ("Final comparison", "SME blind A/B vs B0", NEEDS, "Needs SME time and the blind-pair tooling."),
]


def box(title, sub, status):
    bg, fg = STYLE[status]
    return (f"<div style='flex:1 1 0;min-width:0;border:1px solid #C9D1DB;background:{bg};padding:8px 10px'>"
            f"<div style='font-weight:600;color:#1E2833;font-size:0.92em'>{title}</div>"
            f"<div style='font-size:0.82em;color:#4A5563;margin:2px 0 6px'>{sub}</div>"
            f"<div style='font-size:0.78em;font-weight:600;color:{fg}'>{status}</div></div>")


arrow = "<div style='align-self:center;color:#7A8494;font-size:1.3em'>→</div>"
st.markdown("<div style='display:flex;gap:4px;flex-wrap:nowrap;margin:8px 0 18px'>"
            + arrow.join(box(t, s, x) for t, s, x, _ in STEPS) + "</div>", unsafe_allow_html=True)

st.subheader("Where each part stands")
st.dataframe(pd.DataFrame([{"Part": t, "Status": x, "Detail": d} for t, _, x, d in STEPS])
             .style.map(lambda v: f"background-color:{STYLE[v][0]}" if v in STYLE else "", subset=["Status"]),
             hide_index=True, width="stretch",
             column_config={"Part": st.column_config.TextColumn(width="small"),
                            "Status": st.column_config.TextColumn(width="small"),
                            "Detail": st.column_config.TextColumn(width="large")})
st.caption("Built and tested = runs end to end on public data (Model 1) or the synthetic sandbox (Model 2).")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Sandbox results so far")
    if "checks" in F:
        ch = F["checks"]
        ok = ch.status.str.startswith("passed")
        st.markdown(f"**Scoring checks:** {int(ok.sum())} of {len(ch)} scoring methods pass stage 1.")
        for r in ch.itertuples():
            st.markdown(f"- {r.scorer}: {'passes' if r.status.startswith('passed') else 'stops at stage 1 (' + r.reasons.split(';')[0] + ')'}")
    for run, pols in (F.get("m2_runs") or {}).items():
        st.markdown(f"**Model 2 run `{run}`:** {', '.join(p for p in pols)}")
with c2:
    st.subheader("What AmEx data and SMEs unlock")
    st.markdown("""
- **AmEx calls and outcomes:** Model 1 on real calls (context length, early warning), T1–T3 on real later months.
- **Approved self-hosted models (Qwen, Kimi):** embedding, entailment and LLM-judge metrics.
- **SME ratings:** T4 agreement, reward weights, DPO pair checks, the final blind A/B.
- **Fact sheet and term lists:** invented-offer and forbidden-term checks against real approved numbers.
""")
st.caption("Pages: Model 1 · objection trigger, Model 2 · replies, Model 2 · B0 sensitivity, Scoring checks · T1–T8 "
           "(left menu).")
