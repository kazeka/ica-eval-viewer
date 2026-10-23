"""ICA evaluation viewer, scoring-checks page: the T1–T8 pipeline (ica_eval.validation). Which scoring methods may count
and go into the reward, where each one stops, and why.

Data: data/sandbox_validation/ (adapters/sandbox_validation.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ica_eval import validation as V  # noqa: E402

D = ROOT / "data" / "sandbox_validation"
COLOR = {V.GOOD: "#DCEBDD", V.BAD: "#F6DADA", V.INCONCLUSIVE: "#F7EDD0"}
ORDER = ["T5", "T6", "T1", "T2", "T2_length", "T3", "T8", "T4", "T7"]


@st.cache_data
def load():
    return (pd.read_parquet(D / "validation_long.parquet"), pd.read_csv(D / "validation_status.csv"),
            (D / "reproduction.md").read_text(encoding="utf-8") if (D / "reproduction.md").exists() else "")


res, status, repro = load()
bars = V.load_bars()
st.title("Scoring checks · T1–T8 pipeline (sandbox)")
st.caption("Each scoring method moves to the next stage only if every check in the current one is Good. "
           + " → ".join(f"**{s['name']}**: {', '.join(s['tests'])}" for s in bars["stages"])
           + ". T7 (sandbox only) is reported, not required. Rule checks are a gate: T2, T3, T4 and T8 do not apply to them.")

# ---- summary grid: one cell per scorer x check, the worst verdict over its items and input sets
GRID = ["T5", "T6", "T1", "T2", "T3", "T8", "T4", "T7"]
SYMBOL = {V.GOOD: "✓ Good", V.BAD: "✗ Bad", V.INCONCLUSIVE: "~ Inconclusive", V.NOT_APPLICABLE: "n/a", V.REPORT: "report"}


def worst(vs):
    vs = set(vs)
    for v in (V.BAD, V.INCONCLUSIVE, V.GOOD, V.REPORT, V.NOT_APPLICABLE):
        if v in vs:
            return v
    return V.NOT_RUN


g = res.assign(t=res.test.str.split("_").str[0]).groupby(["scorer", "t"]).verdict.agg(worst).unstack()
g = g.reindex(columns=[c for c in GRID if c in g.columns]).reindex(status.scorer)
grid = g.apply(lambda col: col.map(lambda v: SYMBOL.get(v, v) if isinstance(v, str) else "–"))
SHORT = {"T5": "T5 drops when worse", "T6": "T6 same when reworded", "T1": "T1 stable over time",
         "T2": "T2 predicts sales", "T3": "T3 sold > lost", "T8": "T8 uplift is real", "T4": "T4 agrees with SMEs",
         "T7": "T7 true quality"}
HELP = {c: f"{SHORT.get(c, c)}: {bars['checks'].get(c, {}).get('title', '')}" for c in grid.columns}
grid.insert(0, "Result", status.set_index("scorer").status.reindex(grid.index)
            .map(lambda s: "passes" if s.startswith("passed") else "stops at stage " + s.split("stage ")[-1].split()[0]).values)
grid.index.name = "Scoring method"


def paint_grid(v):
    for k, c in COLOR.items():
        if isinstance(v, str) and v.endswith(k):
            return f"background-color:{c}"
    if isinstance(v, str) and v.startswith("passes"):
        return f"background-color:{COLOR[V.GOOD]};font-weight:600"
    if isinstance(v, str) and v.startswith("stops"):
        return f"background-color:{COLOR[V.BAD]};font-weight:600"
    return "color:#7A8494"


st.dataframe(grid.style.map(paint_grid), width="stretch",
             column_config={c: st.column_config.TextColumn(c, help=h) for c, h in HELP.items()})
st.caption(" · ".join(SHORT[c] for c in HELP if c in SHORT))
st.caption("Each cell is the worst verdict over the check's items (e.g. each kind of worse reply for T5). Columns follow "
           "the run order: stage 1 automatic (T5, T6, T1, T2, T3, T8), then stage 2 SMEs (T4). T7 compares with the "
           "sandbox's hidden truth and is reported only. Simulated SMEs in the sandbox.")

with st.expander("Why each scoring method stops"):
    st.dataframe(status[["scorer", "status", "reasons"]].rename(columns=str.capitalize), hide_index=True, width="stretch")

st.subheader("All checks")
c1, c2 = st.columns(2)
sets = c1.multiselect("Input sets", sorted(x for x in res.set.unique() if x), default=sorted(x for x in res.set.unique() if x))
show_after = c2.toggle("Show checks after a scorer's stopping stage", value=True)
r = res[(res.set.isin(sets)) | (res.set == "")]
if not show_after:
    r = r[r.counts]
r = r.assign(_o=r.test.map({t: i for i, t in enumerate(ORDER)})).sort_values(["_o", "set", "item"])


def cell(x):
    if pd.isna(x.value):
        return f"– · {x.verdict}"
    iv = f" [{x.lo:.2f}, {x.hi:.2f}]" if pd.notna(x.lo) else ""
    return f"{x.value:.2f}{iv} · {x.verdict}"


r["cell"] = r.apply(cell, axis=1)
r["Check"] = r.test.str.replace("_length", " (beyond length)") + " · " + r.test.map(
    lambda t: bars["checks"].get(t, {}).get("title", ""))
r["Target"] = r.apply(lambda x: "–" if pd.isna(x.target) else f"{'≥' if x.higher else '≤'} {x.target:g}", axis=1)
wide = r.pivot_table(index=["_o", "Check", "set", "item", "Target"], columns="scorer", values="cell", aggfunc="first",
                     sort=False).reset_index().drop(columns="_o").rename(columns={"set": "Input set", "item": "Item"})
scorers = [c for c in wide.columns if c not in ("Check", "Input set", "Item", "Target")]


def paint(v):
    for k, c in COLOR.items():
        if isinstance(v, str) and v.endswith(k):
            return f"background-color:{c}"
    return "color:#7A8494" if isinstance(v, str) and ("Not applicable" in v or "Report" in v) else ""


st.dataframe(wide.style.map(paint, subset=scorers), hide_index=True, width="stretch", height=35 * (len(wide) + 1) + 3)
st.caption("Value [95% interval] · verdict. T5: share of worse replies scored lower. T6: mean score change in standard "
           "deviations. T1: rank correlation between scorers refitted on different past periods. T2: AUC gain over "
           "context controls (and over controls + reply length). T3: matched sold/lost pairs ranked right. T8: score drop "
           "when the model's reply is moved to another call. T4/T7: pairs ranked as the SMEs' majority / the hidden truth.")

t8 = res[res.test == "T8"]
if len(t8) and "uplift" in t8:
    st.subheader("T8 detail: uplift over agent replies by each score")
    def pct(v):
        return "–" if pd.isna(v) else f"{v:+.1%}"
    d = pd.DataFrame({
        "Scoring method": t8.scorer, "Replies compared": t8.set,
        "Uplift by this score [95% interval]": [f"{pct(u)} [{pct(lo)}, {pct(hi)}]" for u, lo, hi in zip(t8.uplift, t8.uplift_lo, t8.uplift_hi)],
        "Uplift with replies moved to another call": t8.uplift_moved.map(pct),
        **({"True uplift (sandbox)": t8.true_uplift.map(pct)} if "true_uplift" in t8 else {})})
    st.dataframe(d, hide_index=True, width="stretch")
    st.caption("A trustworthy score's uplift should vanish (or turn negative) when each model reply is moved to an "
               "unrelated call, and should be close to the true uplift.")
if repro:
    with st.expander("Reproduction of Merxhan's evalval numbers"):
        st.markdown(repro)
        st.caption("T6 paraphrase differs slightly by design: replies the rewording left unchanged are excluded here "
                   "(evalval counts them as zero change).")
