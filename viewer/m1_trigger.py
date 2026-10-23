"""ICA evaluation viewer, Model 1 page (objection trigger). Tabs follow the Model 1 evaluation doc:
context length × objection type, metrics and checks with Good / Bad / Inconclusive verdicts, business view,
keyword cues, early warning.

Run the viewer:  uv run streamlit run viewer/app.py   (this file is one of its pages)
Data: data/public_m1/ (m1_points, taxonomy; optional turns + cue outputs from `python -m ica_eval.cues`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ica_eval import checks as ck  # noqa: E402
from ica_eval import sweetspot as ss  # noqa: E402
from ica_eval.cues import FIRST_CUE_BINS, KEYWORD_MODEL  # noqa: E402

DATA = {"Public data (Sakda101, development only)": ROOT / "data" / "public_m1"}
BLUES = [[0, "#F4F6F9"], [0.5, "#9DB4CF"], [1, "#1F3A5F"]]
VERDICT_STYLE = {ck.GOOD: "background-color:#DCEBDD", ck.BAD: "background-color:#F6DADA",
                 ck.INCONCLUSIVE: "background-color:#F7EDD0", ck.NA: "color:#7A8494", "Not run": "color:#7A8494"}
N_BOOT = 500



# ------------------------------------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading scores…")
def load(path: str):
    p = Path(path)
    pts = pd.read_parquet(p / "m1_points.parquet")
    order = list(pts.candidate.cat.categories)
    if (p / "m1_points_keyword.parquet").exists():
        kw = pd.read_parquet(p / "m1_points_keyword.parquet")
        cat = [c for c in pts.columns if isinstance(pts[c].dtype, pd.CategoricalDtype)]
        pts = pd.concat([pts.astype({c: object for c in cat}), kw.astype({c: object for c in cat})], ignore_index=True)
        pts = pts.astype({c: "category" for c in cat})
        order += [KEYWORD_MODEL]
    pts["candidate"] = pts.candidate.cat.set_categories(order)
    tax = pd.read_parquet(p / "taxonomy.parquet")
    flags = pd.read_parquet(p / "cue_flags.parquet") if (p / "cue_flags.parquet").exists() else None
    cues = pd.read_parquet(p / "cues.parquet") if (p / "cues.parquet").exists() else None
    return pts, tax, flags, cues


@st.cache_data(show_spinner="Computing metrics for every context length…")
def compute(path: str, candidate: str, rule: str, budget: float, level: str, groups: tuple | None):
    pts = load(path)[0]
    return ss.grid(pts, candidate, rule, budget, level, list(groups) if groups else None)


@st.cache_data(show_spinner="Counting per call…")
def per_call(path: str, candidate: str, N: int, part: str, t: float, lookahead: int = 6):
    pts, _, flags, _ = load(path)
    d = pts[(pts.candidate == candidate) & (pts.N == N) & (pts.part == part)]
    return ck.per_call(d, t, flags if part == "test" else None, lookahead)


@st.cache_data(show_spinner=False)
def choice(path: str, candidate: str, rule: str, budget: float, min_n: int):
    """Metrics grid, thresholds per N, chosen N* and type flags for one model."""
    g, thr = compute(path, candidate, rule, budget, "type", None)
    n_star, flags = ss.sweet_spot(g, "recall" if rule == "budget" else "f1", min_n)
    return g, thr, n_star, flags


def fmt(v, lo=np.nan, hi=np.nan, d=3):
    if v is None or pd.isna(v):
        return "–"
    return f"{v:.{d}f}" + (f" [{lo:.{d}f}, {hi:.{d}f}]" if pd.notna(lo) else "")


def fdiff(v, lo, hi, d=3):
    return "–" if pd.isna(v) else f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def show_verdicts(df: pd.DataFrame, col: str = "Verdict"):
    st.dataframe(df.style.map(lambda v: VERDICT_STYLE.get(v, ""), subset=[col]), hide_index=True, width="stretch")


def _wrap(text: str, width: int = 16) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


def heatmap(g: pd.DataFrame, metric: str, n_star, min_n: int, flags: pd.DataFrame, cols: list[str]):
    Ns = sorted(g.N.unique())
    z, text = [], []
    flagged = set(flags.group) if not flags.empty else set()
    for N in Ns:
        zr, tr = [], []
        for c in cols:
            cell = g[(g.N == N) & (g.group == c)].set_index("metric")
            n = cell.n_obj.iloc[0] if len(cell) else 0
            v = cell.value.get(metric, np.nan) if len(cell) else np.nan
            if c != ss.OVERALL and n < min_n:
                zr.append(np.nan); tr.append(f"n={int(n)}<br>too few"); continue
            zr.append(v)
            r, p = cell.value.get("recall", np.nan), cell.value.get("precision", np.nan)
            mark = " ⚑" if (N == n_star and c in flagged) else ""
            tr.append(f"<b>{fmt(v)}</b>{mark}<br>R {fmt(r, d=2)} · P {fmt(p, d=2)}<br>n={int(n)}")
        z.append(zr); text.append(tr)
    scale = BLUES[::-1] if metric in ss.LOWER_IS_BETTER else BLUES
    zz = np.array(z, dtype=float)
    zmin, zmax = (np.nanmin(zz), np.nanmax(zz)) if np.isfinite(zz).any() else (0.0, 1.0)
    if zmax - zmin < 0.02:
        zmin = zmax - 0.02
    xlab = [c if len(c) <= 16 else "<br>".join(_wrap(c)) for c in cols]
    ylab = [f"N = {N}" + ("  ★" if N == n_star else "") for N in Ns]
    fig = go.Figure(go.Heatmap(z=z, x=xlab, y=ylab, zmin=zmin, zmax=zmax, text=text, texttemplate="%{text}", colorscale=scale,
                               hoverongaps=False, xgap=2, ygap=2, colorbar=dict(title=ss.LABELS[metric], thickness=12)))
    fig.update_layout(height=110 + 62 * len(Ns), margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#E6E9EE",
                      yaxis=dict(autorange="reversed"), font=dict(size=12), xaxis=dict(side="top"))
    return fig


def lines(g: pd.DataFrame, metric: str, cols: list[str], min_n: int):
    fig = go.Figure()
    for c in cols:
        x = g[(g.group == c) & (g.metric == metric)].sort_values("N")
        if c != ss.OVERALL:
            x = x[x.n_obj >= min_n]
        if x.empty:
            continue
        fig.add_trace(go.Scatter(x=x.N, y=x.value, mode="lines+markers", name=c,
                                 line=dict(width=3 if c == ss.OVERALL else 1.6, dash="solid" if c == ss.OVERALL else "dot"),
                                 error_y=dict(type="data", symmetric=False, array=(x.hi - x.value), arrayminus=(x.value - x.lo),
                                              visible=bool(x.lo.notna().any()), thickness=1)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(type="log", title="context length N (turns)",
                      tickvals=sorted(g.N.unique())), yaxis=dict(title=ss.LABELS[metric]), legend=dict(font=dict(size=11)))
    return fig


# ------------------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Filters")
    src = st.selectbox("Data", list(DATA))
    path = str(DATA[src])
    pts, tax, cue_flags, cues = load(path)
    cands = list(pts.candidate.cat.categories)
    cand = st.selectbox("Model", cands, index=0)
    others = [c for c in cands if c != cand]
    base = st.selectbox("Compare with (stands in for the current trigger, B0)", others,
                        index=others.index(KEYWORD_MODEL) if KEYWORD_MODEL in others else 0)
    rule_lbl = st.radio("Threshold rule (set on the calibration split)",
                        ["Most recall within false-alarm budget", "Best F1"], index=0)
    rule = "budget" if rule_lbl.startswith("Most") else "f1"
    budget = st.select_slider("False-alarm budget per call", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.05,
                              disabled=(rule == "f1"))
    metric = st.selectbox("Heatmap metric", list(ss.LABELS), format_func=ss.LABELS.get, index=0)
    min_n = st.slider("Minimum objections per cell", 10, 100, 30, 5)

g, thr, n_star, flags = choice(path, cand, rule, budget, min_n)
gb, thr_b, n_star_b, _ = choice(path, base, rule, budget, min_n)
Ns = sorted(thr.N)
with st.sidebar:
    n_sel = st.selectbox("Context length for checks and views (turns)", Ns,
                         index=Ns.index(n_star) if n_star in Ns else 0,
                         format_func=lambda n: f"{n} (chosen N*)" if n == n_star else str(n))
    st.caption("Recall is split by the true objection type; precision and false alarms by the predicted type. "
               "Intervals: 95%, resampling whole calls. Comparisons are paired on the calls both models scored.")

t_sel = float(thr.set_index("N").threshold[n_sel])
n_b = n_star_b if n_star_b is not None else int(thr_b.N.iloc[0])
t_b = float(thr_b.set_index("N").threshold[n_b])

st.title("Model 1 · objection trigger evaluation")
st.caption(f"{src} · {cand} at N = {n_sel} (threshold {t_sel:.3f}) · compared with {base} at N = {n_b} (threshold {t_b:.3f}) · "
           f"threshold rule: {rule_lbl.lower()}" + (f" ({budget} per call)" if rule == "budget" else ""))
st.caption("Reading rule: **good** = the whole 95% interval is on the right side of the target; **bad** = the whole interval "
           "is on the wrong side; **inconclusive** = the interval crosses the target.")

_t = st.tabs(["Metrics and checks", "Context length × type", "Business view", "Keyword cues", "Early warning", "Definitions"])
tabs = [_t[1], _t[0], *_t[2:]]  # code below keeps its original tab order; the page opens on Metrics and checks

# ------------------------------------------------------------------------------------------------ 1. context length × type
with tabs[0]:
    o = g[g.group == ss.OVERALL]
    at = o[o.N == n_star].set_index("metric") if n_star is not None else pd.DataFrame()
    c = st.columns(5)
    c[0].metric("Chosen context length N*", f"{n_star} turns" if n_star is not None else "–")
    if not at.empty:
        c[1].metric("Threshold at N*", f"{thr.set_index('N').threshold.get(n_star):.3f}")
        c[2].metric("Recall", fmt(at.value["recall"]), help=fmt(at.value["recall"], at.lo["recall"], at.hi["recall"]))
        c[3].metric("Precision", fmt(at.value["precision"]), help=fmt(at.value["precision"], at.lo["precision"], at.hi["precision"]))
        c[4].metric("False alarms per call", fmt(at.value["fa_per_call"]), help=fmt(at.value["fa_per_call"], at.lo["fa_per_call"], at.hi["fa_per_call"]))
    if not thr.empty and not thr.budget_met.all():
        st.warning("At some context lengths no threshold meets the false-alarm budget on the calibration split; "
                   "the threshold with the fewest false alarms is used there: N = " + ", ".join(map(str, thr[~thr.budget_met].N)))
    if len(thr) == 1:
        st.info("This model was scored at one context length only, so there is no context-length choice to make.")

    st.subheader("All objections, by context length")
    tab = (o.assign(cell=[fmt(v, l, h) for v, l, h in zip(o.value, o.lo, o.hi)])
             .pivot(index="N", columns="metric", values="cell")[list(ss.LABELS)].rename(columns=ss.LABELS))
    tab.insert(0, "Threshold", thr.set_index("N").threshold.map("{:.3f}".format))
    tab.index = [f"{N} ★" if N == n_star else str(N) for N in tab.index]
    tab.index.name = "N"
    st.dataframe(tab, width="stretch")

    st.subheader(f"By objection super-cluster: {ss.LABELS[metric]}")
    types = sorted(g[(g.group != ss.OVERALL)].group.unique())
    cols = [ss.OVERALL] + types
    st.plotly_chart(heatmap(g, metric, n_star, min_n, flags, cols), width="stretch")
    if src.lower().startswith("public"):
        st.info("Rows look alike on public data, and that is expected: each N is scored on a different window (1 to 32 "
                "turns), but every objection is self-contained in one customer message and the synthetic history before it "
                "carries no signal, so the models ignore it (for DeBERTa-v3, 2 of 59,040 test turns move by more than 0.1 "
                "between N = 1 and N = 32). With a generous false-alarm budget the threshold falls in the wide gap between "
                "~0 and ~1 and every N gives the same alerts; with a tight budget it lands among the few borderline turns, "
                "and rows differ by a handful of turns that shift between runs. Real calls are needed to choose N.")
    st.caption("★ chosen context length N* (smallest N whose overall recall is within the 95% interval of the best N). "
               "⚑ at N*, this type's interval lies below its own best context length. Grey: fewer objections than the minimum.")
    if not flags.empty:
        st.markdown("**Types that would prefer a different context length**")
        st.dataframe(flags.rename(columns=dict(group="Type", value_at_n_star="At N*", best_N="Own best N", best_value="At own best")),
                     hide_index=True)
    with st.expander("Trend by context length", expanded=False):
        st.plotly_chart(lines(g, metric, cols, min_n), width="stretch")

    st.subheader("Sub-clusters within a super-cluster")
    by_size = g[(g.group != ss.OVERALL) & (g.metric == "recall")].groupby("group").n_obj.max()
    parent = st.selectbox("Super-cluster", types, index=types.index(by_size.idxmax()) if len(by_size) else 0)
    subs = tuple(sorted(tax[tax.super_cluster == parent].sub_cluster.astype(str)))
    gc, _ = compute(path, cand, rule, budget, "cluster", subs)
    gc = gc[gc.group != ss.OVERALL]
    st.plotly_chart(heatmap(gc, metric, n_star, min_n, pd.DataFrame(), list(subs)), width="stretch")
    st.caption("Same thresholds as above. Precision and false alarms are attributed to the predicted sub-cluster.")

# ------------------------------------------------------------------------------------------------ shared counts
pa = per_call(path, cand, n_sel, "test", t_sel)
pb = per_call(path, base, n_b, "test", t_b)
pa_cal = per_call(path, cand, n_sel, "calib", t_sel)
kw_ok = KEYWORD_MODEL in cands and cand != KEYWORD_MODEL

# ------------------------------------------------------------------------------------------------ 2. metrics and checks
with tabs[1]:
    c1, c2 = st.columns(2)
    p_min = c1.slider("Minimum precision (to agree with AmEx)", 0.5, 1.0, 0.9, 0.01)
    f_floor = c2.slider("Type accuracy floor per super-cluster (macro-F1)", 0.5, 1.0, 0.8, 0.01)
    D = ck.boot_diff(pa, pb, ["recall", "precision", "f1", "fa_per_call", "fa_rate_turn", "ece"], N_BOOT)
    A = ck.boot(pa, ["recall", "precision", "f1", "fa_per_call", "fa_rate_turn", "ece"], N_BOOT)
    gp = g[(g.N == n_sel)].set_index(["group", "metric"]).value
    gbp = gb[(gb.N == n_b)].set_index(["group", "metric"]).value

    def vs_base(m, higher=True, strict=False):
        r = D.loc[m]
        v = ck.verdict(r.lo, r.hi, 0.0, higher)
        if strict and v == ck.GOOD and (r.lo == 0 if higher else r.hi == 0):
            v = ck.INCONCLUSIVE
        return fmt(A.value[m], A.lo[m], A.hi[m]), fmt(r.b), fdiff(r["diff"], r.lo, r.hi), v

    rows = []
    for i, (m, tgt, higher, strict) in enumerate([("recall", "≥ B0", True, False), ("precision", None, True, False),
                                                  ("f1", "≥ B0", True, False), ("fa_per_call", "Fewer than B0", False, True),
                                                  ("fa_rate_turn", "≤ B0", False, False)], start=1):
        val, bv, dv, v = vs_base(m, higher, strict)
        if m == "precision":
            tgt, v = f"≥ {p_min:.2f} (minimum)", ck.verdict(A.lo[m], A.hi[m], p_min, True)
        rows.append({"#": i, "Metric": ck.LABELS[m], "Value [95% interval]": val, "B0": bv, "Difference vs B0": dv,
                     "Target": tgt, "Verdict": v})
    pr_a, pr_b = gp.get((ss.OVERALL, "pr_auc"), np.nan), gbp.get((ss.OVERALL, "pr_auc"), np.nan)
    rows.append({"#": 6, "Metric": "PR-AUC (point estimate)", "Value [95% interval]": fmt(pr_a), "B0": fmt(pr_b),
                 "Difference vs B0": "–" if pd.isna(pr_a - pr_b) else f"{pr_a - pr_b:+.3f}", "Target": "≥ B0",
                 "Verdict": ck.GOOD if pr_a >= pr_b else ck.BAD})
    rows.append({"#": 7, "Metric": ck.LABELS["ece"], "Value [95% interval]": fmt(A.value["ece"], A.lo["ece"], A.hi["ece"]),
                 "B0": fmt(D.loc["ece"].b), "Difference vs B0": fdiff(D.loc["ece"]["diff"], D.loc["ece"].lo, D.loc["ece"].hi),
                 "Target": "≤ 0.05", "Verdict": ck.verdict(A.lo["ece"], A.hi["ece"], 0.05, False)})
    tf = g[(g.N == n_sel) & (g.metric == "type_f1") & (g.group != ss.OVERALL) & (g.n_obj >= min_n)]
    below = tf[tf.value < f_floor].group.tolist()
    rows.append({"#": 8, "Metric": "Objection-type accuracy, macro-F1 (point estimate)", "Value [95% interval]": fmt(gp.get((ss.OVERALL, "type_f1"))),
                 "B0": fmt(gbp.get((ss.OVERALL, "type_f1"))), "Difference vs B0": f"lowest type {tf.value.min():.3f}" if len(tf) else "–",
                 "Target": f"every type ≥ {f_floor:.2f}", "Verdict": ck.BAD if below else ck.GOOD})
    rows.append({"#": 9, "Metric": "p95 latency", "Value [95% interval]": "see the Model 1 notebook, section 9", "B0": "–",
                 "Difference vs B0": "–", "Target": "Within AmEx's real-time budget", "Verdict": ck.NA})
    st.subheader("1. Metrics")
    show_verdicts(pd.DataFrame(rows))
    if below:
        st.caption("Types below the accuracy floor: " + ", ".join(below))

    st.subheader("2. Checks before the results count")
    win = [rows[0]["Verdict"], rows[3]["Verdict"], rows[6]["Verdict"]]
    cal_rec = ck.boot(pa_cal, ["recall"], 0).value["recall"]
    test_rec = A.loc["recall"]
    thr_v = ck.GOOD if test_rec.lo <= cal_rec <= test_rec.hi else ck.BAD
    checks = [
        {"Check": "Win rule (proposal)", "Result": f"recall {rows[0]['Verdict'].lower()}, false alarms {rows[3]['Verdict'].lower()}, "
                                                   f"calibration {rows[6]['Verdict'].lower()}",
         "Good": "Recall ≥ B0, fewer false alarms, ECE ≤ 0.05", "Bad": "Any of the three not met", "Verdict": ck.combine(win)},
        {"Check": "SME label agreement (gate G1)", "Result": "two SMEs label the same customer turns", "Good": "Cohen's kappa ≥ 0.6",
         "Bad": "Below 0.6", "Verdict": ck.NA},
        {"Check": "Stability over time", "Result": "public data has no call dates", "Good": "Same direction in three time windows",
         "Bad": "Direction changes", "Verdict": ck.NA},
        {"Check": "Threshold on new calls", "Result": f"calibration recall {cal_rec:.3f}; test {fmt(test_rec.value, test_rec.lo, test_rec.hi)}",
         "Good": "Calibration recall inside the test interval", "Bad": "Outside the interval", "Verdict": thr_v},
        {"Check": "Noisy transcripts (E1.6, optional)", "Result": "needs clean and speech-to-text versions of the same calls",
         "Good": "No clear drop", "Bad": "Clear drop", "Verdict": ck.NA},
        {"Check": "Reliance on keywords", "Result": "needs the model re-scored with cue words removed (GPU run, not done yet)",
         "Good": "Small drop in recall", "Bad": "Large drop", "Verdict": "Not run"},
    ]
    if kw_ok:
        pk = per_call(path, KEYWORD_MODEL, 1, "test", float(choice(path, KEYWORD_MODEL, rule, budget, min_n)[1].threshold.iloc[0]))
        K = ck.boot_diff(pa, pk, ["recall", "fa_per_call"], N_BOOT)
        checks.append({"Check": "Keyword baseline", "Result": f"recall {K.loc['recall'].a:.3f} vs {K.loc['recall'].b:.3f} "
                       f"(difference {fdiff(K.loc['recall']['diff'], K.loc['recall'].lo, K.loc['recall'].hi)}); false alarms per call "
                       f"{K.loc['fa_per_call'].a:.3f} vs {K.loc['fa_per_call'].b:.3f}",
                       "Good": "Clearly better recall at the same false-alarm budget", "Bad": "Not clearly better",
                       "Verdict": ck.GOOD if K.loc["recall"].lo > 0 else ck.BAD})
    show_verdicts(pd.DataFrame(checks))
    st.caption(f"B0 here is {base} at N = {n_b}; on AmEx data it is the current trigger. "
               "Grey rows need data the public set does not have.")

# ------------------------------------------------------------------------------------------------ 3. business view
with tabs[2]:
    prev_pub = float(pa.n_obj.sum() / pa.turns.sum())
    c1, c2, c3 = st.columns(3)
    a_budget = c1.slider("Alert budget per call (agent load)", 0.5, 5.0, 2.0, 0.1)
    prev = c2.slider("AmEx objection rate (% of customer turns)", 0.5, 20.0, round(100 * prev_pub, 1), 0.1) / 100
    p_min_b = c3.slider("Minimum share of alerts worth acting on", 0.5, 1.0, 0.8, 0.01, key="pmin_b")
    adj = {"adj_precision": ck.adj_precision(prev)}
    A2 = ck.boot(pa, ["alerts_per_call", "recall_now"], N_BOOT, extra=adj)
    D2 = ck.boot_diff(pa, pb, ["recall_now"], N_BOOT)
    r = D2.loc["recall_now"]
    show_verdicts(pd.DataFrame([
        {"Measure": "Agent load: alerts per call", "Value [95% interval]": fmt(*A2.loc["alerts_per_call"]),
         "Good": f"≤ {a_budget:.1f} per call", "Bad": "Over budget", "Verdict": ck.verdict(*A2.loc["alerts_per_call"][["lo", "hi"]], a_budget, False)},
        {"Measure": f"Alerts worth acting on, at an objection rate of {100 * prev:.1f}%", "Value [95% interval]": fmt(*A2.loc["adj_precision"]),
         "Good": f"≥ {p_min_b:.2f}", "Bad": "Below the minimum", "Verdict": ck.verdict(*A2.loc["adj_precision"][["lo", "hi"]], p_min_b, True)},
        {"Measure": "Objections reaching Model 2 (alert on the objection turn itself)", "Value [95% interval]": fmt(*A2.loc["recall_now"]),
         "Good": f"≥ B0 ({r.b:.3f})", "Bad": "Below B0", "Verdict": ck.verdict(r.lo, r.hi, 0.0, True)},
    ]))
    st.caption(f"Alerts per call counts every alert, including repeats on the turns after an objection. The public data's objection "
               f"rate is {100 * prev_pub:.1f}% of customer turns; the share of alerts worth acting on is re-weighted to the rate set above "
               "(precision = rate × recall ÷ (rate × recall + (1 − rate) × false-alarm rate per turn)).")
    rates = np.linspace(0.005, 0.2, 40)
    fig = go.Figure()
    for name, pc in ((cand, pa), (f"{base} (B0)", pb)):
        s = {k: pc[k].sum() for k in pc.columns}
        fig.add_trace(go.Scatter(x=100 * rates, y=[float(ck.adj_precision(x)(s)) for x in rates], mode="lines", name=name))
    fig.add_hline(y=p_min_b, line_dash="dot", line_color="#7A8494")
    fig.add_vline(x=100 * prev, line_dash="dot", line_color="#7A8494")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="objection rate (% of customer turns)",
                      yaxis_title="share of alerts worth acting on", title="How the share of useful alerts depends on how often customers object")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------------------------ 4. keyword cues
with tabs[3]:
    if cues is None or cue_flags is None:
        st.info("No cue lists yet. Build them with `uv run python -m ica_eval.cues data/public_m1` (needs turns.parquet).")
    else:
        ct = cues[cues.level == "super-cluster"]
        st.subheader("Cue words per objection super-cluster")
        st.dataframe(ct.groupby("group").apply(lambda x: ", ".join(x.term), include_groups=False).rename("Cue words (strongest first)")
                     .rename_axis("Super-cluster").reset_index(), hide_index=True, width="stretch")
        with st.expander("Sub-cluster cue words"):
            cc = cues[cues.level == "sub-cluster"].merge(tax.astype(str), left_on="group", right_on="sub_cluster", how="left")
            st.dataframe(cc.groupby(["super_cluster", "group"]).apply(lambda x: ", ".join(x.term), include_groups=False)
                         .rename("Cue words").rename_axis(["Super-cluster", "Sub-cluster"]).reset_index(), hide_index=True, width="stretch")
        st.caption("Learned on the training calls: words and word pairs that set each type apart from all other customer turns "
                   "(log-odds z-score with an informative prior, z ≥ 3; at least 30% of the customer turns containing the term are "
                   "objections of that type). To be reviewed by SMEs on AmEx data.")

        st.subheader(f"Explicit vs implicit objections, cue words without an objection (N = {n_sel})")
        names = ["recall_explicit", "recall_implicit", "fa_rate_cue", "fa_rate_nocue"]
        Ka, Kb = ck.boot(pa, names, N_BOOT), ck.boot(pb, names, N_BOOT)
        st.dataframe(pd.DataFrame({"Measure": [ck.LABELS[n] for n in names],
                                   cand: [fmt(*Ka.loc[n]) for n in names], f"{base} (B0)": [fmt(*Kb.loc[n]) for n in names],
                                   "Turns": [int(pa.n_exp.sum()), int(pa.n_imp.sum()), int(pa.neg_cue.sum()), int(pa.neg_nocue.sum())]}),
                     hide_index=True, width="stretch")
        st.caption("Explicit: the objection turn contains a cue word of its own type. A model that mostly spots keywords shows a gap "
                   "between explicit and implicit recall, and a higher false-alarm rate on turns with a cue word but no objection.")

        pts_t = pts[(pts.candidate == cand) & (pts.N == n_sel) & (pts.part == "test")]
        det_ev = set(pts_t.event[(pts_t.p_cal >= t_sel) & pts_t.in_win])
        ob = pts_t[pts_t.obj == 1].merge(cue_flags[["call_id", "turn_idx", "explicit", "first_cue_bin"]], on=["call_id", "turn_idx"])
        ob["detected"] = ob.event.isin(det_ev)
        ob["cue"] = np.where(ob.explicit == True, "explicit", "implicit")  # noqa: E712
        by = ob.groupby(["true_type", "cue"], observed=True).agg(n=("detected", "size"), recall=("detected", "mean")).unstack("cue")
        by.columns = [f"{b} · {'objections' if a == 'n' else 'recall'}" for a, b in by.columns]
        for side in ("explicit", "implicit"):
            if f"{side} · recall" in by:
                by.loc[by[f"{side} · objections"].fillna(0) < min_n, f"{side} · recall"] = np.nan
        st.markdown("**By super-cluster** (point values; recall left blank below the minimum objections per cell)")
        st.dataframe(by.rename_axis("Super-cluster").reset_index().round(3), hide_index=True, width="stretch")

        st.subheader("Turns since the first cue")
        fc = ck.boot(pa, [f"recall_fc{k}" for k in range(len(FIRST_CUE_BINS))], N_BOOT)
        st.dataframe(pd.DataFrame({"First cue of the objection's type": FIRST_CUE_BINS,
                                   "Objections": [int(pa[f"n_fc{k}"].sum()) for k in range(len(FIRST_CUE_BINS))],
                                   f"Recall at N = {n_sel}": [fmt(*fc.loc[f"recall_fc{k}"]) for k in range(len(FIRST_CUE_BINS))]}),
                     hide_index=True, width="stretch")
        dist = (ob.groupby(["true_type", "first_cue_bin"], observed=True).size().unstack(fill_value=0)
                  .reindex(columns=FIRST_CUE_BINS, fill_value=0))
        st.dataframe(dist.rename_axis("Super-cluster").reset_index(), hide_index=True, width="stretch")
        st.caption("Counted in turns (agent and customer) before the objection; only earlier customer turns are searched. A type whose "
                   "cues usually appear further back than the chosen context length is flagged in step 3 of the context-length choice. "
                   "Public data: the call history is generated independently of the label, so earlier cues here are chance "
                   "matches; the real pattern needs AmEx transcripts.")

        st.subheader("Explicit vs implicit recall by context length")
        tr_rows = []
        for N in Ns:
            kk = ck.boot(per_call(path, cand, N, "test", float(thr.set_index("N").threshold[N])), ["recall_explicit", "recall_implicit"], 200)
            tr_rows += [dict(N=N, kind=lab, value=kk.value[m], lo=kk.lo[m], hi=kk.hi[m])
                        for m, lab in (("recall_explicit", "explicit"), ("recall_implicit", "implicit"))]
        tr = pd.DataFrame(tr_rows)
        fig = go.Figure()
        for lab, x in tr.groupby("kind"):
            fig.add_trace(go.Scatter(x=x.N, y=x.value, mode="lines+markers", name=lab,
                                     error_y=dict(type="data", symmetric=False, array=x.hi - x.value, arrayminus=x.value - x.lo, thickness=1)))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(type="log", tickvals=Ns, title="context length N (turns)"),
                          yaxis_title="recall")
        st.plotly_chart(fig, width="stretch")
        st.info("Not in this view yet: the masking check (re-score with each type's cue words removed). It needs the models "
                "re-run on edited text.")

# ------------------------------------------------------------------------------------------------ 5. early warning
with tabs[4]:
    st.markdown("**Open question for AmEx.** Cues often appear before the objection itself. Today an alert on such an early turn "
                "counts as a false alarm. If an \"objection likely\" alert is useful to agents, it needs its own label and its own "
                "measures, kept separate from the objection trigger's results. This tab shows what those measures would look like.")
    k = st.slider("Look-ahead: an alert counts as early if an objection follows within this many turns", 1, 12, 6)
    pe = per_call(path, cand, n_sel, "test", t_sel, k)
    E = ck.boot(pe, ["fa_per_call", "early_share", "early_chance", "obj_warned"], N_BOOT,
                extra={"lift": lambda s: ck.METRICS["early_share"](s) - ck.METRICS["early_chance"](s)})
    c = st.columns(4)
    c[0].metric("False alarms per call", fmt(E.value["fa_per_call"]), help=fmt(*E.loc["fa_per_call"]))
    c[1].metric("False alarms that come before an objection", fmt(E.value["early_share"]), help=fmt(*E.loc["early_share"]))
    c[2].metric("Same share by chance", fmt(E.value["early_chance"]), help="Share of turns outside objection windows that fall "
                "within the look-ahead before an objection")
    c[3].metric("Objections with an early alert", fmt(E.value["obj_warned"]), help=fmt(*E.loc["obj_warned"]))
    lift = ck.verdict(E.lo["lift"], E.hi["lift"], 0.0, True)
    st.caption(f"Early alerts above chance: {lift.lower()} (difference {fdiff(*E.loc['lift'])}). Lead time and precision of early "
               "alerts are only meaningful on real calls: in the public data the history before an objection carries no signal by construction.")
    d = pts[(pts.candidate == cand) & (pts.N == n_sel) & (pts.part == "test")].sort_values(["call_id", "turn_idx"])
    nxt = d.turn_idx.where(d.obj == 1).groupby(d.call_id).bfill()
    lead = (nxt - d.turn_idx)[(d.p_cal >= t_sel) & ~d.in_win & (d.obj == 0)].dropna()
    lead = lead[(lead >= 1) & (lead <= k)]
    if len(lead):
        fig = go.Figure(go.Histogram(x=lead, xbins=dict(start=0.5, end=k + 0.5, size=1), marker_color="#1F3A5F"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="lead time (turns before the objection)",
                          yaxis_title="early alerts", title="Lead time of early alerts")
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No early alerts at this setting.")

# ------------------------------------------------------------------------------------------------ 6. definitions
with tabs[5]:
    st.markdown("""
| Metric | Definition | Split by |
|---|---|---|
| Recall | Objections with an alert inside their detection window ÷ all objections | true type |
| Precision | Detected objections ÷ (detected objections + false alarms), attributed to the type of the detecting alert | predicted type |
| F1 | Harmonic mean of the two above | — |
| False alarms per call | Alerts outside every objection window ÷ calls | predicted type |
| False-alarm rate per turn | False alarms ÷ customer turns outside every objection window | predicted type |
| PR-AUC | Turns of this type vs turns without an objection, across all thresholds | true type |
| Calibration error (ECE) | 10 bins over all customer turns; the overall value has a 95% interval | predicted type |
| Type accuracy (F1) | Predicted vs true type on objection turns | — |
| Alerts per call | Every alert, including repeats on the turns after an objection ÷ calls | — |
| Alerts worth acting on | Precision re-weighted to the objection rate set in the business view | — |
| Objections reaching Model 2 | Objections with an alert on the objection turn itself ÷ all objections | — |
| Explicit / implicit | Objection turn contains a cue word of its own type, or not | true type |
| False-alarm rate with / without a cue word | Alerts ÷ turns outside objection windows that contain / do not contain any type's cue word | — |
| Early alert | Alert outside every objection window, with an objection later in the same call within the look-ahead | — |

The detection window is the objection turn and the next two customer turns. Thresholds are chosen on the calibration
calls and every result is reported on the test calls. "B0" is the model chosen under *Compare with*; on AmEx data it is the
current trigger. The keyword trigger scores the current customer turn only (summed z of matched cue words of the best
type, calibrated on the calibration calls).
""")
    st.caption("Development data: public dataset, near ceiling for the fine-tuned models, so the choices here only exercise "
               "the method. The real choices need AmEx data.")
