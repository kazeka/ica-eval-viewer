"""ICA evaluation viewer, Model 2 page (reply recommender, sandbox runs): LLM-free metrics from the Model 2 LLM-free
evaluation doc, the sandbox's rule checks and failure labels, by objection super- and sub-cluster, with replies to read.

Data: data/sandbox_m2/ (adapters/sandbox_m2_export.py, then `python -m ica_eval.m2 data/sandbox_m2`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ica_eval import checks as ck  # noqa: E402
from ica_eval import m2  # noqa: E402

D = ROOT / "data" / "sandbox_m2"
AGENT = "Agent (historical reply)"
VERDICT_STYLE = {ck.GOOD: "background-color:#DCEBDD", ck.BAD: "background-color:#F6DADA",
                 ck.INCONCLUSIVE: "background-color:#F7EDD0", "Not run": "color:#7A8494", "Not available": "color:#7A8494"}
BLUES = [[0, "#F4F6F9"], [0.5, "#9DB4CF"], [1, "#1F3A5F"]]
FAIL_LABELS = {"fail_does_not_answer": "Does not answer the objection", "fail_wrong_objection": "Answers a different objection",
               "fail_invented_offer": "Invents an offer or number", "fail_too_long": "Too long to say aloud",
               "fail_no_empathy": "Does not acknowledge the concern", "fail_not_personal": "Does not use the customer's context",
               "fail_no_next_step": "No clear next step", "fail_pushy": "Pushy or high pressure",
               "fail_repeats_pitch": "Repeats the earlier pitch", "fail_bad_placeholder": "Breaks or invents a placeholder",
               "fail_keyword_list": "Keyword list that cannot be said aloud"}
N_BOOT = 500


@st.cache_data(show_spinner="Loading Model 2 replies…")
def load():
    ctx = pd.read_parquet(D / "m2_contexts.parquet")
    refs = pd.read_parquet(D / "m2_references.parquet")
    f = D / "m2_reply_metrics.parquet"
    rep = pd.read_parquet(f) if f.exists() else m2.reply_metrics(pd.read_parquet(D / "m2_replies.parquet"), ctx, refs)
    rep = m2.prepare(rep)
    rep["run"], rep["policy"] = rep.run.astype(str), rep.policy.astype(str)
    runs = json.loads((D / "m2_runs.json").read_text(encoding="utf-8"))
    return rep, ctx, refs, runs


@st.cache_data(show_spinner="Resampling contexts…")
def compare_cached(run: str, a: str, b: str | None, ids: tuple):
    rep = load()[0]
    r = rep[rep.run == run]
    cat = [m for m in m2.CATALOG + m2.SANDBOX if m.id in ids or not ids]
    return m2.compare(r[r.policy == a], r[r.policy == b] if b else None, cat, N_BOOT)


def fmt(v, lo=np.nan, hi=np.nan, d=3):
    if v is None or pd.isna(v):
        return "–"
    return f"{v:.{d}f}" + (f" [{lo:.{d}f}, {hi:.{d}f}]" if pd.notna(lo) else "")


def fdiff(v, lo, hi, d=3):
    return "–" if pd.isna(v) else f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def show_verdicts(df, col="Verdict"):
    st.dataframe(df.style.map(lambda v: VERDICT_STYLE.get(v, ""), subset=[col]), hide_index=True, width="stretch",
                 height=36 * (len(df) + 1) + 4)


def heat(t: pd.DataFrame, policies: list[str], groups: list[str], min_n: int, lower_better: bool, title: str):
    z, txt = [], []
    for p in policies:
        zr, tr = [], []
        for g in groups:
            c = t[(t.policy == p) & (t.group == g)]
            n = int(c.n.iloc[0]) if len(c) else 0
            if n < min_n:
                zr.append(np.nan); tr.append(f"n={n}<br>too few"); continue
            v = float(c.value.iloc[0]); zr.append(v); tr.append(f"<b>{v:.3f}</b><br>n={n}")
        z.append(zr); txt.append(tr)
    zz = np.array(z, dtype=float)
    zmin, zmax = (np.nanmin(zz), np.nanmax(zz)) if np.isfinite(zz).any() else (0, 1)
    if zmax - zmin < 0.02:
        zmin = zmax - 0.02
    wrap = [g if len(g) <= 18 else g[:18] + "<br>" + g[18:40] for g in groups]
    fig = go.Figure(go.Heatmap(z=z, x=wrap, y=policies, text=txt, texttemplate="%{text}", zmin=zmin, zmax=zmax,
                               colorscale=BLUES[::-1] if lower_better else BLUES, xgap=2, ygap=2, hoverongaps=False,
                               colorbar=dict(title=title, thickness=12)))
    fig.update_layout(height=120 + 55 * len(policies), margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#E6E9EE",
                      yaxis=dict(autorange="reversed"), xaxis=dict(side="top"), font=dict(size=12))
    return fig


# ------------------------------------------------------------------------------------------------ sidebar
rep, ctx, refs, runs = load()
with st.sidebar:
    st.markdown("### Filters")
    run = st.selectbox("Run", sorted(rep.run.unique(), key=lambda r: (r != "full", r)))
    pols = [p for p in rep[rep.run == run].policy.unique()]
    trained = [p for p in pols if p != AGENT]
    policy = st.selectbox("Policy", trained, index=len(trained) - 1 if trained else 0)
    others = [p for p in pols if p != policy]
    base = st.selectbox("Compare with", others, index=others.index("B0_sft") if "B0_sft" in others else 0,
                        help="The comparison model. B0 = the SFT model (the starting point every later model must beat); "
                             "the agent's historical reply is the human reference.")
    min_n = st.slider("Minimum contexts per cell", 5, 100, 30, 5)
    st.caption("Intervals: 95%, resampling evaluation contexts. Comparisons are paired on the same contexts.")

R = rep[rep.run == run]
man = (runs.get(run) or {}).get("manifest") or {}
st.title("Model 2 · reply recommender (sandbox)")
_stages = [p for p in pols if p != AGENT]
_status = "complete" if man else f"in progress: {', '.join(_stages)} done"
st.caption(f"Run **{run}** ({_status}) · {man.get('base_model', 'GPT-OSS-20B')} 4-bit + LoRA · "
           f"{R.call_id.nunique()} held-out contexts ({(man.get('eval') or {}).get('split', 'last time window')}) · "
           f"**{policy}** compared with **{base}**")
st.caption("Synthetic data: an invented product and customers. The `true_*` columns and failure labels come from the "
           "sandbox's hidden truth and detectors; on AmEx data, SME reviews and validated judges take their place.")

tabs = st.tabs(["Scorecard", "Metrics vs B0", "By objection type", "Failures", "Replies", "Training", "Definitions"])

# ------------------------------------------------------------------------------------------------ scorecard
with tabs[0]:
    head = [("F1–F4", "All rule checks pass"), ("F3", "No invented offers (known patterns)"),
            ("F8", "Off target (does not answer, or answers another objection)"), ("F14", "Words per reply"),
            ("F12", "Distinct word pairs (distinct-2)"), ("R3", "ROUGE-L F1 vs agent's reply"),
            ("S", "Reward v2 (what training optimised)"), ("S", "True quality (sandbox only)"),
            ("S", "True P(convert) (sandbox only)")]
    rows = []
    for p in pols:
        c = compare_cached(run, p, None, ())
        c = c.set_index(["id", "metric"])
        rows.append({"Policy": p, **{lab: fmt(*c.loc[(i, lab), ["value", "lo", "hi"]]) if (i, lab) in c.index else "–" for i, lab in head}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("If reward v2 rises while the true columns do not, the policy is gaming the reward. "
               "The agent's historical reply is the reference point that the SFT targets were drawn from.")
    if (runs.get(run) or {}).get("scorecard"):
        with st.expander("Scorecard written by the training run (its own metric code)"):
            st.dataframe(pd.DataFrame(runs[run]["scorecard"]), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------------ metrics vs B0
with tabs[1]:
    C = compare_cached(run, policy, base, ())
    out = pd.DataFrame({"#": C.id, "Metric": C.metric, "Group": C.group, "Role": C.role,
                        "Value [95% interval]": [fmt(v, l, h) for v, l, h in zip(C.value, C.lo, C.hi)],
                        "Comparison": [fmt(v) for v in C.b], "Difference": [fdiff(v, l, h) for v, l, h in zip(C["diff"], C.dlo, C.dhi)],
                        "Target": [m2.target_text(m) for m in m2.CATALOG + m2.SANDBOX], "Verdict": C.verdict})
    if not st.toggle("Show metric group and role", value=False):
        out = out.drop(columns=["Group", "Role"])
    show_verdicts(out)
    st.caption(f"Comparison model: {base}. Good / Bad / Inconclusive follow the reading rule: the whole 95% interval of the "
               "difference on the right side of the target, on the wrong side, or crossing it. Grey rows need an approved "
               "embedding, entailment or question-answering model (R6–R8, F7, F10–F11) or SME term lists (F15–F16).")

# ------------------------------------------------------------------------------------------------ by objection type
with tabs[2]:
    usable = [m for m in m2.CATALOG + m2.SANDBOX if m2.available(m, R)]
    labels = [f"{m.id} · {m.label}" for m in usable]
    pick = st.selectbox("Metric", labels, index=next((i for i, m in enumerate(usable) if m.col == "rule_passed"), 0))
    met = usable[labels.index(pick)]
    show = [p for p in pols]
    t = m2.by_group(R, ctx, "objection_type", met, show)
    st.subheader("By objection super-cluster")
    st.plotly_chart(heat(t, show, sorted(ctx.objection_type.unique()), min_n, met.higher is False, met.id), width="stretch")
    st.subheader("Sub-clusters (objection phrasings) within a super-cluster")
    sup = st.selectbox("Super-cluster", sorted(ctx.objection_type.unique()))
    subs = sorted(ctx[ctx.objection_type == sup].sub_cluster.unique())
    ts = m2.by_group(R, ctx[ctx.objection_type == sup], "sub_cluster", met, show)
    biggest = int(ts.n.max()) if len(ts) else 0
    if biggest < min_n:
        st.info(f"Every phrasing in {sup} has fewer than {min_n} evaluation contexts (largest: {biggest}), so no cell is "
                f"shown. Lower *Minimum contexts per cell* in the sidebar to {max(5, biggest // 5 * 5)} to see them, with wide "
                "intervals, or score a larger frozen set.")
    else:
        st.plotly_chart(heat(ts, show, subs, min_n, met.higher is False, met.id), width="stretch")
    n_eval = R.call_id.nunique()
    st.caption(f"{n_eval} evaluation contexts in this run, so sub-cluster cells are small; lower the minimum to see them, "
               "or score a larger frozen set with the Model 2 runner.")

# ------------------------------------------------------------------------------------------------ failures
with tabs[3]:
    fcols = [c for c in FAIL_LABELS if c in R]
    if not fcols:
        st.info("This source has no failure labels.")
    else:
        ft = R.groupby("policy")[fcols].mean().reindex(pols).rename(columns=FAIL_LABELS).T
        st.subheader("Failure rates by policy (sandbox detectors)")
        st.dataframe(ft.style.format("{:.1%}"), width="stretch")
        st.subheader(f"{policy}: failure rates by objection type")
        x = R[R.policy == policy].merge(ctx[["call_id", "objection_type"]], on="call_id")
        st.dataframe(x.groupby("objection_type")[fcols].mean().rename(columns=FAIL_LABELS).T.style.format("{:.0%}"),
                     width="stretch")
        md = (runs.get(run) or {}).get("failure_md")
        if md:
            with st.expander("Failure analysis of B0 written by the training run"):
                st.markdown(md)

# ------------------------------------------------------------------------------------------------ replies
with tabs[4]:
    c1, c2, c3 = st.columns(3)
    typ = c1.selectbox("Objection type", ["All"] + sorted(ctx.objection_type.unique()))
    flt = c2.selectbox("Show contexts where the policy's reply", ["(any)"] + [FAIL_LABELS[c] for c in fcols] + ["fails a rule check"])
    nshow = c3.slider("Contexts to show", 3, 30, 8)
    P = R[R.policy == policy].set_index("call_id")
    ids = ctx if typ == "All" else ctx[ctx.objection_type == typ]
    ids = [i for i in ids.call_id if i in P.index]
    if flt == "fails a rule check" and "rule_passed" in P:
        ids = [i for i in ids if not P.rule_passed[i]]
    elif flt != "(any)":
        col = {v: k for k, v in FAIL_LABELS.items()}[flt]
        ids = [i for i in ids if P[col][i]]
    st.caption(f"{len(ids)} matching contexts")
    C_ = ctx.set_index("call_id")
    for cid in ids[:nshow]:
        with st.container(border=True):
            st.markdown(f"**{cid}** · {C_.sub_cluster[cid]} · {C_.segment[cid]}, {C_.channel[cid]}  \n"
                        f"*Customer:* {C_.objection_text[cid]}")
            shown = list(dict.fromkeys([AGENT, base, policy]))
            cols = st.columns(len(shown))
            for col, p in zip(cols, shown):
                if p in pols:
                    r = R[(R.policy == p) & (R.call_id == cid)].iloc[0]
                    tags = [FAIL_LABELS[c] for c in fcols if r[c]]
                    rule = "" if "rule_passed" not in r or r.rule_passed else " · ⚠ rule check failed"
                    role = " (agent)" if p == AGENT else " (selected)" if p == policy else " (comparison)"
                    col.markdown(f"**{p}**{role if p != AGENT else ''}{rule}  \n{r.reply}  \n<span style='color:#7A8494;font-size:0.85em'>"
                                 f"{'; '.join(tags) or 'no failure flagged'}"
                                 + (f" · true quality {r.true_quality:+.2f}" if "true_quality" in r else "") + "</span>",
                                 unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------ training
with tabs[5]:
    info = runs.get(run) or {}
    if info.get("steps"):
        st.subheader("Stages")
        last = {}
        for x in info["steps"]:
            last[x["step"]] = x                      # resumed runs repeat steps: keep the latest
        nice = {"sft_B0": "B0: supervised fine-tuning", "failure_analysis": "B0 failure analysis",
                "reward_check": "Reward check on B0's replies", "grpo_pilot": "GRPO pilot"}
        rows = []
        for k, x in last.items():
            if k in ("data_and_reward", "load_model", "done"):
                continue
            label = nice.get(k) or (f"DPO round {k[-1]}" if k.startswith("dpo_round") else
                                    f"Evaluate {k[5:].replace('_', ' ')}" if k.startswith("eval_") else k)
            det = []
            if x.get("pairs") is not None:
                det.append(f"{x['pairs']} preference pairs, mean margin {x.get('margin', 0):.2f}")
            if x.get("dpo_target"):
                det.append("DPO targets: " + ", ".join(str(t).replace('_', ' ') for t in x["dpo_target"]))
            if x.get("gen_minutes") is not None:
                det.append(f"300 replies in {x['gen_minutes']:.1f} min")
            if x.get("adapter"):
                det.append(f"saved {x['adapter']}")
            rows.append({"Stage": label, "Finished at (min)": x.get("minutes"), "Details": "; ".join(det)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander("Raw step log"):
            st.dataframe(pd.DataFrame(info["steps"]), hide_index=True, width="stretch")
    logs = info.get("trainlogs") or {}
    if logs:
        st.subheader("Training curves")
        name = st.selectbox("Stage", list(logs))
        lg = pd.DataFrame([r for r in logs[name] if "loss" in r or "reward" in r])
        ys = [c for c in ("loss", "reward", "rewards/chosen", "rewards/rejected", "rewards/margins", "kl", "completion_length") if c in lg]
        if len(lg) and ys:
            fig = go.Figure([go.Scatter(x=lg.get("step", lg.index), y=lg[y], mode="lines+markers", name=y) for y in ys])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="step")
            st.plotly_chart(fig, width="stretch")
    if info.get("reward_tricks"):
        st.subheader("Reward check on B0's replies: does a trick raise reward v2?")
        rt = pd.DataFrame(info["reward_tricks"]).rename(columns={"trick": "Trick", "reward went up": "Reward went up",
                                                                 "mean change": "Mean change in reward"})
        rt["Trick"] = rt.Trick.str.replace("_", " ")
        st.dataframe(rt.style.format({"Reward went up": "{:.0%}", "Mean change in reward": "{:+.2f}"}),
                     hide_index=True, width="stretch")
        st.caption("Good: tricks that do not help the customer should not raise the reward (0% is the target).")
    if man:
        with st.expander("Run manifest"):
            st.json(man)

# ------------------------------------------------------------------------------------------------ definitions
with tabs[6]:
    st.markdown("""
| Group | What | How it is computed here |
|---|---|---|
| References | Agent's reply; converted replies | The agent's actual reply at this moment; up to 8 polished replies to the same objection phrasing from converted calls (the SFT targets) |
| R1 BLEU | Word-sequence overlap | Sentence BLEU-4, add-one smoothing, closest-reference brevity penalty |
| R2 / R3 ROUGE | Word, word-pair and longest-sequence overlap | F1 against the reference (best of 8 for converted replies) |
| R4 JS divergence | Difference of word-pair distributions | Base 2; 0 = same distribution, 1 = disjoint |
| R5 / F13 near-copy | Repeats a training reply almost verbatim | Similarity ratio ≥ 0.9 with any converted reply to the same phrasing |
| F1–F4 rules | Format, new numbers, invented offers, placeholders | The sandbox's rule checks (code), as in the compliance rules of the reward |
| F5 conduct, F8 off target | Pushy, repeats the pitch; does not answer / answers another objection | Sandbox detectors (on AmEx data: an approved judge checked by SMEs) |
| F9 ROUGE-C | Overlap with the objection turn | ROUGE-1 F1 against the customer's objection |
| F12 distinct-2 | Variety of wording | Distinct word pairs ÷ all word pairs, over the policy's replies |
| F13 stock openings | One opening fits all | Replies whose first 3 words are shared by ≥ 5% of the policy's replies |
| F14 length | Words per reply | Drift check against B0 |
| Sandbox | Reward v2, true quality, true P(convert) | Reward = outcome models + rule penalty; truth from the sandbox oracle (not available on real data) |
""")
