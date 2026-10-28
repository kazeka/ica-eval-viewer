"""ICA evaluation viewer, B0 page (synthetic SFT run): the sensitivity checks of the V0 failure-analysis doc on B0 (data
fit, consistency, knowledge gap, uncertain spots, buckets and DPO readiness), with samples to read, and the same checks
per SFT epoch to confirm the epoch early stopping kept.

Data: data/synth_b0/ (adapters/synth_b0_export.py; the same tables as nb/ica_b0_report.ipynb).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "synth_b0"
RULES = {"answers_objection": "answers the customer's turn", "no_invented_facts": "no invented facts",
         "placeholders": "placeholders used correctly", "length": "1–3 sentences, ≤ 70 words"}
BUCKETS = {"Data problem": "Fix SFT data → SFT v2", "Knowledge gap": "New SFT data for the slice",
           "Guessing": "Guardrail gate + targeted DPO pairs", "Consistency": "On-policy DPO", "Passes": "–"}
VERDICT_STYLE = {"Better": "background-color:#DCEBDD", "Worse": "background-color:#F6DADA",
                 "no clear change": "background-color:#F7EDD0"}
FMT = {"pct": "{:.0%}", "f3": "{:.3f}", "f0": "{:.0f}", "int": "{:.0f}"}
N_BOOT = 1000


@st.cache_data(show_spinner="Loading B0 results…")
def load():
    t = {f.stem: pd.read_parquet(f) for f in D.glob("*.parquet")}
    return t, json.loads((D / "meta.json").read_text(encoding="utf-8"))


def ci(x) -> tuple[float, float, float]:
    """Mean and bootstrap 95% interval over the rows (prompts or records) of x."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    b = x[np.random.default_rng(0).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def fmt(v, lo=np.nan, hi=np.nan, f="pct"):
    if v is None or pd.isna(v):
        return "–"
    s = FMT[f]
    return s.format(v) + (f" [{s.format(lo)}, {s.format(hi)}]" if pd.notna(lo) else "")


def by_kind(d: pd.DataFrame, cols: dict, f="pct") -> pd.DataFrame:
    """Columns of per-prompt values as 'mean [lo, hi]', overall and per record type."""
    groups = [("all", d)] + list(d.groupby("kind"))
    return pd.DataFrame({g: {"prompts": str(len(x)), **{lab: fmt(*ci(x[c]), f=f) for c, lab in cols.items()}}
                         for g, x in groups})


def table(df, **kw):
    n = len(getattr(df, "data", df))   # a Styler holds its frame in .data
    st.dataframe(df, width="stretch", height=min(36 * (n + 1) + 4, 720), **kw)


# ------------------------------------------------------------------------------------------------ sidebar
T, M = load()
K = M["k"]
SCORED = [s for s in M["sources"] if s in set(T["prompts"].source)]
with st.sidebar:
    st.markdown("### Filters")
    src = st.selectbox("Checkpoint", SCORED, index=0,
                       format_func=lambda s: s + (" (training run)" if s == "B0" else
                                                  " · early stopping's pick" if s == M.get("picked") else ""),
                       help="B0 = the adapter the training notebook kept, with its own saved results. Epochs = the "
                            "checkpoints at the end of each epoch, scored the same way afterwards.")
    kinds = sorted(T["prompts"].kind.unique())
    kind = st.selectbox("Record type", ["All"] + kinds)
    st.caption(f"Intervals: 95%, resampling prompts ({K} samples each) or held-out records.")

PR = T["prompts"][T["prompts"].source == src]
SR = T["samples"][T["samples"].source == src]
if kind != "All":
    PR, SR = PR[PR.kind == kind], SR[SR.kind == kind]
REC = T["records"].set_index("call_id")
st.title("Model 2 · B0 sensitivity checks (synthetic)")
st.caption(f"GPT-OSS-20B 4-bit + LoRA, SFT on synthetic GLM records ({M['n_sft_train']} training records, "
           f"{M['n_heldout']} held out) · **{src}** · {PR.call_id.nunique()} held-out sensitivity prompts × {K} samples"
           + ("" if kind == "All" else f" · {kind}"))
st.caption("Synthetic data: invented customers and calls. The checks follow the V0 failure-analysis doc; every rate has a "
           "95% bootstrap interval and is also shown per record type. These numbers are the baseline later models are "
           "compared with.")

tabs = st.tabs(["Summary", "1 Data fit", "2 Consistency", "3 Knowledge gap", "4 Uncertain spots", "Buckets and DPO",
                "Per epoch", "Definitions"])

# ------------------------------------------------------------------------------------------------ summary
with tabs[0]:
    G = T["greedy"][T["greedy"].source == src]
    if kind != "All":
        G = G[G.call_id.map(REC.kind) == kind]
    c = st.columns(5)
    for col, (lab, x) in zip(c, [("Rules pass (greedy)", G.rule_passed), (f"Avg@{K}", PR.avg),
                                 (f"Pass@{K}", PR.pass_at_k), ("Mixed prompts", PR.mixed),
                                 ("All fail (gap)", PR.all_fail)]):
        v, lo, hi = ci(x)
        col.metric(lab, fmt(v), help=f"95% interval {fmt(lo)} – {fmt(hi)}")
    st.subheader("Baseline numbers by record type")
    table(by_kind(PR, {"avg": f"Avg@{K}", "pass_at_k": f"Pass@{K}", "mixed": "mixed prompts",
                       "all_fail": "all fail (knowledge gap)", **{r: f"rule: {lab}" for r, lab in RULES.items()}}))
    st.subheader("Greedy reply on every held-out record vs the base model")
    gb = T["greedy"][T["greedy"].source.isin(["base", src])]
    if kind != "All":
        gb = gb[gb.call_id.map(REC.kind) == kind]
    table(pd.DataFrame({s: {"records": str(len(g)), "all rule checks pass": fmt(*ci(g.rule_passed)),
                            "repeats an earlier agent turn": fmt(*ci(g.has_fail_repeats_pitch)),
                            "too long to say aloud": fmt(*ci(g.has_rule_length_ok)),
                            "ROUGE-L vs agent reply": fmt(*ci(g.rougeL_agent), f="f3"),
                            "words per reply": fmt(*ci(g.words), f="f0")} for s, g in gb.groupby("source", sort=False)}))
    st.caption("The rules in this table are the training notebook's failure checks on one greedy reply per record; "
               "the four binary rules above are the doc's, checked on every sample.")

# ------------------------------------------------------------------------------------------------ data fit
with tabs[1]:
    NG = T["nll_groups"][T["nll_groups"].source == src]
    if NG.empty:
        st.info(f"No target NLL for {src}.")
    else:
        st.markdown("Negative log-likelihood per token of the reference reply. Training targets were seen in SFT; "
                    "held-out ones were not. A large gap means memorising; a high held-out NLL in one cluster points at "
                    "data the model finds unlike its training.")
        k_ = NG[NG.by == "kind"]
        table(k_.assign(v=[fmt(r.nll, r.lo, r.hi, "f3") for r in k_.itertuples()])
              .pivot(index="group", columns="targets", values="v"))
        cl = NG[NG.by == "cluster"]
        wide = cl.pivot(index="group", columns="targets", values="nll")
        wide["n held out"] = cl[cl.targets == "held-out targets"].set_index("group").n
        if {"held-out targets", "training targets"} <= set(wide):
            wide["gap"] = wide["held-out targets"] - wide["training targets"]
        st.subheader("By cluster, highest held-out NLL first")
        table(wide.sort_values("held-out targets", ascending=False).round(3))
        st.caption("Clusters have few held-out records each: read them as pointers, not results.")

# ------------------------------------------------------------------------------------------------ consistency
with tabs[2]:
    st.markdown(f"{K} samples per prompt at temperature 1. **Avg@{K}** = share of samples passing all four rules; "
                f"**Pass@{K}** = at least one passes; **mixed** = some pass, some fail (DPO has a signal there).")
    dist = pd.crosstab(PR.kind, (PR.avg * K).round().astype(int).rename(f"samples passing (of {K})"), margins=True,
                       margins_name="all")
    fig = go.Figure(go.Bar(x=dist.columns[:-1].astype(str), y=dist.loc["all"].iloc[:-1], marker_color="#4A6A92"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=f"samples passing (of {K})",
                      yaxis_title="prompts")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Mass at {K} = solved, at 0 = knowledge gap, in between = consistency problem.")
    mixed = PR[PR.mixed].call_id
    fails = SR[SR.call_id.isin(mixed) & ~SR.pass_all]
    if len(fails):
        st.subheader("Which rules the failing samples break, on mixed prompts")
        broken = pd.DataFrame({RULES[r]: ~fails[r].astype(bool) for r in RULES})
        table(broken.groupby(fails.kind.values).mean().T.assign(all=broken.mean()).style.format("{:.0%}"))
    st.subheader("Read a prompt's samples")
    order = PR.sort_values("avg")
    pick = st.selectbox("Prompt (fewest passing samples first)", list(order.call_id),
                        format_func=lambda c: f"{c} · {int(round(order.set_index('call_id').avg[c] * K))}/{K} pass · "
                                              f"{order.set_index('call_id').kind[c]}")
    if pick:
        r = REC.loc[pick]
        st.markdown(f"**{r.candidate_id}** · {r.kind} · cluster {r.cluster}  \n> **Customer (last turn):** "
                    f"{r.last_customer_turn}\n\n**Agent (reference):** {r.verbatim}")
        for s in SR[SR.call_id == pick].sort_values("sample").itertuples():
            broke = [RULES[k] for k in RULES if not getattr(s, k)]
            st.markdown(f"{'✅' if s.pass_all else '❌'} **sample {s.sample}** "
                        f"{'' if s.pass_all else '(breaks: ' + ', '.join(broke) + ')'}  \n{s.reply}")

# ------------------------------------------------------------------------------------------------ knowledge gap
with tabs[3]:
    gap = PR[PR.all_fail]
    v, lo, hi = ci(PR.all_fail)
    st.markdown(f"Prompts where all {K} samples fail: **{len(gap)} of {len(PR)}** ({fmt(v, lo, hi)}). No sample to "
                "prefer, so DPO has no signal there: these need new SFT data for the slice.")
    for g in gap.itertuples():
        r = REC.loc[g.call_id]
        s = SR[SR.call_id == g.call_id]
        rates = ", ".join(f"{RULES[k]} {s[k].mean():.0%}" for k in RULES)
        with st.container(border=True):
            st.markdown(f"**{r.candidate_id}** · {r.kind} · cluster {r.cluster} · rule pass rates: {rates}  \n"
                        f"> **Customer:** {r.last_customer_turn[-400:]}\n\n**Agent (reference):** {r.verbatim}  \n"
                        f"**Sample 0:** {s.sort_values('sample').reply.iloc[0]}")

# ------------------------------------------------------------------------------------------------ uncertain spots
with tabs[4]:
    C = T["confidence"][T["confidence"].source == src]
    st.markdown("Token probabilities of the samples. Low confidence = below the 10% quantile of all reply tokens. An "
                "invented fact on a low-confidence token is a guess (a guardrail can catch it); on a confident token "
                "the model believes it (needs DPO or data).")
    if len(C):
        table(C.assign(**{"low-confidence share [95%]": [fmt(r.low_conf_share, r.lo, r.hi) for r in C.itertuples()]})
              [["span", "tokens", "prompts", "mean_prob", "mean_entropy", "low-confidence share [95%]"]].round(3),
              hide_index=True)
    F = T["facts"][T["facts"].source == src]
    if kind != "All":
        F = F[F.call_id.map(REC.kind) == kind]
    st.subheader(f"Invented facts ({len(F)})")
    if len(F):
        st.caption(f"{F.low_conf.mean():.0%} of them have a low-confidence token.")
        table(F.sort_values("min_prob")[["call_id", "sample", "fact", "min_prob", "low_conf", "context"]].round(3),
              hide_index=True)

# ------------------------------------------------------------------------------------------------ buckets
with tabs[5]:
    st.markdown("Each prompt goes to the first bucket that applies: **data problem** (reference reply's NLL in the top "
                "quarter and some samples fail) → **knowledge gap** (all fail) → **guessing** (an invented fact on a "
                "low-confidence token) → **consistency** (mixed) → **passes**.")
    table(pd.DataFrame({g: {"prompts": str(len(x)), **{f"{b} → {nxt}": fmt(*ci(x.bucket == b)) for b, nxt in BUCKETS.items()}}
                        for g, x in [("all", PR)] + list(PR.groupby("kind"))}))
    DP = T["dpo"][(T["dpo"].source == src) & T["dpo"].call_id.isin(PR.call_id)]
    pool = PR.bucket.isin(["Consistency", "Guessing"]).sum()
    st.subheader("DPO pairs from B0's own samples")
    st.markdown(f"DPO-ready prompts (consistency + guessing): **{pool} of {len(PR)}**; **{len(DP)}** length-matched pairs "
                "(chosen passes all rules, rejected fails one, word counts within 1.3×).")
    if len(DP):
        table(DP.failed_rules.str.split(", ").explode().map(RULES).value_counts().rename("rejected reply breaks").to_frame())
        with st.expander("Read the pairs"):
            for p in DP.itertuples():
                st.markdown(f"**{p.call_id}** · rejected breaks: {p.failed_rules}  \n✅ {p.chosen}  \n❌ {p.rejected}")

# ------------------------------------------------------------------------------------------------ per epoch
with tabs[6]:
    if "matrix" not in T or not M.get("picked"):
        st.info("No per-epoch results yet: train with `epoch_matrix` (training notebook section 7b), or run "
                "`nb/epoch_sweep.py`, then re-run `adapters/synth_b0_export.py`.")
    else:
        P = M["picked"]
        L = T["loss"]
        half = max(1, M["per_epoch"] // 2)
        tl = L[L.series == "training loss"].set_index("epoch").loss.rolling(half, min_periods=1).mean()
        vl = L[L.series == "validation loss"]
        fig = go.Figure([go.Scatter(x=tl.index, y=tl.values, name="training loss (rolling half epoch)"),
                         go.Scatter(x=vl.epoch, y=vl.loss, name="validation loss", mode="lines+markers")])
        fig.add_vline(x=M["best_step"] / M["per_epoch"], line_dash="dot", annotation_text="early stopping's pick")
        fig.update_layout(xaxis_title="epoch", yaxis_title="loss (nats per token)", height=340,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"Early stopping (patience {M['run']['es_patience']} half epochs) stops at epoch "
                    f"{M['stop_step'] / M['per_epoch']:g} and keeps **{P}**. Scored: {', '.join(M['epochs'])}.")
        MX = T["matrix"]
        MX = MX.assign(v=[fmt(r.value, r.lo, r.hi, r.fmt) for r in MX.itertuples()])
        cols = list(dict.fromkeys(MX.source))
        rows = list(dict.fromkeys(zip(MX.section, MX.check)))
        wide = MX.pivot_table(index=["section", "check"], columns="source", values="v", aggfunc="first").reindex(rows)
        wide = wide[cols].fillna("").rename(columns={"base": "epoch 0 (base)", P: P + " ◀ picked"})
        st.subheader("Result matrix")
        table(wide)
        DF = T["diff"]
        if len(DF):
            st.subheader(f"Paired differences from {P} (same records and prompts)")
            d = DF.assign(difference=[f"{r.diff:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for r in DF.itertuples()])
            show = d.pivot(index="check", columns="source", values=["difference", "verdict"])
            show.columns = [f"{s} · {c}" for c, s in show.columns]
            vcols = [c for c in show.columns if c.endswith("verdict")]
            table(show.style.map(lambda v: VERDICT_STYLE.get(v, ""), subset=vcols))
            later = [n for n in M["epochs"] if n != P and float(n.split()[1]) > float(P.split()[1])]
            better = DF[(DF.verdict == "Better") & DF.source.isin(later)]
            worse = DF[(DF.verdict == "Worse") & DF.source.isin(later)]
            st.markdown("\n".join([
                "- Later epochs clearly better on: " + ("; ".join(f"{r.check} ({r.source})" for r in better.itertuples())
                                                       or "nothing") + ".",
                "- Later epochs clearly worse on: " + ("; ".join(f"{r.check} ({r.source})" for r in worse.itertuples())
                                                      or "nothing") + ".",
                "- Held-out minus training NLL: " + ", ".join(f"{n} {v:.3f}" for n, v in M["nll_gap"].items()) + ".",
                "- **Early stopping " + ("is confirmed" if better.empty else "is not confirmed: a later epoch wins on "
                                         "some checks") + "** by the rule and consistency checks."]))
        else:
            st.info("Only one epoch scored so far: nothing to compare.")
        if "picked_same_greedy_as_B0" in M:
            st.caption(f"{P} retrained with B0's settings and seeds: {M['picked_same_greedy_as_B0']:.0%} of its greedy "
                       "replies are word for word B0's (GPU kernels are not bit-exact, so long replies drift).")

# ------------------------------------------------------------------------------------------------ definitions
with tabs[7]:
    st.markdown(f"""
**Checks** (V0 failure analysis — sensitivity checks)
1. **Data fit:** NLL per token of the reference reply, training vs held-out targets.
2. **Consistency:** {K} samples per prompt; Avg@{K}, Pass@{K}, mixed prompts.
3. **Knowledge gap:** all {K} samples fail.
4. **Uncertain spots:** token confidence where a reply invents a fact.

**Binary rules** on every sample: {'; '.join(RULES.values())}.

**Buckets → next step:** {'; '.join(f'{b} → {n}' for b, n in BUCKETS.items())}.

**Per epoch:** the checkpoint at the end of each epoch, scored like B0. Paired differences resample the same records
(greedy) or prompts (samples); Better / Worse = the whole 95% interval on one side of zero.
""")
