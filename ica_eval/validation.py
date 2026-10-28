"""T1–T8 pipeline: is a scoring method trustworthy enough to count, and to go into the reward?

A scoring method ("scorer": lookalike, outcome model, rules, later an LLM judge) gives a number for a reply in a context.
The pipeline runs the checks in stages from `bars.yaml`; a scorer moves to the next stage only if every check in the
current stage is Good:

    stage 1, automatic (no AmEx input): T5 sensitivity, T6 invariance, T1 stability, T2 predicts later sales,
             T3 matched pairs, T8 uplift is real
    stage 2, SME: T4 agreement with SME choices
    report only: T7 agreement with the hidden true quality (sandbox data only)

Every check is computed on tables, so the same code runs on sandbox, public and AmEx data; an adapter builds the
inputs (see adapters/sandbox_validation.py). T5, T6, T4 and T8 use replies from trained models (B0 first, then each
checkpoint); T1–T3 use historical agent replies, the only replies with outcomes (fit on an earlier slice, test on a
later one). T2, T3, T4 and the T1/T5/T6 statistics follow Merxhan's `ica_sandbox.evalval` so the numbers can be checked
against it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

GOOD, BAD, INCONCLUSIVE = "Good", "Bad", "Inconclusive"
REPORT, NOT_RUN, NOT_APPLICABLE = "Report", "Not run", "Not applicable"
BARS_FILE = Path(__file__).resolve().parents[1] / "bars.yaml"


# ------------------------------------------------------------------------------------------------ inputs
@dataclass
class Scorer:
    """`score(rows, replies)` -> one value per reply (NaN = cannot score). `refit(hist)` -> the same scorer fitted on
    another past period (T1); None for fixed scorers such as rules. `to_prob` maps scores to a conversion-like scale
    for uplift (T8). `catches`: kinds of damage the scorer is meant to detect (T5); None = all of them."""
    name: str
    score: Callable[[pd.DataFrame, list[str]], np.ndarray]
    refit: Callable[[pd.DataFrame], Scorer] | None = None
    to_prob: Callable[[np.ndarray], np.ndarray] | None = None
    catches: set[str] | None = None
    binary: bool = False
    role: str = "ranker"   # "ranker" (may pick pairs or drive training) or "gate" (pass/fail check, e.g. rules):
    #                        a gate is not asked to predict sales or agree with SME preferences (T2, T3, T4, T8 do not
    #                        apply); its own check is precision and recall on an SME-labelled sample (spec E-07)


@dataclass
class Probes:
    """T5/T6: the same contexts with a base reply, worse versions (T5) and same-meaning versions (T6)."""
    label: str
    rows: pd.DataFrame
    base: list[str]
    worse: dict[str, list[str]]
    same: dict[str, list[str]]


@dataclass
class Pairs:
    """T4 (and T7): two replies per context. `sme_a`: share of SMEs preferring A (NaN = not rated).
    `truth_a`: 1 if A is truly better (sandbox only)."""
    label: str
    rows: pd.DataFrame
    a: list[str]
    b: list[str]
    sme_a: np.ndarray
    truth_a: np.ndarray | None = None
    sme_sme: float | None = None


@dataclass
class Uplift:
    """T8: a model's replies vs the agents' replies in the same contexts; true conversion only in the sandbox."""
    label: str
    rows: pd.DataFrame
    model: list[str]
    agent: list[str]
    true_model: np.ndarray | None = None
    true_agent: np.ndarray | None = None


@dataclass
class ValidationData:
    hist: pd.DataFrame                      # earlier slice: call_id, window, reply column, outcomes, controls
    test: pd.DataFrame                      # later slice
    controls: Callable[[pd.DataFrame], np.ndarray]
    reply_col: str = "reply"
    outcomes: tuple[str, ...] = ("converted",)
    match_on: tuple[str, ...] = ("objection_type", "segment", "channel")
    probes: list[Probes] = field(default_factory=list)
    pairs: list[Pairs] = field(default_factory=list)
    uplift: list[Uplift] = field(default_factory=list)
    truth_col: str | None = None           # sandbox: hidden true quality of the historical replies (T7)
    n_stability: int = 400                  # T1 sample of later contexts


def load_bars(path: Path = BARS_FILE) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["validation"]


# ------------------------------------------------------------------------------------------------ statistics
def boot_ci(values, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Interval for a mean by resampling (as ica_sandbox.stats.boot_ci, so the numbers match)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    means = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def pair_agree(a, b, label_a) -> np.ndarray:
    """1 if the score order matches the label, 0.5 on a tie, NaN if a score is missing."""
    a, b, y = (np.asarray(x, float) for x in (a, b, label_a))
    out = np.where(a == b, 0.5, ((a > b) == (y > 0.5)).astype(float))
    return np.where(np.isnan(a) | np.isnan(b) | np.isnan(y), np.nan, out)


def verdict(lo: float, hi: float, bar: dict) -> str:
    t = bar.get("target")
    if t is None:
        return REPORT
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return INCONCLUSIVE
    strict = bar.get("strict", False)
    if bar.get("higher", True):
        good, bad = (lo > t, hi <= t) if strict else (lo >= t, hi < t)
    else:
        good, bad = (hi < t, lo >= t) if strict else (hi <= t, lo > t)
    return GOOD if good else BAD if bad else INCONCLUSIVE


def _row(scorer, test, item, value, lo, hi, n, bars, set_label="", applies=True, **extra):
    bar = bars["checks"].get(test, {})
    v = verdict(lo, hi, bar) if applies else NOT_APPLICABLE
    return dict(scorer=scorer, test=test, set=set_label, item=item, value=value, lo=lo, hi=hi, n=n,
                target=bar.get("target"), higher=bar.get("higher", True), verdict=v, **extra)


# ------------------------------------------------------------------------------------------------ the checks
def t5_t6(s: Scorer, p: Probes, bars: dict) -> list[dict]:
    base = np.asarray(s.score(p.rows, p.base), float)
    scale = 1.0 if s.binary else (np.nanstd(base) + 1e-9)
    out = []
    for change, replies in p.worse.items():
        changed = np.array([r != b for r, b in zip(replies, p.base)])
        d = np.asarray(s.score(p.rows, replies), float) - base
        drop = np.where(np.isnan(d) | ~changed, np.nan, (d < 0).astype(float))
        lo, hi = boot_ci(drop)
        applies = s.catches is None or change in s.catches
        out.append(_row(s.name, "T5", change, np.nanmean(drop), lo, hi, int((~np.isnan(drop)).sum()), bars, p.label,
                        applies, rise=float(np.nanmean(np.where(np.isnan(d) | ~changed, np.nan, (d > 0).astype(float))))))
    for change, replies in p.same.items():
        changed = np.array([r != b for r, b in zip(replies, p.base)])
        d = np.asarray(s.score(p.rows, replies), float) - base
        a = np.where(np.isnan(d) | ~changed, np.nan, np.abs(d) / scale)
        lo, hi = boot_ci(a)
        out.append(_row(s.name, "T6", change, np.nanmean(a), lo, hi, int((~np.isnan(a)).sum()), bars, p.label))
    return out


def t1_stability(s: Scorer, data: ValidationData, bars: dict, seed: int = 0, n_boot: int = 300) -> list[dict]:
    if s.refit is None:
        return [_row(s.name, "T1", "refitted on each past period", np.nan, np.nan, np.nan, 0, bars, "", False,
                     note="fixed scorer: nothing is learned from data")]
    sample = data.test.sample(min(data.n_stability, len(data.test)), random_state=seed)
    reps = list(sample[data.reply_col])
    S = {w: np.asarray(s.refit(data.hist[data.hist.window == w]).score(sample, reps), float)
         for w in sorted(data.hist.window.unique())}
    names = list(S)

    def mean_rho(idx):
        r = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = S[names[i]][idx], S[names[j]][idx]
                m = ~np.isnan(a) & ~np.isnan(b)
                if m.sum() > 20:
                    r.append(spearmanr(a[m], b[m])[0])
        return float(np.mean(r)) if r else np.nan

    all_idx = np.arange(len(sample))
    rng = np.random.default_rng(seed)
    boots = [mean_rho(rng.integers(0, len(sample), len(sample))) for _ in range(n_boot)]
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    cover = float(np.mean([np.mean(~np.isnan(v)) for v in S.values()]))
    return [_row(s.name, "T1", f"refitted on each of {len(names)} past periods", mean_rho(all_idx), lo, hi, len(sample),
                 bars, "", coverage=cover)]


def t2_t3(scores: dict[str, np.ndarray], data: ValidationData, bars: dict, seed: int = 0) -> list[dict]:
    """T2 and T3 on historical agent replies of the later slice (as evalval.predictive_validity / matched_pairs)."""
    test = data.test.reset_index(drop=True)
    Xc_all = data.controls(test)
    words = test[data.reply_col].str.split().str.len().to_numpy(float)
    out = []
    for outcome in data.outcomes:
        y_all = test[outcome].to_numpy()
        for name, s in scores.items():
            s = np.asarray(s, float)
            m = ~np.isnan(s)
            y, Xc = y_all[m], Xc_all[m]
            z = (s[m] - s[m].mean()) / (s[m].std() + 1e-9)
            L = ((words[m] - words[m].mean()) / (words[m].std() + 1e-9))[:, None]
            cv, lr = StratifiedKFold(5, shuffle=True, random_state=seed), LogisticRegression(max_iter=3000)
            p0 = cross_val_predict(lr, Xc, y, cv=cv, method="predict_proba")[:, 1]
            p1 = cross_val_predict(lr, np.column_stack([Xc, z]), y, cv=cv, method="predict_proba")[:, 1]
            pl = cross_val_predict(lr, np.column_stack([Xc, L]), y, cv=cv, method="predict_proba")[:, 1]
            pls = cross_val_predict(lr, np.column_stack([Xc, L, z]), y, cv=cv, method="predict_proba")[:, 1]
            rng = np.random.default_rng(seed)
            b1, b2 = [], []
            for _ in range(300):
                i = rng.integers(0, len(y), len(y))
                if y[i].min() != y[i].max():
                    b1.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
                    b2.append(roc_auc_score(y[i], pls[i]) - roc_auc_score(y[i], pl[i]))
            d1 = roc_auc_score(y, p1) - roc_auc_score(y, p0)
            d2 = roc_auc_score(y, pls) - roc_auc_score(y, pl)
            out.append(_row(name, "T2", outcome, d1, *np.quantile(b1, [0.025, 0.975]), int(m.sum()), bars, "",
                            auc_controls=float(roc_auc_score(y, p0)), auc_with_score=float(roc_auc_score(y, p1))))
            out.append(_row(name, "T2_length", outcome, d2, *np.quantile(b2, [0.025, 0.975]), int(m.sum()), bars, ""))
        # T3: matched pairs, one converted and one not, same objection type, segment and channel
        rng = np.random.default_rng(seed)
        pos_idx, neg_idx = [], []
        for _, g in test.groupby(list(data.match_on)):
            pos, neg = g.index[g[outcome] == 1].to_numpy(), g.index[g[outcome] == 0].to_numpy()
            k = min(len(pos), len(neg))
            pos_idx += list(rng.permutation(pos)[:k])
            neg_idx += list(rng.permutation(neg)[:k])
        sel = rng.permutation(len(pos_idx))[:1500]
        p, q = np.array(pos_idx)[sel], np.array(neg_idx)[sel]
        for name, s in scores.items():
            s = np.asarray(s, float)
            acc = pair_agree(s[p], s[q], np.ones(len(p)))
            lo, hi = boot_ci(acc)
            out.append(_row(name, "T3", outcome, np.nanmean(acc), lo, hi, int((~np.isnan(acc)).sum()), bars, ""))
    return out


def t4_t7_pairs(s: Scorer, pr: Pairs, bars: dict) -> list[dict]:
    sa, sb = np.asarray(s.score(pr.rows, pr.a), float), np.asarray(s.score(pr.rows, pr.b), float)
    maj = np.where(np.isnan(pr.sme_a) | (pr.sme_a == 0.5), np.nan, (pr.sme_a > 0.5).astype(float))
    agree = pair_agree(sa, sb, maj)
    lo, hi = boot_ci(agree)
    out = [_row(s.name, "T4", "majority of SMEs", np.nanmean(agree), lo, hi, int((~np.isnan(agree)).sum()), bars,
                pr.label, coverage=float(np.mean(~np.isnan(agree))))]
    if pr.truth_a is not None:
        t = pair_agree(sa, sb, pr.truth_a)
        lo, hi = boot_ci(t)
        out.append(_row(s.name, "T7", "pairs ranked as the true quality", np.nanmean(t), lo, hi,
                        int((~np.isnan(t)).sum()), bars, pr.label))
    return out


def t8_uplift(s: Scorer, u: Uplift, bars: dict, seed: int = 0, n_boot: int = 1000) -> list[dict]:
    """Uplift of model replies over agent replies by this score; it must drop when each model reply is moved to
    another call (preferably one with a different objection type): otherwise the score is not reading the reply's fit
    to the call, and the uplift it reports is not evidence of a better reply."""
    rng = np.random.default_rng(seed)
    n = len(u.rows)
    typ = u.rows.objection_type.to_numpy() if "objection_type" in u.rows else np.zeros(n)
    donor = np.empty(n, int)
    for i in range(n):
        cand = np.flatnonzero(typ != typ[i])
        cand = cand if len(cand) else np.delete(np.arange(n), i)
        donor[i] = rng.choice(cand)
    moved = [u.model[j] for j in donor]
    f = s.to_prob or (lambda x: x)
    pm = f(np.asarray(s.score(u.rows, u.model), float))
    pa = f(np.asarray(s.score(u.rows, u.agent), float))
    px = f(np.asarray(s.score(u.rows, moved), float))
    ok = ~np.isnan(pm) & ~np.isnan(pa) & ~np.isnan(px)
    pm, pa, px = pm[ok], pa[ok], px[ok]

    def ratio(x, i):
        d = pa[i].mean()
        return x[i].mean() / d - 1 if d > 0 else np.nan

    # the check: how much the score falls when the reply is moved, in score units (its sign is what matters; the
    # ratios below are the uplift as usually claimed, shown when the agents' mean score is positive)
    full = np.arange(len(pm))
    B = rng.integers(0, len(pm), (n_boot, len(pm)))
    drop = np.array([(pm[i] - px[i]).mean() for i in B])
    real, placebo = ratio(pm, full), ratio(px, full)
    R = np.array([ratio(pm, i) for i in B])
    dlo, dhi = np.quantile(drop, [0.025, 0.975])
    rlo, rhi = np.nanquantile(R, [0.025, 0.975]) if np.isfinite(R).any() else (np.nan, np.nan)
    out = [_row(s.name, "T8", "score drop when the model's reply is moved to another call", float((pm - px).mean()),
                dlo, dhi, int(ok.sum()), bars, u.label, uplift=float(real), uplift_lo=float(rlo), uplift_hi=float(rhi),
                uplift_moved=float(placebo))]
    if u.true_model is not None:
        true = float(np.mean(u.true_model[ok]) / np.mean(u.true_agent[ok]) - 1)
        out[0]["true_uplift"] = true
        out[0]["truth_in_interval"] = bool(rlo <= true <= rhi)
    return out


def t7_truth(scores: dict[str, np.ndarray], data: ValidationData, bars: dict) -> list[dict]:
    out = []
    if data.truth_col is None:
        return out
    t = data.test[data.truth_col].to_numpy(float)
    for name, s in scores.items():
        s = np.asarray(s, float)
        m = ~np.isnan(s)
        out.append(_row(name, "T7", "Spearman with true quality, agent replies", float(spearmanr(s[m], t[m])[0]),
                        np.nan, np.nan, int(m.sum()), bars, "historical replies"))
    return out


# ------------------------------------------------------------------------------------------------ pipeline
def run(scorers: list[Scorer], data: ValidationData, bars: dict | None = None, seed: int = 0,
        log: Callable[[str], None] = print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every check for every scorer. Returns (long table of results, one status row per scorer).
    Results after a scorer's stopping stage are kept for diagnosis but marked `counts=False`."""
    bars = bars or load_bars()
    rows = []
    hist_scores = {}
    for s in scorers:
        log(f"{s.name}: scoring historical replies")
        hist_scores[s.name] = np.asarray(s.score(data.test, list(data.test[data.reply_col])), float)
    rows += t2_t3(hist_scores, data, bars, seed)
    rows += t7_truth(hist_scores, data, bars)
    for s in scorers:
        log(f"{s.name}: T5/T6, T1, T4, T8")
        for p in data.probes:
            rows += t5_t6(s, p, bars)
        rows += t1_stability(s, data, bars, seed)
        for pr in data.pairs:
            rows += t4_t7_pairs(s, pr, bars)
        for u in data.uplift:
            rows += t8_uplift(s, u, bars, seed)
    res = pd.DataFrame(rows)
    gates = [s.name for s in scorers if s.role == "gate"]
    m = res.scorer.isin(gates) & res.test.isin(["T2", "T2_length", "T3", "T4", "T8"])
    res.loc[m, "verdict"] = NOT_APPLICABLE
    res["stage"] = res.test.map({t.split("_")[0]: st["name"] for st in bars["stages"] for t in st["tests"]}
                                | {"T2_length": next(st["name"] for st in bars["stages"] if "T2" in st["tests"])}
                                ).fillna("report only")
    status, res["counts"] = [], True
    for s in scorers:
        stopped = None
        for st in bars["stages"]:
            r = res[(res.scorer == s.name) & (res.stage == st["name"])]
            if stopped:
                res.loc[r.index, "counts"] = False
                continue
            judged = r[r.verdict.isin([GOOD, BAD, INCONCLUSIVE])]
            missing = [t for t in st["tests"] if t not in set(r.test.str.split("_").str[0])]
            bad = judged[judged.verdict != GOOD]
            if len(bad) or missing:
                why = "; ".join(f"{t} {i} ({v})" for t, i, v in bad[["test", "item", "verdict"]].head(4).itertuples(index=False))
                why += (f"; not run: {', '.join(missing)}" if missing else "")
                more = len(bad) - 4
                stopped = dict(scorer=s.name, status=f"stopped at stage {st['name']}", stage=st["name"],
                               reasons=why + (f"; +{more} more" if more > 0 else ""),
                               n_bad=int((bad.verdict == BAD).sum()), n_inconclusive=int((bad.verdict == INCONCLUSIVE).sum()))
        status.append(stopped or dict(scorer=s.name, status="passed: may enter the reward", stage="all", reasons="",
                                      n_bad=0, n_inconclusive=0))
    return res, pd.DataFrame(status)


def render(res: pd.DataFrame, status: pd.DataFrame, title: str = "T1–T8 pipeline") -> str:
    def f(x, d=3):
        return "–" if x is None or pd.isna(x) else f"{x:.{d}f}"

    L = [f"# {title}", "", "## Where each scoring method stops", "",
         "| Scorer | Status | Why |", "|---|---|---|"]
    L += [f"| {r.scorer} | {r.status} | {r.reasons or '–'} |" for r in status.itertuples()]
    for test in ["T5", "T6", "T1", "T2", "T2_length", "T3", "T8", "T4", "T7"]:
        r = res[res.test == test]
        if not len(r):
            continue
        L += ["", f"## {test}", "", "| Scorer | Set | Item | Value [95% interval] | n | Target | Verdict |",
              "|---|---|---|---|---|---|---|"]
        for x in r.itertuples():
            iv = f"{f(x.value)} [{f(x.lo)}, {f(x.hi)}]" if pd.notna(x.lo) else f(x.value)
            L.append(f"| {x.scorer} | {x.set or '–'} | {x.item} | {iv} | {x.n} | {f(x.target, 2)} | {x.verdict}"
                     f"{'' if x.counts else ' (after stop)'} |")
    return "\n".join(L) + "\n"
