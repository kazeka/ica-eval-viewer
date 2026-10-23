"""Model 1 metrics by context length (N) and objection super-/sub-cluster, and the sweet-spot rule (SPEC §4.1, §7).

Input: the `m1_points` table (one row per candidate x N x split x customer turn) with calibrated score `p_cal`, truth
(`obj`, `true_type`, `true_cluster`), predictions (`pred_type`, `pred_cluster`) and event matching (`in_win`, `event`,
`rel`) as produced by an adapter. Thresholds are chosen on the `calib` split and every metric is reported on `test`.

Definitions (SPEC §4.1):
- an alert is a turn with p_cal >= threshold; an objection is detected if an alert falls inside its detection window;
- recall by the TRUE group of the objection; precision and false alarms by the PREDICTED group of the alert
  (a false alarm has no true group); the detecting alert is the first one in the window;
- false-alarm rate per turn = false alarms / customer turns outside every objection window;
- 95% intervals come from resampling whole calls (recall, precision, F1, false alarms); PR-AUC, ECE and type F1 are
  point estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

CI_METRICS = ["recall", "precision", "f1", "fa_per_call", "fa_rate_turn"]
POINT_METRICS = ["pr_auc", "ece", "type_f1"]
LABELS = {"recall": "Recall", "precision": "Precision", "f1": "F1", "fa_per_call": "False alarms per call",
          "fa_rate_turn": "False-alarm rate per turn", "pr_auc": "PR-AUC", "ece": "Calibration error (ECE)",
          "type_f1": "Type accuracy (F1)"}
LOWER_IS_BETTER = {"fa_per_call", "fa_rate_turn", "ece"}
OVERALL = "All objections"
DECIMALS = 6   # calibrated scores closer than 1e-6 are one level (isotonic interpolation leaves float noise between ties)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    if len(p) == 0:
        return np.nan
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


# --------------------------------------------------------------------------------------------- threshold choice
def _overall_curve(cal: pd.DataFrame) -> pd.DataFrame:
    """Recall, precision, false alarms per call on the calibration split for every distinct calibrated score."""
    n_calls = cal.call_id.nunique()
    win = cal[cal.in_win]
    ev_max = win.groupby("event", observed=True).p_cal.max()          # an event is detected at t if its best in-window score >= t
    n_obj = int(cal.obj.sum())
    neg = np.sort(np.round(cal.loc[~cal.in_win, "p_cal"].to_numpy(), DECIMALS))
    evs = np.sort(np.round(ev_max.to_numpy(), DECIMALS))
    ts = np.unique(np.round(cal.p_cal.to_numpy(), DECIMALS))
    det = len(evs) - np.searchsorted(evs, ts, side="left")
    fa = len(neg) - np.searchsorted(neg, ts, side="left")
    rec = det / max(n_obj, 1)
    prec = np.where(det + fa > 0, det / np.maximum(det + fa, 1), np.nan)
    f1 = np.where((rec + np.nan_to_num(prec)) > 0, 2 * rec * np.nan_to_num(prec) / (rec + np.nan_to_num(prec) + 1e-12), 0)
    return pd.DataFrame(dict(t=ts, recall=rec, precision=prec, f1=f1, fa_per_call=fa / max(n_calls, 1)))


def choose_threshold(cal: pd.DataFrame, rule: str = "budget", budget: float = 0.05) -> tuple[float, bool]:
    """-> (threshold, budget_met). rule='budget': most recall with false alarms per call <= budget, i.e. the lowest
    threshold within budget (recall only falls as the threshold rises; the lowest one also leaves the most margin for
    new calls); rule='f1': best F1. If no threshold meets the budget, the one with the fewest false alarms is returned."""
    c = _overall_curve(cal)
    if rule == "f1":
        t, met = float(c.t.iloc[int(np.nanargmax(c.f1.to_numpy()))]), True
    else:
        ok = c[c.fa_per_call <= budget]
        t, met = (float(c.t.iloc[-1]), False) if ok.empty else (float(ok.t.min()), True)
    # Place the threshold halfway to the next lower calibration score level: same result on the calibration split,
    # and test scores that sit exactly on (or a rounding error below) a calibrated level are not cut off by chance.
    lower = c.t[c.t < t - 10 ** -DECIMALS / 2]
    return (float((t + lower.max()) / 2) if len(lower) else t), met


# --------------------------------------------------------------------------------------------- metrics at a threshold
def _counts(d: pd.DataFrame, t: float, true_col: str, pred_col: str, groups: list[str]):
    """Per-call count matrices (calls x groups) at threshold t."""
    calls = pd.Index(d.call_id.unique())
    ci = calls.get_indexer(d.call_id)
    gi_true = pd.Index(groups).get_indexer(d[true_col].astype(object))
    gi_pred = pd.Index(groups).get_indexer(d[pred_col].astype(object))
    G, C = len(groups), len(calls)
    fire = d.p_cal.to_numpy() >= t
    inw = d.in_win.to_numpy()
    obj = d.obj.to_numpy() == 1

    def mat(mask, gidx):
        m = mask & (gidx >= 0)
        return np.bincount(ci[m] * G + gidx[m], minlength=C * G).reshape(C, G).astype(float)

    n_obj = mat(obj, gi_true)
    # detection: first alert inside each objection's window; group of the objection = true group at the objection turn
    ev_true = d.loc[obj, ["event", true_col]].drop_duplicates("event").set_index("event")[true_col].astype(object)
    hits = d.loc[fire & inw, ["event", "rel", "call_id", pred_col]]
    first = hits.sort_values("rel").drop_duplicates("event")
    first_ci = calls.get_indexer(first.call_id)
    first_true = pd.Index(groups).get_indexer(first.event.map(ev_true).astype(object))
    first_pred = pd.Index(groups).get_indexer(first[pred_col].astype(object))

    def mat_idx(cidx, gidx):
        m = gidx >= 0
        return np.bincount(cidx[m] * G + gidx[m], minlength=C * G).reshape(C, G).astype(float)

    det_true = mat_idx(first_ci, first_true)
    det_pred = mat_idx(first_ci, first_pred)
    fa_pred = mat(fire & ~inw, gi_pred)
    neg = np.bincount(ci[~inw], minlength=C).astype(float)
    det_all = np.bincount(first_ci, minlength=C).astype(float)
    fa_all = np.bincount(ci[fire & ~inw], minlength=C).astype(float)
    nobj_all = np.bincount(ci[obj], minlength=C).astype(float)
    return dict(n_obj=n_obj, det_true=det_true, det_pred=det_pred, fa_pred=fa_pred, neg=neg,
                det_all=det_all, fa_all=fa_all, nobj_all=nobj_all, n_calls=C)


def _ratios(s: dict, n_calls: int) -> dict:
    """Metrics from summed counts. Arrays of shape (..., G) for groups and (...,) for overall."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = s["det_true"] / s["n_obj"]
        prec = s["det_pred"] / (s["det_pred"] + s["fa_pred"])
        out = dict(recall=rec, precision=prec, f1=2 * rec * prec / (rec + prec),
                   fa_per_call=s["fa_pred"] / n_calls, fa_rate_turn=s["fa_pred"] / s["neg"][..., None])
        r0 = s["det_all"] / s["nobj_all"]; p0 = s["det_all"] / (s["det_all"] + s["fa_all"])
        overall = dict(recall=r0, precision=p0, f1=2 * r0 * p0 / (r0 + p0), fa_per_call=s["fa_all"] / n_calls,
                       fa_rate_turn=s["fa_all"] / s["neg"])
    return out, overall


def metrics_at(d: pd.DataFrame, t: float, level: str = "type", groups: list[str] | None = None,
               n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """Long table: group, metric, value, lo, hi, n_obj for the test frame `d` at threshold t.
    level='type' uses super-clusters, level='cluster' sub-clusters (optionally restricted to `groups`)."""
    true_col, pred_col = ("true_type", "pred_type") if level == "type" else ("true_cluster", "pred_cluster")
    if groups is None:
        groups = sorted(set(d[true_col].dropna().astype(str)) | set(d[pred_col].dropna().astype(str)))
    k = _counts(d, t, true_col, pred_col, groups)
    C = k["n_calls"]
    keys = ("n_obj", "det_true", "det_pred", "fa_pred", "neg", "det_all", "fa_all", "nobj_all")
    point_g, point_o = _ratios({x: k[x].sum(0) for x in keys}, C)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, C, C), minlength=C) for _ in range(n_boot)]).astype(float)  # B x C
    boot_g, boot_o = _ratios({x: W @ k[x] for x in keys}, C)
    rows = []
    n_obj_g = k["n_obj"].sum(0)
    for m in CI_METRICS:
        lo_g, hi_g = np.nanpercentile(boot_g[m], [2.5, 97.5], axis=0)
        for j, g in enumerate(groups):
            rows.append(dict(group=g, metric=m, value=point_g[m][j], lo=lo_g[j], hi=hi_g[j], n_obj=n_obj_g[j]))
        lo, hi = np.nanpercentile(boot_o[m], [2.5, 97.5])
        rows.append(dict(group=OVERALL, metric=m, value=float(point_o[m]), lo=lo, hi=hi, n_obj=k["nobj_all"].sum()))
    # point-only metrics
    y, p = d.obj.to_numpy(), d.p_cal.to_numpy()
    o = d[d.obj == 1]
    tf1 = f1_score(o[true_col].astype(str), o[pred_col].astype(str), labels=groups, average=None, zero_division=0) if len(o) else np.full(len(groups), np.nan)
    for j, g in enumerate(groups):
        sel = (d[true_col].astype(object) == g).to_numpy() | (y == 0)
        ap = average_precision_score(y[sel], p[sel]) if (y[sel] == 1).any() else np.nan
        pm = (d[pred_col].astype(object) == g).to_numpy()
        rows += [dict(group=g, metric="pr_auc", value=ap, lo=np.nan, hi=np.nan, n_obj=n_obj_g[j]),
                 dict(group=g, metric="ece", value=ece(p[pm], y[pm]) if pm.sum() >= 50 else np.nan, lo=np.nan, hi=np.nan, n_obj=n_obj_g[j]),
                 dict(group=g, metric="type_f1", value=tf1[j], lo=np.nan, hi=np.nan, n_obj=n_obj_g[j])]
    rows += [dict(group=OVERALL, metric="pr_auc", value=average_precision_score(y, p) if y.any() else np.nan, lo=np.nan, hi=np.nan, n_obj=y.sum()),
             dict(group=OVERALL, metric="ece", value=ece(p, y), lo=np.nan, hi=np.nan, n_obj=y.sum()),
             dict(group=OVERALL, metric="type_f1", value=float(np.mean(tf1)) if len(o) else np.nan, lo=np.nan, hi=np.nan, n_obj=y.sum())]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- grid and sweet spot
def grid(points: pd.DataFrame, candidate: str, rule: str = "budget", budget: float = 0.05, level: str = "type",
         groups: list[str] | None = None, n_boot: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (long metrics table with an N column, thresholds per N). Thresholds are chosen on calib, overall."""
    P = points[points.candidate == candidate]
    out, thr = [], []
    for N in sorted(P.N.unique()):
        cal, test = P[(P.N == N) & (P.part == "calib")], P[(P.N == N) & (P.part == "test")]
        if cal.empty or test.empty:
            continue
        t, met = choose_threshold(cal, rule, budget)
        thr.append(dict(N=int(N), threshold=t, budget_met=met))
        out.append(metrics_at(test, t, level, groups, n_boot).assign(N=int(N)))
    return pd.concat(out, ignore_index=True), pd.DataFrame(thr)


def sweet_spot(g: pd.DataFrame, metric: str = "recall", min_n: int = 30) -> tuple[int | None, pd.DataFrame]:
    """N* = smallest N whose overall `metric` is within the 95% interval of the best N.
    Flags groups whose interval at N* lies entirely below their own best N (groups with >= min_n objections)."""
    o = g[(g.group == OVERALL) & (g.metric == metric)].sort_values("N")
    if o.empty or o.value.isna().all():
        return None, pd.DataFrame()
    best = o.loc[o.value.idxmax()]
    floor = best.lo if pd.notna(best.lo) else best.value
    n_star = int(o[o.value >= floor].N.min())
    flags = []
    for grp, x in g[(g.metric == metric) & (g.group != OVERALL)].groupby("group"):
        x = x[x.n_obj >= min_n]
        if x.empty or x.value.isna().all():
            continue
        b = x.loc[x.value.idxmax()]
        at = x[x.N == n_star]
        if not at.empty and pd.notna(at.hi.iat[0]) and at.hi.iat[0] < b.value:
            flags.append(dict(group=grp, value_at_n_star=at.value.iat[0], best_N=int(b.N), best_value=b.value))
    return n_star, pd.DataFrame(flags)
