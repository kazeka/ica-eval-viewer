"""Model 1 checks, business view, keyword-cue breakdowns and early-warning counts (Model 1 doc sections 2–6).

Everything is computed from per-call counts at one threshold, so 95% intervals come from resampling whole calls, and a
comparison between two models resamples the calls they share with the same weights (paired difference).

Reading rule (all three AmEx docs): good = the whole 95% interval is on the right side of the target; bad = the whole
interval is on the wrong side; inconclusive = the interval crosses the target.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .cues import FIRST_CUE_BINS

BINS = 10
GOOD, BAD, INCONCLUSIVE, NA = "Good", "Bad", "Inconclusive", "Needs AmEx data"


def per_call(d: pd.DataFrame, t: float, flags: pd.DataFrame | None = None, lookahead: int = 6) -> pd.DataFrame:
    """Per-call counts for the rows `d` of one model at one context length (one split), at threshold t.
    `flags` (from cues.annotate) adds explicit/implicit, cue-word and turns-since-first-cue counts.
    `lookahead`: an alert outside every objection window counts as a possible early warning if an objection follows
    in the same call within this many turns."""
    d = d.sort_values(["call_id", "turn_idx"]).reset_index(drop=True)
    if flags is not None:
        d = d.merge(flags[["call_id", "turn_idx", "has_cue", "explicit", "first_cue_bin"]], on=["call_id", "turn_idx"], how="left")
    calls, ci = np.unique(d.call_id.astype(str).to_numpy(), return_inverse=True)
    C = len(calls)
    fire = d.p_cal.to_numpy() >= t
    inw = d.in_win.to_numpy()
    obj = d.obj.to_numpy() == 1
    rel = d.rel.to_numpy()
    ev = d.event.astype(str).to_numpy()
    det_ev = set(ev[fire & inw])
    now_ev = set(ev[fire & (rel == 0)])
    is_det = obj & np.isin(ev, list(det_ev))
    is_now = obj & np.isin(ev, list(now_ev))

    def cnt(m):
        return np.bincount(ci[m], minlength=C).astype(float)

    out = dict(calls=np.ones(C), n_obj=cnt(obj), det=cnt(is_det), det_now=cnt(is_now), fa=cnt(fire & ~inw),
               alerts=cnt(fire), neg=cnt(~inw), turns=cnt(np.ones(len(d), bool)))
    # possible early warnings: turns outside every window that come up to `lookahead` turns before an objection
    nxt = d.turn_idx.where(obj).groupby(d.call_id).bfill().to_numpy()
    lead = nxt - d.turn_idx.to_numpy()
    early_turn = ~inw & ~obj & (lead >= 1) & (lead <= lookahead)
    early_alert = fire & early_turn
    warned = set(zip(d.call_id[early_alert], (d.turn_idx[early_alert] + lead[early_alert]).astype(int)))
    obj_warned = obj & np.array([(c, int(j)) in warned for c, j in zip(d.call_id, d.turn_idx)])
    out.update(early_turns=cnt(early_turn), early_alerts=cnt(early_alert), obj_warned=cnt(obj_warned))
    # calibration bins over all customer turns
    p = d.p_cal.to_numpy()
    b = np.minimum((p * BINS).astype(int), BINS - 1)
    for k in range(BINS):
        m = b == k
        out[f"cal_n{k}"] = cnt(m)
        out[f"cal_p{k}"] = np.bincount(ci[m], weights=p[m], minlength=C)
        out[f"cal_y{k}"] = np.bincount(ci[m], weights=obj[m].astype(float), minlength=C)
    if flags is not None:
        exp = d.explicit.to_numpy()
        e, i = obj & (exp == True), obj & (exp == False)  # noqa: E712  (object column with NaN for non-objections)
        cue = d.has_cue.fillna(False).to_numpy(dtype=bool)
        out.update(n_exp=cnt(e), det_exp=cnt(e & is_det), n_imp=cnt(i), det_imp=cnt(i & is_det),
                   neg_cue=cnt(~inw & cue), fa_cue=cnt(fire & ~inw & cue),
                   neg_nocue=cnt(~inw & ~cue), fa_nocue=cnt(fire & ~inw & ~cue))
        fb = d.first_cue_bin.astype(object).to_numpy()
        for k, lab in enumerate(FIRST_CUE_BINS):
            m = obj & (fb == lab)
            out[f"n_fc{k}"], out[f"det_fc{k}"] = cnt(m), cnt(m & is_det)
    return pd.DataFrame(out, index=pd.Index(calls, name="call_id"))


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def ece_from(s) -> np.ndarray:
    n = sum(s[f"cal_n{k}"] for k in range(BINS))
    return _div(sum(np.abs(s[f"cal_p{k}"] - s[f"cal_y{k}"]) for k in range(BINS)), n)


def adj_precision(prevalence: float) -> Callable:
    """Precision at another objection rate: prevalence * recall / (prevalence * recall + (1 - prevalence) * false-alarm rate per turn)."""
    def f(s):
        r, fpr = _div(s["det"], s["n_obj"]), _div(s["fa"], s["neg"])
        return _div(prevalence * r, prevalence * r + (1 - prevalence) * fpr)
    return f


METRICS: dict[str, Callable] = {
    "recall": lambda s: _div(s["det"], s["n_obj"]),
    "precision": lambda s: _div(s["det"], s["det"] + s["fa"]),
    "f1": lambda s: _div(2 * _div(s["det"], s["n_obj"]) * _div(s["det"], s["det"] + s["fa"]),
                         _div(s["det"], s["n_obj"]) + _div(s["det"], s["det"] + s["fa"])),
    "fa_per_call": lambda s: _div(s["fa"], s["calls"]),
    "fa_rate_turn": lambda s: _div(s["fa"], s["neg"]),
    "ece": ece_from,
    "alerts_per_call": lambda s: _div(s["alerts"], s["calls"]),
    "recall_now": lambda s: _div(s["det_now"], s["n_obj"]),
    "recall_explicit": lambda s: _div(s["det_exp"], s["n_exp"]),
    "recall_implicit": lambda s: _div(s["det_imp"], s["n_imp"]),
    "fa_rate_cue": lambda s: _div(s["fa_cue"], s["neg_cue"]),
    "fa_rate_nocue": lambda s: _div(s["fa_nocue"], s["neg_nocue"]),
    "early_share": lambda s: _div(s["early_alerts"], s["fa"]),
    "early_chance": lambda s: _div(s["early_turns"], s["neg"]),
    "obj_warned": lambda s: _div(s["obj_warned"], s["n_obj"]),
    **{f"recall_fc{k}": (lambda s, k=k: _div(s[f"det_fc{k}"], s[f"n_fc{k}"])) for k in range(len(FIRST_CUE_BINS))},
}
LABELS = {"recall": "Recall", "precision": "Precision", "f1": "F1", "fa_per_call": "False alarms per call",
          "fa_rate_turn": "False-alarm rate per turn", "ece": "Calibration error (ECE)", "alerts_per_call": "Alerts per call",
          "recall_now": "Objections flagged on the objection turn itself", "recall_explicit": "Recall, explicit objections",
          "recall_implicit": "Recall, implicit objections", "fa_rate_cue": "False-alarm rate, turns with a cue word",
          "fa_rate_nocue": "False-alarm rate, turns without a cue word",
          "early_share": "False alarms that come before an objection", "early_chance": "Same share by chance",
          "obj_warned": "Objections with an early alert"}


def _weights(C: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([np.ones(C)] + [np.bincount(rng.integers(0, C, C), minlength=C) for _ in range(n_boot)]).astype(float)


def boot(pc: pd.DataFrame, names: list[str], n_boot: int = 500, seed: int = 0, extra: dict | None = None) -> pd.DataFrame:
    """-> DataFrame(metric, value, lo, hi): point value and 95% interval from resampling calls."""
    fns = {**{n: METRICS[n] for n in names}, **(extra or {})}
    W = _weights(len(pc), n_boot, seed)
    s = {c: W @ pc[c].to_numpy() for c in pc.columns}
    rows = []
    for n, f in fns.items():
        v = f(s)
        lo, hi = (np.nanpercentile(v[1:], [2.5, 97.5]) if np.isfinite(v[1:]).any() else (np.nan, np.nan))
        rows.append(dict(metric=n, value=float(v[0]), lo=float(lo), hi=float(hi)))
    return pd.DataFrame(rows).set_index("metric")


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, names: list[str], n_boot: int = 500, seed: int = 0,
              extra: dict | None = None) -> pd.DataFrame:
    """Paired difference a - b on the calls both were scored on, same resampling weights for both."""
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    fns = {**{n: METRICS[n] for n in names}, **(extra or {})}
    W = _weights(len(common), n_boot, seed)
    sa = {c: W @ a[c].to_numpy() for c in a.columns}
    sb = {c: W @ b[c].to_numpy() for c in b.columns}
    rows = []
    for n, f in fns.items():
        va, vb = f(sa), f(sb)
        v = va - vb
        lo, hi = (np.nanpercentile(v[1:], [2.5, 97.5]) if np.isfinite(v[1:]).any() else (np.nan, np.nan))
        rows.append(dict(metric=n, a=float(va[0]), b=float(vb[0]), diff=float(v[0]), lo=float(lo), hi=float(hi), calls=len(common)))
    return pd.DataFrame(rows).set_index("metric")


def verdict(lo: float, hi: float, target: float, higher_is_better: bool = True) -> str:
    """Good / Bad / Inconclusive from a 95% interval and a target."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return INCONCLUSIVE
    if higher_is_better:
        return GOOD if lo >= target else BAD if hi < target else INCONCLUSIVE
    return GOOD if hi <= target else BAD if lo > target else INCONCLUSIVE


def combine(verdicts: list[str]) -> str:
    return BAD if BAD in verdicts else GOOD if all(v == GOOD for v in verdicts) else INCONCLUSIVE
