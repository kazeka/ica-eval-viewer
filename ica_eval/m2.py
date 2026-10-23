"""Model 2 (reply recommender): LLM-free metrics on a replies table (Model 2 LLM-free evaluation doc, sections 1–6).

Input (data contract, written by an adapter such as adapters/sandbox_m2_export.py):
  m2_replies   run, policy, call_id, reply, plus any labels the source provides (rule_*, fail_*, reward_v2, true_*)
  m2_contexts  call_id, objection_type (super-cluster), sub_cluster, segment, channel, objection_text, agent_reply
  m2_references sub_cluster, call_id, reference  (replies to the same objection phrasing from converted calls)

Per-reply metrics are computed here from text only (no model, no LLM). Every metric is reported as a mean with a 95%
interval from resampling evaluation contexts, and as the paired difference from the comparison policy (B0) on the same
contexts. Corpus metrics (distinct-2, stock openings) are recomputed on every resample.

Own implementations, on purpose (for now): BLEU, ROUGE, JS divergence and the Levenshtein ratio are written here rather
than taken from sacrebleu / rouge-score / scipy / rapidfuzz (the libraries RAGAS's non-LLM metrics wrap), so the runtime
needs no packages beyond what the rest of ica-eval uses until those are approved inside AmEx. Those libraries are
test-only dependencies: tests/test_metrics_m2.py checks every formula here against them (identical results with our
tokenizer `words()` and sacrebleu's add-k smoothing, effective_order off). If they are approved, switching is a drop-in.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .checks import BAD, GOOD, INCONCLUSIVE, NA, verdict

_W = re.compile(r"[a-z0-9$%']+")
_S = re.compile(r"(?<=[.!?])\s+")


def words(t: str) -> list[str]:
    return _W.findall(str(t).lower())


def ngrams(w: list[str], n: int) -> Counter:
    return Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))


def rouge_n(hyp: str, ref: str, n: int) -> float:
    h, r = ngrams(words(hyp), n), ngrams(words(ref), n)
    ov = sum((h & r).values())
    if not ov:
        return 0.0
    p, rc = ov / max(1, sum(h.values())), ov / max(1, sum(r.values()))
    return 2 * p * rc / (p + rc)


def rouge_l(hyp: str, ref: str) -> float:
    a, b = words(hyp), words(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def bleu(hyp: str, refs: list[str], n_max: int = 4) -> float:
    """Sentence BLEU-4 with add-one smoothing for n > 1 and the closest-reference brevity penalty."""
    h = words(hyp)
    if not h or not refs:
        return 0.0
    rw = [words(r) for r in refs]
    logp = 0.0
    for n in range(1, n_max + 1):
        hc = ngrams(h, n)
        mx = Counter()
        for r in rw:
            mx |= ngrams(r, n)
        clip = sum(min(c, mx[g]) for g, c in hc.items())
        if n == 1 and not clip:
            return 0.0                  # no word in common: BLEU is 0 (as sacrebleu), not a tiny positive number
        tot = sum(hc.values())          # 0 when the reply is shorter than n: (0 + 1) / (0 + 1) = 1, as sacrebleu
        logp += math.log((clip + (n > 1)) / (tot + (n > 1))) / n_max
    rl = min((abs(len(r) - len(h)), len(r)) for r in rw)[1]
    bp = 1.0 if len(h) > rl else math.exp(1 - rl / len(h))
    return bp * math.exp(logp)


def js2(hyp: str, ref: str) -> float:
    """Jensen–Shannon divergence (base 2) between the word-pair distributions of reply and reference (0 = same)."""
    a, b = ngrams(words(hyp), 2), ngrams(words(ref), 2)
    if not a or not b:
        return 1.0
    ta, tb = sum(a.values()), sum(b.values())
    js = 0.0
    for g in set(a) | set(b):
        p, q = a[g] / ta, b[g] / tb
        m = (p + q) / 2
        js += (p * math.log2(p / m) if p else 0) / 2 + (q * math.log2(q / m) if q else 0) / 2
    return js


def _lcs_chars(a: str, b: str) -> int:
    """Length of the longest common character subsequence (bit-parallel, Hyyrö 2004: one big-int step per char of a)."""
    if not a or not b:
        return 0
    masks: dict[str, int] = {}
    for i, ch in enumerate(b):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    full = (1 << len(b)) - 1
    v = full
    for ch in a:
        u = v & masks.get(ch, 0)
        v = ((v + u) | (v - u)) & full
    return len(b) - v.bit_count()


def lev_ratio(a: str, b: str) -> float:
    """Levenshtein ratio, the "simple ratio" of SPEC R5 (= rapidfuzz `fuzz.ratio` / 100, checked in the tests):
    1 − insertions and deletions needed ÷ total characters = 2 · LCS ÷ total characters. Not difflib: its ratio is a
    different algorithm and junks frequent characters past 200 chars."""
    a, b = str(a).lower(), str(b).lower()
    return 2 * _lcs_chars(a, b) / (len(a) + len(b)) if a or b else 1.0


def max_ratio_at_least(a: str, pool: list[str], t: float = 0.9) -> bool:
    """True if any string in `pool` has a Levenshtein ratio ≥ t with `a` (length bound first: ratio ≤ 2·min ÷ total)."""
    a = str(a).lower()
    for b in pool:
        b = str(b).lower()
        n = len(a) + len(b)
        if (2 * min(len(a), len(b)) >= t * n) and (2 * _lcs_chars(a, b) >= t * n if n else True):
            return True
    return False


def n_sentences(t: str) -> int:
    return len([s for s in _S.split(str(t).strip()) if s])


# ------------------------------------------------------------------------------------------------ per-reply table
def reply_metrics(rep: pd.DataFrame, ctx: pd.DataFrame, refs: pd.DataFrame, n_refs: int = 8, seed: int = 0) -> pd.DataFrame:
    """Adds the text metrics to every row of `rep` (needs columns call_id, reply)."""
    c = ctx.set_index("call_id")
    rng = np.random.default_rng(seed)
    pool = {k: list(g.reference) for k, g in refs.groupby("sub_cluster")}
    conv = {cid: (lambda xs: [xs[i] for i in rng.permutation(len(xs))[:n_refs]])(pool.get(c.sub_cluster[cid], []))
            for cid in c.index}
    rows = []
    for cid, reply in zip(rep.call_id, rep.reply):
        agent, rs, obj = c.agent_reply[cid], conv[cid], c.objection_text[cid]
        w, ns = len(words(reply)), max(1, n_sentences(reply))
        rows.append(dict(
            bleu_agent=bleu(reply, [agent]), bleu_conv=bleu(reply, rs),
            rouge1_agent=rouge_n(reply, agent, 1), rouge2_agent=rouge_n(reply, agent, 2),
            rougeL_agent=rouge_l(reply, agent), rougeL_conv=max((rouge_l(reply, r) for r in rs), default=0.0),
            js2_agent=js2(reply, agent), lev_agent=lev_ratio(reply, agent),
            near_copy=max_ratio_at_least(reply, pool.get(c.sub_cluster[cid], [])), rouge_c=rouge_n(reply, obj, 1),
            words=w, sentences=ns, words_per_sentence=w / ns))
    return pd.concat([rep.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def distinct2(replies: list[str]) -> float:
    g, tot = set(), 0
    for r in replies:
        w = words(r)
        b = list(zip(w, w[1:]))
        g.update(b); tot += len(b)
    return len(g) / max(1, tot)


def stock_rate(replies: list[str], share: float = 0.05) -> float:
    """Share of replies whose first three words are used by at least `share` of all replies (stock openings)."""
    op = [" ".join(words(r)[:3]) for r in replies]
    cnt = Counter(op)
    lim = max(2, share * len(op))
    return float(np.mean([cnt[o] >= lim for o in op])) if op else np.nan


# ------------------------------------------------------------------------------------------------ metric catalogue
@dataclass
class M:
    id: str
    label: str
    col: str | None                 # per-reply column (mean); None for corpus metrics
    higher: bool | None             # True = higher is better, False = lower, None = no direction (report)
    role: str
    target: str                     # "b0" (vs B0) | "abs:<x>" (absolute, e.g. abs:1.0 for 100%) | "none"
    group: str
    corpus: Callable | None = None
    needs: str | None = None        # column that must exist (labels from the source)


CATALOG = [
    M("R1", "BLEU vs agent's reply", "bleu_agent", None, "Diagnostic", "none", "Reference-based"),
    M("R1", "BLEU vs converted replies (same phrasing)", "bleu_conv", None, "Diagnostic", "none", "Reference-based"),
    M("R2", "ROUGE-1 F1 vs agent's reply", "rouge1_agent", None, "Diagnostic", "none", "Reference-based"),
    M("R2", "ROUGE-2 F1 vs agent's reply", "rouge2_agent", None, "Diagnostic", "none", "Reference-based"),
    M("R3", "ROUGE-L F1 vs agent's reply", "rougeL_agent", None, "Diagnostic", "none", "Reference-based"),
    M("R3", "ROUGE-L F1 vs converted replies (best of 8)", "rougeL_conv", None, "Diagnostic", "none", "Reference-based"),
    M("R4", "JS divergence vs agent's reply (word pairs)", "js2_agent", None, "Diagnostic", "none", "Reference-based"),
    M("R5", "Near-copy of a converted reply (Levenshtein ≥ 0.9)", "near_copy", False, "Near-copy detection", "b0", "Reference-based"),
    M("R6–R8", "Embedding similarity, BERTScore, MoverScore", None, True, "Ranker after checks", "na", "Reference-based"),
    M("F1", "Format passes (1–3 sentences, not empty, no meta text)", "f_format", True, "Gate", "b0", "Rules", needs="rule_length_ok"),
    M("F2", "No new numbers or fees", "rule_no_unknown_numbers", True, "Gate", "b0", "Rules", needs="rule_no_unknown_numbers"),
    M("F3", "No invented offers (known patterns)", "rule_no_invented_offer", True, "Gate", "b0", "Rules", needs="rule_no_invented_offer"),
    M("F4", "Placeholders kept", "rule_placeholders_ok", True, "Gate", "abs:1.0", "Rules", needs="rule_placeholders_ok"),
    M("F1–F4", "All rule checks pass", "rule_passed", True, "Gate", "b0", "Rules", needs="rule_passed"),
    M("F5", "Pushy or high pressure", "fail_pushy", False, "Failure rate", "b0", "Rules", needs="fail_pushy"),
    M("F5", "Repeats the earlier pitch", "fail_repeats_pitch", False, "Failure rate", "b0", "Rules", needs="fail_repeats_pitch"),
    M("F6", "Words per sentence", "words_per_sentence", None, "Report", "none", "Rules"),
    M("F7", "Objection relevance (embedding similarity)", None, True, "Ranker after checks", "na", "Relevance"),
    M("F8", "Off target (does not answer, or answers another objection)", "f_off_target", False, "Failure rate", "b0", "Relevance", needs="fail_does_not_answer"),
    M("F9", "ROUGE-C: word overlap with the objection turn", "rouge_c", None, "Diagnostic", "none", "Relevance"),
    M("F10–F11", "Unsupported claims (entailment, question-answer checks)", None, False, "Gate after checks", "na", "Consistency"),
    M("F12", "Distinct word pairs (distinct-2)", None, True, "Report", "b0", "Style and diversity", corpus=distinct2),
    M("F13", "Stock openings (first 3 words shared by ≥ 5% of replies)", None, False, "Failure rate", "b0", "Style and diversity", corpus=stock_rate),
    M("F13", "Near-copy of a training reply", "near_copy", False, "Failure rate", "b0", "Style and diversity"),
    M("F14", "Words per reply", "words", None, "Drift check", "none", "Style and diversity"),
    M("F15–F16", "Required / forbidden terms per type; coverage of earlier cues", None, None, "Gate / report", "na", "Keywords"),
]
SANDBOX = [  # labels that only the sandbox provides (its detectors and hidden truth); on AmEx data, SMEs and judges
    M("S", "Reward v2 (what training optimised)", "reward_v2", True, "Sandbox", "b0", "Sandbox", needs="reward_v2"),
    M("S", "True quality (sandbox only)", "true_quality", True, "Sandbox truth", "b0", "Sandbox", needs="true_quality"),
    M("S", "True P(convert) (sandbox only)", "true_p_convert", True, "Sandbox truth", "b0", "Sandbox", needs="true_p_convert"),
]


def prepare(rep: pd.DataFrame) -> pd.DataFrame:
    rep = rep.copy()
    if {"rule_length_ok", "rule_not_empty", "rule_no_meta_text"} <= set(rep):
        rep["f_format"] = rep.rule_length_ok & rep.rule_not_empty & rep.rule_no_meta_text
    if {"fail_does_not_answer", "fail_wrong_objection"} <= set(rep):
        rep["f_off_target"] = rep.fail_does_not_answer | rep.fail_wrong_objection
    return rep


# ------------------------------------------------------------------------------------------------ statistics
def _weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([np.ones(n)] + [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]).astype(float)


def _stat(m: M, d: pd.DataFrame, W: np.ndarray) -> np.ndarray:
    """Point (row 0) and resampled values (rows 1..) of metric m on frame d (one row per context, aligned to W)."""
    if m.corpus is not None:
        reps = list(d.reply)
        out = [m.corpus(reps)]
        for w in W[1:]:
            idx = np.repeat(np.arange(len(reps)), w.astype(int))
            out.append(m.corpus([reps[i] for i in idx]))
        return np.array(out, dtype=float)
    x = d[m.col].astype(float).to_numpy()
    return W @ x / W.sum(1)


def available(m: M, rep: pd.DataFrame) -> bool:
    if m.target == "na":
        return False
    return (m.needs is None or m.needs in rep) and (m.col is None or m.col in rep)


def compare(a: pd.DataFrame, b: pd.DataFrame | None, metrics: list[M], n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Value [95% interval] for policy a, and the paired difference a − b on the contexts both answered."""
    if b is not None:
        common = sorted(set(a.call_id) & set(b.call_id))
        a = a.set_index("call_id").loc[common].reset_index()
        b = b.set_index("call_id").loc[common].reset_index()
    W = _weights(len(a), n_boot, seed)
    rows = []
    for m in metrics:
        if not available(m, a):
            rows.append(dict(id=m.id, metric=m.label, group=m.group, role=m.role, value=np.nan, lo=np.nan, hi=np.nan,
                             b=np.nan, diff=np.nan, dlo=np.nan, dhi=np.nan, verdict="Not run" if m.target == "na" else "Not available"))
            continue
        va = _stat(m, a, W)
        lo, hi = np.nanpercentile(va[1:], [2.5, 97.5])
        r = dict(id=m.id, metric=m.label, group=m.group, role=m.role, value=va[0], lo=lo, hi=hi,
                 b=np.nan, diff=np.nan, dlo=np.nan, dhi=np.nan, verdict="Report")
        if b is not None:
            vb = _stat(m, b, W)
            dv = va - vb
            dlo, dhi = np.nanpercentile(dv[1:], [2.5, 97.5])
            r.update(b=vb[0], diff=dv[0], dlo=dlo, dhi=dhi)
            if m.target == "b0" and m.higher is not None:
                r["verdict"] = verdict(dlo, dhi, 0.0, m.higher)
        if m.target.startswith("abs:"):
            t = float(m.target[4:])
            r["verdict"] = verdict(lo, hi, t, True) if t < 1 else (GOOD if va[0] >= t else BAD)
        rows.append(r)
    return pd.DataFrame(rows)


def by_group(rep: pd.DataFrame, ctx: pd.DataFrame, col: str, metric: M, policies: list[str]) -> pd.DataFrame:
    """Point value and n per policy × group (super- or sub-cluster)."""
    d = rep.merge(ctx[["call_id", col]], on="call_id")
    out = []
    for p in policies:
        for g, x in d[d.policy == p].groupby(col, observed=True):
            v = metric.corpus(list(x.reply)) if metric.corpus is not None else x[metric.col].astype(float).mean()
            out.append(dict(policy=p, group=g, value=v, n=len(x)))
    return pd.DataFrame(out)


TARGET_TEXT = {"b0": "{} B0", "none": "No target (explains other results)", "na": "–"}


def target_text(m: M) -> str:
    if m.target == "b0":
        return ("≥ B0" if m.higher else "≤ B0") if m.higher is not None else "Report"
    if m.target.startswith("abs:"):
        return "100%" if float(m.target[4:]) == 1 else f"≥ {m.target[4:]}"
    return TARGET_TEXT.get(m.target, m.target)


__all__ = ["CATALOG", "SANDBOX", "M", "reply_metrics", "prepare", "compare", "by_group", "target_text",
           "GOOD", "BAD", "INCONCLUSIVE", "NA"]


def build(data_dir) -> pd.DataFrame:
    """Compute the per-reply text metrics once and store them next to the replies (m2_reply_metrics.parquet)."""
    from pathlib import Path
    d = Path(data_dir)
    rep = pd.read_parquet(d / "m2_replies.parquet")
    out = reply_metrics(rep, pd.read_parquet(d / "m2_contexts.parquet"), pd.read_parquet(d / "m2_references.parquet"))
    out.to_parquet(d / "m2_reply_metrics.parquet", index=False)
    return out


if __name__ == "__main__":
    import sys
    import time
    t = time.time()
    r = build(sys.argv[1] if len(sys.argv) > 1 else "data/sandbox_m2")
    print(f"{len(r)} replies, {time.time() - t:.0f} s")
