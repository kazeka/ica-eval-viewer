"""Keyword cues per objection super-/sub-cluster (SPEC §4.1, "Keyword cues"; Model 1 doc sections 2, 4, 5, 6).

- `learn_cues`: words and word pairs that set each objection type apart from all other customer turns, ranked by a
  log-odds z-score with an informative prior (Monroe, Colaresi & Quinn 2008, "Fightin' Words"), on document presence.
  Learned on the training split only.
- `annotate`: per customer turn of the calibration and test calls: which types' cues it contains; for objection turns,
  explicit (a cue of its own type in the turn) vs implicit, and how many turns earlier a cue of its type first appeared.
- `keyword_trigger`: a simple per-type keyword trigger (score = summed z of the matched cues of the best type,
  calibrated on the calibration split) in the same `m1_points` format, so every view can show it as a baseline.

Build (from ica-eval/):  uv run python -m ica_eval.cues data/public_m1
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

KEYWORD_MODEL = "Keyword trigger (baseline)"
FIRST_CUE_BINS = ["none earlier", "1–2 turns", "3–5 turns", "6+ turns"]
STOP = set("""a an the and or but if of to in on for with at by from as is are was were be been being it its this that these
those i you he she we they me my your our their him her them us do does did doing have has had will would can could should
may might must not no so just also very too than then there here what which who whom when where why how all any some more
most such only own same other into about up down out over again once i'm it's don't i'll you're i've let okay ok yes yeah
please thank thanks hello hi sir madam brother sister admin bro""".split())
_TOK = re.compile(r"[a-z][a-z']*")


def grams(text: str) -> set[str]:
    """Content words (3+ letters, not stop words) and word pairs that are not two stop words."""
    w = _TOK.findall(str(text).lower())
    return ({x for x in w if len(x) > 2 and x not in STOP}
            | {f"{a} {b}" for a, b in zip(w, w[1:]) if not (a in STOP and b in STOP)})


def learn_cues(turns: pd.DataFrame, label_col: str = "objection_type", top: int = 12, z_min: float = 3.0,
               min_docs: int = 5, alpha: float = 10.0, min_purity: float = 0.3) -> pd.DataFrame:
    """-> DataFrame(group, term, z, docs, share): top cues per group, learned on the customer turns of `turns`.
    For each group vs every other customer turn, with prior rate p_w over all customer turns:
    delta = logit((y_i + a)/(n_i + alpha)) - logit((y_j + a)/(n_j + alpha)), a = alpha * p_w, z = delta / sd."""
    c = turns[turns.speaker == "Customer"]
    lab = np.where(c.obj.to_numpy() == 1, c[label_col].astype(str).to_numpy(), "")
    docs = [grams(t) for t in c.text]
    vocab = pd.Index(sorted(set().union(*docs)))
    rows, cols = [], []
    for i, g in enumerate(docs):
        idx = vocab.get_indexer(list(g))
        rows += [i] * len(idx); cols += list(idx)
    from scipy.sparse import csr_matrix                                   # scipy ships with scikit-learn
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(vocab)))
    n_all = X.shape[0]
    y_all = np.asarray(X.sum(0)).ravel()
    p = (y_all + 0.5) / (n_all + 1.0)
    a, b = alpha * p, alpha * (1 - p)
    out = []
    for grp in sorted(set(lab) - {""}):
        m = lab == grp
        n_i, n_j = m.sum(), (~m).sum()
        y_i = np.asarray(X[m].sum(0)).ravel(); y_j = y_all - y_i
        d = (np.log((y_i + a) / (n_i - y_i + b)) - np.log((y_j + a) / (n_j - y_j + b)))
        sd = np.sqrt(1 / (y_i + a) + 1 / (n_i - y_i + b) + 1 / (y_j + a) + 1 / (n_j - y_j + b))
        z = d / sd
        # purity: of the customer turns that contain the term, at least `min_purity` are objections of this group,
        # so everyday domain words ("visa", "need") that fill neutral turns do not count as cues
        purity = y_i / np.maximum(y_all, 1)
        ok = (z >= z_min) & (y_i >= min(min_docs, max(2, n_i // 10))) & (purity >= min_purity)
        sel = np.argsort(-z * ok)[:top]
        sel = sel[ok[sel]]
        out += [dict(group=grp, term=vocab[k], z=float(z[k]), docs=int(y_i[k]), share=float(y_i[k] / n_i),
                     purity=float(purity[k])) for k in sel]
    return pd.DataFrame(out, columns=["group", "term", "z", "docs", "share", "purity"])


def _cue_hits(texts, cues: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """-> (groups, matrix turns x groups of summed z over matched cues)."""
    groups = sorted(cues.group.unique())
    gi = {g: i for i, g in enumerate(groups)}
    lut: dict[str, list[tuple[int, float]]] = {}
    for r in cues.itertuples():
        lut.setdefault(r.term, []).append((gi[r.group], r.z))
    S = np.zeros((len(texts), len(groups)))
    for i, t in enumerate(texts):
        for gm in grams(t) & lut.keys():
            for j, z in lut[gm]:
                S[i, j] += z
    return groups, S


def annotate(turns: pd.DataFrame, cues: pd.DataFrame) -> pd.DataFrame:
    """Per customer turn: cue_types ('|'-joined), has_cue, and for objection turns explicit / first_cue_bin."""
    c = turns[turns.speaker == "Customer"].sort_values(["call_id", "turn_idx"]).reset_index(drop=True)
    groups, S = _cue_hits(c.text.tolist(), cues)
    hit = S > 0
    c["cue_types"] = ["|".join(g for g, h in zip(groups, r) if h) for r in hit]
    c["has_cue"] = hit.any(1)
    gi = {g: i for i, g in enumerate(groups)}
    explicit = np.full(len(c), np.nan, dtype=object)
    first_bin = np.full(len(c), None, dtype=object)
    first_dist = np.full(len(c), np.nan)
    starts = c.groupby("call_id", sort=False).indices
    for _, idx in starts.items():
        idx = np.sort(idx)
        for k, r in enumerate(idx):
            if c.obj.iat[r] != 1:
                continue
            j = gi.get(str(c.objection_type.iat[r]))
            explicit[r] = bool(j is not None and hit[r, j])
            earlier = [q for q in idx[:k] if j is not None and hit[q, j]]
            if earlier:
                dist = int(c.turn_idx.iat[r] - c.turn_idx.iat[earlier[0]])
                first_dist[r] = dist
                first_bin[r] = FIRST_CUE_BINS[1] if dist <= 2 else FIRST_CUE_BINS[2] if dist <= 5 else FIRST_CUE_BINS[3]
            else:
                first_bin[r] = FIRST_CUE_BINS[0]
    c["explicit"], c["first_cue_dist"], c["first_cue_bin"] = explicit, first_dist, first_bin
    return c[["call_id", "turn_idx", "split", "obj", "cue_types", "has_cue", "explicit", "first_cue_dist", "first_cue_bin"]]


def keyword_trigger(points: pd.DataFrame, turns: pd.DataFrame, cues_type: pd.DataFrame, cues_cluster: pd.DataFrame,
                    taxonomy: pd.DataFrame) -> pd.DataFrame:
    """The keyword trigger in m1_points format. It sees the current customer turn only (N = 1). Rows (calls, truth,
    event matching) are copied from the candidate that covers the most calls."""
    cov = points.groupby("candidate", observed=True).call_id.nunique()
    src = points[(points.candidate == cov.idxmax())]
    src = src[src.N == src.N.min()].copy()
    txt = turns.set_index(["call_id", "turn_idx"]).text
    texts = txt.reindex(pd.MultiIndex.from_arrays([src.call_id, src.turn_idx])).fillna("").tolist()
    tg, St = _cue_hits(texts, cues_type)
    cg, Sc = _cue_hits(texts, cues_cluster)
    tr = turns[(turns.split == "train") & (turns.obj == 1)]
    top_type = tr.objection_type.astype(str).value_counts().idxmax()
    top_cluster = tr.groupby(tr.objection_type.astype(str)).objection_cluster.agg(lambda x: x.astype(str).value_counts().idxmax())
    score = St.max(1)
    pred_type = np.where(score > 0, np.array(tg)[St.argmax(1)], top_type)
    pred_cluster = np.where(Sc.max(1) > 0, np.array(cg)[Sc.argmax(1)], [top_cluster.get(t, cg[0]) for t in pred_type])
    src["p_raw"] = score
    cal = (src.part == "calib").to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(score[cal], src.obj.to_numpy()[cal])
    src["p_cal"] = iso.predict(score)
    src["pred_type"] = pd.Categorical(pred_type)
    src["pred_cluster"] = pd.Categorical(pred_cluster)
    src["candidate"] = KEYWORD_MODEL
    src["N"] = 1
    return src.reset_index(drop=True)


def build(data_dir: str | Path) -> None:
    d = Path(data_dir)
    turns = pd.read_parquet(d / "turns.parquet")
    points = pd.read_parquet(d / "m1_points.parquet")
    tax = pd.read_parquet(d / "taxonomy.parquet")
    train = turns[turns.split == "train"]
    cues_t = learn_cues(train, "objection_type").assign(level="super-cluster")
    cues_c = learn_cues(train, "objection_cluster", top=8, min_docs=3).assign(level="sub-cluster")
    pd.concat([cues_t, cues_c], ignore_index=True).to_parquet(d / "cues.parquet", index=False)
    ids = set(points.call_id.unique())
    ann = annotate(turns[turns.call_id.isin(ids)], cues_t)
    ann.to_parquet(d / "cue_flags.parquet", index=False)
    kw = keyword_trigger(points, turns, cues_t, cues_c, tax)
    kw.to_parquet(d / "m1_points_keyword.parquet", index=False)
    o = ann[ann.obj == 1]
    print(f"cues: {len(cues_t)} super-cluster, {len(cues_c)} sub-cluster | explicit share {o.explicit.astype(float).mean():.3f} "
          f"| first cue: {o.first_cue_bin.value_counts().to_dict()} | keyword rows {len(kw)}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/public_m1")
