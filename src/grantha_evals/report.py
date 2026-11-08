"""Bootstrap confidence intervals for chrF++ and paired model differences over the same verses."""
import json
import random

from sacrebleu.metrics import CHRF

from . import translate
from .models import ROOT

RESAMPLES = 1000
RAW = ROOT / "results" / "raw"


def _stats(hyps, refs):
    return CHRF(word_order=2)._extract_corpus_statistics(hyps, [refs])


def _score(metric, stats, idx):
    total = [sum(col) for col in zip(*(stats[i] for i in idx))]
    return metric._compute_score_from_stats(total).score


def load(model, corpus):
    refs = {i["id"]: i["english"] for i in translate.select(corpus, translate.PILOT_WORKS, 10**6)}
    rows = [json.loads(line) for line in (RAW / f"translate_{model}.jsonl").read_text().splitlines()]
    return {r["id"]: (r["hyp"], refs[r["id"]]) for r in rows}


def bootstrap(models, corpus, seed=0):
    data = {m: load(m, corpus) for m in models}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    metric = CHRF(word_order=2)
    stats = {m: _stats([data[m][i][0] for i in ids], [data[m][i][1] for i in ids]) for m in models}
    rng = random.Random(seed)
    draws = [[rng.randrange(len(ids)) for _ in ids] for _ in range(RESAMPLES)]
    scores = {m: sorted(_score(metric, stats[m], d) for d in draws) for m in models}
    point = {m: _score(metric, stats[m], range(len(ids))) for m in models}
    lo, hi = int(0.025 * RESAMPLES), int(0.975 * RESAMPLES) - 1
    out = {"n": len(ids), "models": {m: {"chrf++": round(point[m], 2), "ci95": [round(scores[m][lo], 2), round(scores[m][hi], 2)]} for m in models}}
    diffs = {}
    for a in models:
        for b in models:
            if a < b:
                d = sorted(_score(metric, stats[a], x) - _score(metric, stats[b], x) for x in draws)
                diffs[f"{a} - {b}"] = {"diff": round(point[a] - point[b], 2), "ci95": [round(d[lo], 2), round(d[hi], 2)], "separable": d[lo] > 0 or d[hi] < 0}
    out["pairwise"] = diffs
    return out
