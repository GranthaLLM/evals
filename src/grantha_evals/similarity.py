"""Semantic similarity between a model translation and the English reference, English against English on CPU."""
import random

import numpy as np
from fastembed import TextEmbedding

from .report import RESAMPLES, load

EMBEDDER = "BAAI/bge-base-en-v1.5"


def _cosines(pairs):
    embed = TextEmbedding(EMBEDDER)
    hyp = np.array(list(embed.embed([h for h, _ in pairs])))
    ref = np.array(list(embed.embed([r for _, r in pairs])))
    return (hyp * ref).sum(1) / (np.linalg.norm(hyp, axis=1) * np.linalg.norm(ref, axis=1))


def bootstrap(names, corpus, per_work, seed=0):
    data = {m: load(m, corpus, per_work) for m in names}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    sims = {m: _cosines([data[m][i] for i in ids]) for m in names}
    rng = random.Random(seed)
    draws = np.array([[rng.randrange(len(ids)) for _ in ids] for _ in range(RESAMPLES)])
    lo, hi = int(0.025 * RESAMPLES), int(0.975 * RESAMPLES) - 1

    def ci(v):
        means = np.sort(v[draws].mean(1))
        return [round(float(means[lo]), 4), round(float(means[hi]), 4)]

    out = {"embedder": EMBEDDER, "n": len(ids), "models": {m: {"cosine": round(float(sims[m].mean()), 4), "ci95": ci(sims[m])} for m in names}, "pairwise": {}}
    for a in names:
        for b in names:
            if a < b:
                d = sims[a] - sims[b]
                c = ci(d)
                out["pairwise"][f"{a} - {b}"] = {"diff": round(float(d.mean()), 4), "ci95": c, "separable": c[0] > 0 or c[1] < 0}
    return out
