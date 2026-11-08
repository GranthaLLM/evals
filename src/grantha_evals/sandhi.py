"""Sandhi splitting on SandhiKosh (Bhardwaj et al., LREC 2018), prompted as a generation task."""
import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SHEETS = [
    ("Astaadhyaayii Corpus.xls", None, "ashtadhyayi"),
    ("Bhagvad_Gita Corpus.xls", None, "gita"),
    ("Rule-based Corpus and Literature Corpus.xls", "Internal", "internal"),
    ("Rule-based Corpus and Literature Corpus.xls", "External", "external"),
    ("Rule-based Corpus and Literature Corpus.xls", "Literature", "literature"),
    ("UoH_Corpus.xls", None, "uoh"),
]
SHOTS = 5
SEED = 0


@dataclass(frozen=True)
class Item:
    tag: str
    word: str
    parts: tuple[str, ...]


def _norm(s):
    return unicodedata.normalize("NFC", s).strip()


def load(data_dir):
    items = []
    for f, sheet, tag in SHEETS:
        df = pd.read_excel(Path(data_dir) / f, sheet_name=sheet or 0, header=None, engine="xlrd")
        for _, r in df.iterrows():
            w, sp = r.get(1), r.get(2)
            if not isinstance(w, str) or not isinstance(sp, str) or "+" not in sp:
                continue
            w, parts = _norm(w), tuple(_norm(c) for c in sp.split("+"))
            if " " in w or len(w) < 3 or len(parts) < 2 or any(not c for c in parts):
                continue
            items.append(Item(tag, w, parts))
    return items


def split_demos_and_test(items, per_tag, seed=SEED):
    rng = random.Random(seed)
    pool = [i for i in items if i.tag == "uoh"]
    demos = rng.sample(pool, SHOTS)
    held = set(demos)
    by_tag = defaultdict(list)
    for i in items:
        if i not in held:
            by_tag[i.tag].append(i)
    test = []
    for tag in sorted(by_tag):
        xs = by_tag[tag]
        test += rng.sample(xs, min(per_tag, len(xs)))
    return demos, test


def prompt(demos, word):
    shots = "\n".join(f"{d.word} => {' + '.join(d.parts)}" for d in demos)
    return [
        {
            "role": "user",
            "content": "Split each Sanskrit word at its sandhi and compound junctions into its component words "
            "in unjoined Devanagari, separated by ' + '. Answer with the split only.\n\n"
            f"{shots}\n{word} =>",
        }
    ]


def parse(reply):
    line = _norm(reply.splitlines()[0]) if reply.strip() else ""
    line = line.split("=>")[-1]
    return tuple(c for c in (_norm(c) for c in line.split("+")) if c)


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cuts(word, parts):
    pos, out = 0, set()
    for c in parts[:-1]:
        pos += _lcp(word[pos:], c)
        if 0 < pos < len(word):
            out.add(pos)
    return out


def score_one(item, pred):
    """Exact component match plus boundary tp/fp/fn. A gold cut counts as found within one letter, as in the tokenizer eval."""
    gold, got = cuts(item.word, item.parts), cuts(item.word, pred) if len(pred) > 1 else set()
    found = {g for g in gold if g in got or g + 1 in got}
    near = gold | {g + 1 for g in gold}
    return {
        "exact": pred == item.parts,
        "tp": len(found),
        "fn": len(gold) - len(found),
        "fp": len([c for c in got if c not in near]),
    }


def aggregate(rows):
    def f1(rs):
        tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {
            "n": len(rs),
            "exact": round(sum(r["exact"] for r in rs) / len(rs), 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4),
        }

    by = defaultdict(list)
    for r in rows:
        by[r["tag"]].append(r)
    return {"all": f1(rows), **{t: f1(v) for t, v in sorted(by.items())}}
