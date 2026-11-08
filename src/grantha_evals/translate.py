"""Sanskrit to English on corpus works that carry a translation of the verse itself, scored by chrF++ against it."""
import json
import random
from pathlib import Path

PILOT_WORKS = [
    "sahitya/vidura_niti",
    "sahitya/sanatsujatiya",
    "sahitya/yaksha_prashna",
    "sahitya/kumarasambhava",
    "sahitya/shatakatraya",
    "philosophical/samkhyakarika",
    "philosophical/vimshatika_trimshika",
    "philosophical/tarkasamgraha",
    "philosophical/sarvadarshanasamgraha",
]
MAX_CHARS = 600
INSTRUCTION = "Translate the following Sanskrit text into English. Output only the translation.\n\n"


PILOT_SIZE = 10


def _prefix_stable(pool, n, seed):
    """The first PILOT_SIZE units are the original pilot sample; larger n only appends, so cached responses stay valid."""
    head = random.Random(seed).sample(pool, min(PILOT_SIZE, len(pool)))
    rest = [u for u in pool if u not in head]
    random.Random(seed + 1).shuffle(rest)
    return (head + rest)[:n]


def select(corpus_dir, works, per_work, seed=0):
    items = []
    for w in works:
        name = w.split("/")[-1]
        data = json.loads((Path(corpus_dir) / f"{w}.json").read_text())
        pool = [
            u for u in data["units"]
            if u["scope"] in ("mula", "text")
            and u.get("english", "").strip()
            and 10 < len(u["sanskrit"]) <= MAX_CHARS
            and "-" not in u["ref"]
        ]
        for u in _prefix_stable(pool, per_work, seed):
            items.append({"id": f"{name}:{u['ref']}", "work": name, "sanskrit": u["sanskrit"], "english": u["english"].strip()})
    return items


def prompt(sanskrit):
    return [{"role": "user", "content": INSTRUCTION + sanskrit}]
