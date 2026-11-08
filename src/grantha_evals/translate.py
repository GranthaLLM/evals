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


def select(corpus_dir, works, per_work, seed=0):
    rng = random.Random(seed)
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
        for u in rng.sample(pool, min(per_work, len(pool))):
            items.append({"id": f"{name}:{u['ref']}", "work": name, "sanskrit": u["sanskrit"], "english": u["english"].strip()})
    return items


def prompt(sanskrit):
    return [{"role": "user", "content": INSTRUCTION + sanskrit}]
