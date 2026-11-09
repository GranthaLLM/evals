"""GEMBA-MQM style judge: the model lists translation errors with a category and severity, and the score is the MQM penalty."""
import json
import re

from .client import complete

WEIGHTS = {"minor": 1, "major": 5, "critical": 25}
CATEGORIES = "accuracy, omission, addition, terminology, fluency, other"
COMMENTARY_CHARS = 14000
SYSTEM = (
    "You are a strict annotator of Sanskrit to English translation quality, following the MQM framework. "
    "Find every error in the English translation of the Sanskrit source. For each error give the offending English span, "
    f"a category ({CATEGORIES}), and a severity: minor (small flaw), major (changes or loses meaning), "
    "critical (wrong or unrelated content, or most of the passage missing). "
    'Reply with JSON only: {"errors": [{"span": "...", "category": "...", "severity": "..."}]}. '
    'If the translation is correct reply {"errors": []}.'
)


def prompt(item, with_commentary):
    parts = [f"Sanskrit source:\n{item['sanskrit']}"]
    if with_commentary:
        parts.append(
            "Traditional commentary on this passage (Bhashya, then Dipika, in IAST); use it to decide what the source means:\n"
            + (item["gretil_bhashya_iast"] + "\n" + item["dipika_iast"])[:COMMENTARY_CHARS]
        )
    parts.append(f"English translation to judge:\n{item['english']}")
    return [{"role": "user", "content": SYSTEM + "\n\n" + "\n\n".join(parts)}]


def parse(reply):
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        errors = json.loads(m.group(0)).get("errors", [])
    except json.JSONDecodeError:
        return None
    return [e for e in errors if isinstance(e, dict) and e.get("severity") in WEIGHTS]


def penalty(errors):
    return sum(WEIGHTS[e["severity"]] for e in errors)


def run(model, item, with_commentary):
    errors = parse(complete(model, prompt(item, with_commentary), max_tokens=1500))
    return {"id": item["id"], "errors": errors, "penalty": None if errors is None else penalty(errors)}


def planted_check(rows, key):
    """Does the judge separate clean translations from planted errors? Uses the labelled planted-error set."""
    clean = [r["penalty"] for r in rows if key[r["id"]]["planted"].startswith("clean") and r["penalty"] is not None]
    bad = [r["penalty"] for r in rows if not key[r["id"]]["planted"].startswith("clean") and r["penalty"] is not None]
    wins = sum((b > c) + 0.5 * (b == c) for b in bad for c in clean)
    return {
        "n_clean": len(clean),
        "n_planted": len(bad),
        "mean_penalty_clean": round(sum(clean) / max(len(clean), 1), 2),
        "mean_penalty_planted": round(sum(bad) / max(len(bad), 1), 2),
        "auc": round(wins / max(len(clean) * len(bad), 1), 3),
        "unparsed": sum(r["penalty"] is None for r in rows),
    }
