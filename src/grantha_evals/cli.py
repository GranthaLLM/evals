import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import metrics, models, report, sandhi, translate
from .client import complete
from .models import ROOT

DEFAULT_CORPUS = ROOT.parent / "corpus"
DEFAULT_SANDHIKOSH = ROOT.parent / "tokenizer" / "data" / "SandhiKosh"
OUT = ROOT / "results"


def _map(fn, xs, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, xs))


def run_sandhi(a):
    model = models.load(a.model)
    demos, test = sandhi.split_demos_and_test(sandhi.load(a.data), a.per_tag)

    def one(item):
        reply = complete(model, sandhi.prompt(demos, item.word), max_tokens=48)
        pred = sandhi.parse(reply)
        return {"tag": item.tag, "word": item.word, "gold": list(item.parts), "pred": list(pred), **sandhi.score_one(item, pred)}

    rows = _map(one, test, a.workers)
    OUT.mkdir(exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    (OUT / "raw" / f"sandhi_{a.model}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    summary = {"model": a.model, "shots": sandhi.SHOTS, "per_tag": a.per_tag, **sandhi.aggregate(rows)}
    (OUT / f"sandhi_{a.model}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["all"]), a.model)


def run_translate(a):
    model = models.load(a.model)
    works = a.works or translate.PILOT_WORKS
    items = translate.select(a.corpus, works, a.per_work)

    def one(item):
        return complete(model, translate.prompt(item["sanskrit"]), max_tokens=1024)

    hyps = _map(one, items, a.workers)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "raw" / f"translate_{a.model}.jsonl").write_text(
        "\n".join(json.dumps({"id": i["id"], "work": i["work"], "hyp": h}, ensure_ascii=False) for i, h in zip(items, hyps))
    )
    refs = [i["english"] for i in items]
    summary = {"model": a.model, "per_work": a.per_work, **metrics.chrf(hyps, refs)}
    by = {}
    for w in sorted({i["work"] for i in items}):
        idx = [k for k, i in enumerate(items) if i["work"] == w]
        by[w] = metrics.chrf([hyps[k] for k in idx], [refs[k] for k in idx])["chrf++"]
    summary["by_work"] = by
    (OUT / f"translate_{a.model}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("chrf++", "chrf", "n")}), a.model)


def run_report(a):
    out = report.bootstrap(a.models, a.corpus)
    (OUT / "translate_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(prog="grantha-evals")
    sub = ap.add_subparsers(required=True)
    s = sub.add_parser("sandhi")
    s.add_argument("model", choices=models.names())
    s.add_argument("--data", type=Path, default=DEFAULT_SANDHIKOSH)
    s.add_argument("--per-tag", type=int, default=100)
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=run_sandhi)
    t = sub.add_parser("translate")
    t.add_argument("model", choices=models.names())
    t.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    t.add_argument("--works", nargs="*")
    t.add_argument("--per-work", type=int, default=10)
    t.add_argument("--workers", type=int, default=8)
    t.set_defaults(fn=run_translate)
    r = sub.add_parser("report")
    r.add_argument("models", nargs="+", choices=models.names())
    r.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    r.set_defaults(fn=run_report)
    a = ap.parse_args()
    a.fn(a)
