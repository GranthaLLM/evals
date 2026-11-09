import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import judge, metrics, models, report, sandhi, similarity, translate
from .client import cached, complete
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
    out = report.bootstrap(a.models, a.corpus, a.per_work)
    (OUT / "translate_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def run_progress(a):
    items = translate.select(a.corpus, translate.PILOT_WORKS, a.per_work)
    for name in a.models:
        m = models.load(name)
        done = sum(cached(m, translate.prompt(i["sanskrit"])) is not None for i in items)
        print(f"{name}: {done}/{len(items)} ({100 * done // len(items)}%)")


def run_similarity(a):
    out = similarity.bootstrap(a.models, a.corpus, a.per_work)
    (OUT / "similarity_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def run_judge(a):
    model = models.load(a.model)
    items = json.loads((a.items / "test_items.json").read_text())
    key = json.loads((a.items / "test_key.json").read_text())
    out = {"model": a.model}
    for mode, flag in (("without_commentary", False), ("with_commentary", True)):
        rows = _map(lambda it: judge.run(model, it, flag), items, a.workers)
        out[mode] = judge.planted_check(rows, key)
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
        (OUT / "raw" / f"judge_{a.model}_{mode}.json").write_text(json.dumps(rows, ensure_ascii=False))
    (OUT / f"judge_{a.model}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def run_judge_candidates(a):
    judge_model = models.load(a.judge)
    pilot = json.loads((a.items / "pilot_bsb_madhva.json").read_text())
    items = judge.candidate_items(pilot, a.n)
    out = {"judge": a.judge, "n": len(items)}
    for cand in a.candidates:
        cm = models.load(cand)
        hyps = _map(lambda it: complete(cm, translate.prompt(it["sanskrit"]), max_tokens=1500), items, a.workers)
        scored = {}
        for mode, flag in (("without_commentary", False), ("with_commentary", True)):
            rows = _map(lambda p: judge.run(judge_model, {**p[0], "english": p[1]}, flag), list(zip(items, hyps)), a.workers)
            pens = [r["penalty"] for r in rows if r["penalty"] is not None]
            scored[mode] = {"mean_penalty": round(sum(pens) / max(len(pens), 1), 2), "scored": len(pens), "per_item": {r["id"]: r["penalty"] for r in rows}}
        out[cand] = scored
        print(cand, {m: scored[m]["mean_penalty"] for m in scored}, flush=True)
    (OUT / f"judge_candidates_{a.judge}.json").write_text(json.dumps(out, indent=1))


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
    r.add_argument("--per-work", type=int, default=150)
    r.set_defaults(fn=run_report)
    q = sub.add_parser("similarity")
    q.add_argument("models", nargs="+", choices=models.names())
    q.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    q.add_argument("--per-work", type=int, default=150)
    q.set_defaults(fn=run_similarity)
    j = sub.add_parser("judge")
    j.add_argument("model", choices=models.names())
    j.add_argument("--items", type=Path, default=ROOT.parent / "judge")
    j.add_argument("--workers", type=int, default=2)
    j.set_defaults(fn=run_judge)
    c = sub.add_parser("judge-candidates")
    c.add_argument("judge", choices=models.names())
    c.add_argument("candidates", nargs="+", choices=models.names())
    c.add_argument("--n", type=int, default=30)
    c.add_argument("--items", type=Path, default=ROOT.parent / "judge")
    c.add_argument("--workers", type=int, default=2)
    c.set_defaults(fn=run_judge_candidates)
    g = sub.add_parser("progress")
    g.add_argument("models", nargs="+", choices=models.names())
    g.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    g.add_argument("--per-work", type=int, default=150)
    g.set_defaults(fn=run_progress)
    a = ap.parse_args()
    a.fn(a)
