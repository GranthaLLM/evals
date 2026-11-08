from sacrebleu.metrics import CHRF


def chrf(hyps, refs):
    """Corpus chrF++ (word_order=2) and chrF, with sacreBLEU signatures so the variant is verifiable."""
    plus, plain = CHRF(word_order=2), CHRF(word_order=0)
    return {
        "chrf++": round(plus.corpus_score(hyps, [refs]).score, 2),
        "chrf": round(plain.corpus_score(hyps, [refs]).score, 2),
        "signature": str(plus.get_signature()),
        "n": len(hyps),
    }
