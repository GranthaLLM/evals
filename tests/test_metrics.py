from grantha_evals import metrics


def test_chrf_identical_is_100_and_reports_word_order_two():
    out = metrics.chrf(["the king spoke"], ["the king spoke"])
    assert out["chrf++"] == 100.0 and "nw:2" in out["signature"]


def test_larger_translation_sample_extends_the_pilot():
    from grantha_evals import translate

    pool = [{"ref": str(i)} for i in range(200)]
    small = translate._prefix_stable(pool, 10, 0)
    assert translate._prefix_stable(pool, 50, 0)[:10] == small
