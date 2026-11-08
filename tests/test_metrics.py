from grantha_evals import metrics


def test_chrf_identical_is_100_and_reports_word_order_two():
    out = metrics.chrf(["the king spoke"], ["the king spoke"])
    assert out["chrf++"] == 100.0 and "nw:2" in out["signature"]
