from grantha_evals import sandhi


def test_parse_takes_first_line_and_splits():
    assert sandhi.parse("विद्या + आलय\nextra") == ("विद्या", "आलय")


def test_score_exact_and_boundary():
    item = sandhi.Item("t", "विद्यालय", ("विद्या", "आलय"))
    assert sandhi.score_one(item, ("विद्या", "आलय"))["exact"]
    miss = sandhi.score_one(item, ("विद्यालय",))
    assert not miss["exact"] and miss["tp"] == 0 and miss["fn"] == 1
