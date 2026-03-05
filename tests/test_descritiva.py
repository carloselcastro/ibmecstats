import ibmecstats as ibs


def test_summary_stats_basic():
    s = ibs.summary_stats([1, 2, 3, 4, 5])
    assert "n" in s
    assert int(s["n"]) == 5
    assert "mean" in s


def test_freq_table():
    df = ibs.freq_table(["A", "A", "B"], normalize=True)
    assert "count" in df.columns
    assert "proportion" in df.columns
    assert df.loc["A", "count"] == 2
