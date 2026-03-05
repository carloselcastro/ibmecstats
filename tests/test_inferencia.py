import ibmecstats as ibs


def test_ci_mean():
    lo, hi = ibs.ci_mean([1, 2, 3, 4, 5], alpha=0.05, method="t")
    assert lo < hi


def test_t_test_1samp():
    out = ibs.t_test_1samp([1, 2, 3, 4, 5], mu0=3.0)
    assert "statistic" in out and "pvalue" in out
    assert 0.0 <= out["pvalue"] <= 1.0


def test_normality_tests():
    df = ibs.normality_tests([1, 2, 3, 4, 5, 6, 7])
    assert "test" in df.columns
    assert df.shape[0] >= 2
