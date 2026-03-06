import ibmecstats as ibs
import pandas as pd
import pytest


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


def test_iqr_outliers_and_corr_matrix():
    out = ibs.iqr_outliers([1, 2, 3, 4, 100], k=1.5)
    assert "is_outlier" in out.columns
    assert out["is_outlier"].sum() >= 1

    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    corr = ibs.correlation_matrix(df, method="pearson")
    assert corr.shape == (2, 2)


def test_correlation_matrix_validation():
    with pytest.raises(ValueError):
        ibs.correlation_matrix(pd.DataFrame({"cat": ["a", "b", "c"]}))
