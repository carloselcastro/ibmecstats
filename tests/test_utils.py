import numpy as np
import pandas as pd
import pytest

import ibmecstats as ibs
from ibmecstats.utils import as_series, dropna_series


def test_as_series_and_dropna_series():
    s = as_series([1, 2, 3], name="v")
    assert isinstance(s, pd.Series)
    assert s.name == "v"

    s2 = dropna_series([1, np.nan, 2], name="v")
    assert len(s2) == 2


def test_as_series_invalid_ndim():
    with pytest.raises(ValueError):
        as_series([[1, 2], [3, 4]])


def test_ensure_datetime_index_from_col_and_freq():
    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "y": [1, 2, 3],
        }
    )
    out = ibs.ensure_datetime_index(df, date_col="date", freq="D")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.freqstr == "D"


def test_ensure_datetime_index_invalid():
    with pytest.raises(ValueError):
        ibs.ensure_datetime_index(pd.DataFrame({"y": [1, 2, 3]}))


def test_train_test_split_time_float_and_int():
    train, test = ibs.train_test_split_time([1, 2, 3, 4, 5], test_size=0.4)
    assert len(train) == 3 and len(test) == 2

    train2, test2 = ibs.train_test_split_time([1, 2, 3, 4, 5], test_size=2)
    assert len(train2) == 3 and len(test2) == 2


def test_train_test_split_time_invalid():
    with pytest.raises(ValueError):
        ibs.train_test_split_time([1], test_size=0.5)
    with pytest.raises(ValueError):
        ibs.train_test_split_time([1, 2, 3], test_size=1.0)
    with pytest.raises(ValueError):
        ibs.train_test_split_time([1, 2, 3], test_size=3)
