import pandas as pd
import pytest

import ibmecstats as ibs


class _FakeFit:
    def __init__(self, s):
        self._s = s
        self.fittedvalues = pd.Series([float(s.mean())] * len(s), index=s.index, name="fitted")
        self.params = {"alpha": 0.3}

    def forecast(self, h):
        return pd.Series([float(self._s.iloc[-1])] * h)


class _FakeSES:
    def __init__(self, s, initialization_method=None):
        self._s = s

    def fit(self, smoothing_level=None, optimized=True):
        return _FakeFit(self._s)


class _FakeHolt:
    def __init__(self, s, damped_trend=False, initialization_method=None):
        self._s = s

    def fit(self, smoothing_level=None, smoothing_trend=None, optimized=True):
        return _FakeFit(self._s)


class _FakeHW:
    def __init__(
        self,
        s,
        trend=None,
        seasonal=None,
        seasonal_periods=None,
        damped_trend=False,
        initialization_method=None,
    ):
        self._s = s

    def fit(self, optimized=True):
        return _FakeFit(self._s)


def test_statsmodels_forecast_wrappers(monkeypatch):
    import statsmodels.tsa.holtwinters as hw

    monkeypatch.setattr(hw, "SimpleExpSmoothing", _FakeSES)
    monkeypatch.setattr(hw, "Holt", _FakeHolt)
    monkeypatch.setattr(hw, "ExponentialSmoothing", _FakeHW)

    y = [10, 11, 12, 13, 14, 15, 16, 17]
    out_ses = ibs.ses_forecast(y, h=3, alpha=0.2, optimized=False)
    out_holt = ibs.holt_forecast(y, h=2, damped_trend=True)
    out_hw = ibs.holt_winters_forecast(y + y, h=2, seasonal_periods=4, trend="add", seasonal="add")

    assert len(out_ses.yhat) == 3
    assert len(out_holt.yhat) == 2
    assert len(out_hw.yhat) == 2


def test_statsmodels_forecast_validation_errors():
    y = [10, 11, 12, 13]

    with pytest.raises(ValueError):
        ibs.ses_forecast(y, h=0)
    with pytest.raises(ValueError):
        ibs.holt_forecast(y, h=0)
    with pytest.raises(ValueError):
        ibs.holt_winters_forecast(y, h=1, seasonal_periods=1)
    with pytest.raises(ValueError):
        ibs.holt_winters_forecast(y, h=1, seasonal_periods=3)
