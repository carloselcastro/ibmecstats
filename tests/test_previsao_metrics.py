import pandas as pd
import pytest

import ibmecstats as ibs


def test_naive_forecast_len():
    y = [10, 11, 12, 13]
    train, test = ibs.train_test_split_time(y, test_size=1)
    res = ibs.naive_forecast(train, h=len(test))
    assert len(res.yhat) == len(test)


def test_metrics():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 4]
    assert ibs.mae(y_true, y_pred) >= 0
    assert ibs.rmse(y_true, y_pred) >= 0
    assert ibs.mse(y_true, y_pred) >= 0
    assert ibs.mape(y_true, y_pred) >= 0
    assert ibs.smape(y_true, y_pred) >= 0
    assert ibs.wape(y_true, y_pred) >= 0

    m = ibs.mase(y_true, y_pred, y_train=[1, 2, 3, 4], seasonal_period=1)
    assert m >= 0
    df = ibs.forecast_accuracy(y_true, y_pred, y_train=[1, 2, 3, 4], seasonal_period=1)
    assert "MASE" in df.columns


def test_trend_projection_forecast_len():
    y = [10, 11, 13, 14, 16, 18]
    res = ibs.trend_projection_forecast(y, h=3, model="linear")
    assert len(res.yhat) == 3


def test_autoregressive_forecast_len():
    y = [10, 11, 12, 13, 12, 14, 15, 16, 15, 17]
    res = ibs.autoregressive_forecast(y, h=2, lags=2)
    assert len(res.yhat) == 2


def test_previsao_additional_branches():
    y = [10, 11, 12, 13, 14, 15, 16, 17]
    res_seasonal = ibs.naive_forecast(y, h=4, seasonal_period=4)
    assert len(res_seasonal.yhat) == 4

    res_ma = ibs.moving_average_forecast(y, h=3, window=3)
    assert len(res_ma.yhat) == 3

    res_q = ibs.trend_projection_forecast(y, h=2, model="quadratic")
    assert len(res_q.yhat) == 2

    y_pos = [2, 3, 4, 6, 9, 13]
    res_e = ibs.trend_projection_forecast(y_pos, h=2, model="exponential")
    assert len(res_e.yhat) == 2

    res_ar_noconst = ibs.autoregressive_forecast(y, h=2, lags=2, trend="n")
    assert len(res_ar_noconst.yhat) == 2

    y_dt = pd.Series(
        [10, 11, 12, 13],
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )
    res_dt = ibs.naive_forecast(y_dt, h=2)
    assert isinstance(res_dt.yhat.index, pd.DatetimeIndex)


def test_previsao_validation_errors():
    with pytest.raises(ValueError):
        ibs.naive_forecast([1, 2, 3], h=0)
    with pytest.raises(ValueError):
        ibs.naive_forecast([1, 2], h=1, seasonal_period=3)

    with pytest.raises(ValueError):
        ibs.moving_average_forecast([1, 2], h=1, window=3)
    with pytest.raises(ValueError):
        ibs.trend_projection_forecast([1, 2], h=1, model="linear")
    with pytest.raises(ValueError):
        ibs.trend_projection_forecast([1, 2, 3], h=1, model="bad")
    with pytest.raises(ValueError):
        ibs.trend_projection_forecast([1, 2, 0], h=1, model="exponential")

    with pytest.raises(NotImplementedError):
        ibs.autoregressive_forecast([1, 2, 3, 4, 5], h=1, lags=1, seasonal=True)
    with pytest.raises(NotImplementedError):
        ibs.autoregressive_forecast([1, 2, 3, 4, 5], h=1, lags=1, period=12)
    with pytest.raises(ValueError):
        ibs.autoregressive_forecast([1, 2, 3], h=1, lags=3)

    with pytest.raises(ValueError):
        ibs.mase([1, 2], [1, 2], y_train=[1], seasonal_period=1)
    with pytest.raises(ValueError):
        ibs.mase([1, 2], [1, 2], y_train=[1, 2, 3], seasonal_period=0)
