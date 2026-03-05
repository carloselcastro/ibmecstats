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
