import numpy as np
import pandas as pd
import pytest

import ibmecstats as ibs


def test_ols_fit_and_predict():
    x = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [2, 1, 3, 2, 5]})
    y = 1.0 + 2.0 * x["x1"] - 0.5 * x["x2"]
    model = ibs.ols_fit(y, x, add_constant=True)
    pred = ibs.ols_predict(model, x)
    assert len(pred) == len(y)


def test_ols_diagnostics_and_dummies():
    df = pd.DataFrame(
        {
            "y": [10, 12, 13, 15, 14, 16],
            "x_num": [1, 2, 3, 4, 5, 6],
            "grupo": ["A", "A", "B", "B", "A", "B"],
        }
    )
    x = ibs.add_dummy_variables(df[["x_num", "grupo"]], columns=["grupo"], drop_first=True)
    model = ibs.ols_fit(df["y"], x)
    diag = ibs.ols_diagnostics(model)
    assert "r2" in diag and "aic" in diag


def test_model_selection_methods():
    rng = np.random.default_rng(42)
    n = 60
    x = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    y = 3.0 + 2.0 * x["x1"] - 1.0 * x["x2"] + rng.normal(scale=0.1, size=n)

    out_best = ibs.best_subset_selection(y, x, criterion="aic")
    assert len(out_best["features"]) >= 1

    out_fwd = ibs.model_selection(y, x, method="forward", criterion="aic")
    out_bwd = ibs.model_selection(y, x, method="backward", criterion="aic")
    out_stp = ibs.model_selection(y, x, method="stepwise", criterion="aic")
    assert len(out_fwd["features"]) >= 1
    assert len(out_bwd["features"]) >= 1
    assert len(out_stp["features"]) >= 1


def test_regressao_validation_errors():
    with pytest.raises(ValueError):
        ibs.add_dummy_variables(pd.DataFrame({"a": [1, 2]}), columns=["missing"])

    with pytest.raises(ValueError):
        ibs.best_subset_selection([1, 2, 3], pd.DataFrame({"x": [1, 2, 3]}), criterion="bad")

    with pytest.raises(ValueError):
        ibs.model_selection([1, 2, 3], pd.DataFrame({"x": [1, 2, 3]}), method="invalid")
