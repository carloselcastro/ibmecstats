from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import as_series


def _align(y_true, y_pred):
    yt = as_series(y_true, name="y_true").astype(float)
    yp = as_series(y_pred, name="y_pred").astype(float)
    # alinha por índice, se houver
    df = pd.concat([yt, yp], axis=1).dropna()
    return df.iloc[:, 0].values, df.iloc[:, 1].values


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-9) -> float:
    yt, yp = _align(y_true, y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def smape(y_true, y_pred, eps: float = 1e-9) -> float:
    yt, yp = _align(y_true, y_pred)
    denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, eps)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def wape(y_true, y_pred, eps: float = 1e-9) -> float:
    yt, yp = _align(y_true, y_pred)
    denom = np.maximum(np.sum(np.abs(yt)), eps)
    return float(np.sum(np.abs(yt - yp)) / denom * 100.0)


def mase(
    y_true,
    y_pred,
    y_train,
    seasonal_period: int = 1,
    eps: float = 1e-9,
) -> float:
    """
    Mean Absolute Scaled Error (MASE).

    Escala pelo MAE do método sazonal ingênuo no treino.
    """
    yt, yp = _align(y_true, y_pred)
    ytr = as_series(y_train, name="y_train").astype(float).dropna().values

    m = int(seasonal_period)
    if m < 1:
        raise ValueError("seasonal_period deve ser >= 1.")
    if len(ytr) <= m:
        raise ValueError("y_train muito curto para seasonal_period informado.")

    naive = ytr[m:]
    lagged = ytr[:-m]
    scale = np.mean(np.abs(naive - lagged))
    scale = max(scale, eps)

    return float(np.mean(np.abs(yt - yp)) / scale)


def forecast_accuracy(
    y_true,
    y_pred,
    y_train: object | None = None,
    seasonal_period: int = 1,
) -> pd.DataFrame:
    """
    Tabela-resumo de métricas comuns. Se y_train for fornecido, calcula MASE.
    """
    out = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
        "sMAPE(%)": smape(y_true, y_pred),
        "WAPE(%)": wape(y_true, y_pred),
    }
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train=y_train, seasonal_period=seasonal_period)

    return pd.DataFrame([out])
