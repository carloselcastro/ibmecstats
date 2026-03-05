from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .utils import ForecastResult, as_series


def _make_index_like(y: pd.Series, h: int) -> pd.Index:
    """
    Cria um índice para o horizonte h, preservando DatetimeIndex com freq se possível.
    """
    if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is not None:
        start = y.index[-1] + y.index.freq
        return pd.date_range(start=start, periods=h, freq=y.index.freq)
    # fallback: RangeIndex
    return pd.RangeIndex(start=len(y), stop=len(y) + h, step=1)


def naive_forecast(y, h: int, seasonal_period: Optional[int] = None) -> ForecastResult:
    """
    Previsão ingênua:
      - sem sazonalidade: repete o último valor
      - com sazonalidade (m): repete o valor de t-m
    """
    s = as_series(y, name="y").astype(float).dropna()
    if h < 1:
        raise ValueError("h deve ser >= 1.")

    if seasonal_period is None:
        last = float(s.iloc[-1])
        yhat = np.repeat(last, h)
    else:
        m = int(seasonal_period)
        if m < 1:
            raise ValueError("seasonal_period deve ser >= 1.")
        if len(s) < m:
            raise ValueError("Série muito curta para seasonal_period informado.")
        season_vals = s.iloc[-m:].values
        reps = int(np.ceil(h / m))
        yhat = np.tile(season_vals, reps)[:h]

    idx = _make_index_like(s, h)
    yhat_s = pd.Series(yhat, index=idx, name="yhat")

    fitted = s.shift(1) if seasonal_period is None else s.shift(seasonal_period)
    residuals = s - fitted
    return ForecastResult(yhat=yhat_s, fitted=fitted, residuals=residuals, info={"model": "naive", "seasonal_period": seasonal_period})


def drift_forecast(y, h: int) -> ForecastResult:
    """
    Previsão por drift (tendência linear entre primeiro e último ponto).
    """
    s = as_series(y, name="y").astype(float).dropna()
    if h < 1:
        raise ValueError("h deve ser >= 1.")
    if len(s) < 2:
        raise ValueError("Série muito curta para drift.")

    y0 = float(s.iloc[0])
    yT = float(s.iloc[-1])
    T = len(s) - 1
    slope = (yT - y0) / T
    steps = np.arange(1, h + 1)
    yhat = yT + slope * steps

    idx = _make_index_like(s, h)
    yhat_s = pd.Series(yhat, index=idx, name="yhat")
    return ForecastResult(yhat=yhat_s, info={"model": "drift", "slope": float(slope)})


def moving_average_forecast(y, h: int, window: int = 3) -> ForecastResult:
    """
    Previsão via média móvel simples (usa a média dos últimos 'window' valores e repete no horizonte).
    """
    s = as_series(y, name="y").astype(float).dropna()
    if h < 1:
        raise ValueError("h deve ser >= 1.")
    w = int(window)
    if w < 1:
        raise ValueError("window deve ser >= 1.")
    if len(s) < w:
        raise ValueError("Série muito curta para window informado.")

    level = float(s.iloc[-w:].mean())
    yhat = np.repeat(level, h)
    idx = _make_index_like(s, h)
    yhat_s = pd.Series(yhat, index=idx, name="yhat")

    fitted = s.rolling(w).mean().shift(1)
    residuals = s - fitted
    return ForecastResult(yhat=yhat_s, fitted=fitted, residuals=residuals, info={"model": "moving_average", "window": w})


# -----------------------------
# Suavização Exponencial (statsmodels)
# -----------------------------
def ses_forecast(y, h: int, alpha: Optional[float] = None, optimized: bool = True) -> ForecastResult:
    """
    Simple Exponential Smoothing (SES).
    Usa statsmodels se disponível.

    alpha:
      - None: otimiza (se optimized=True)
      - float em (0,1): fixa alpha
    """
    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    except Exception as e:
        raise ImportError("Para ses_forecast, instale statsmodels: pip install statsmodels") from e

    s = as_series(y, name="y").astype(float).dropna()
    if h < 1:
        raise ValueError("h deve ser >= 1.")

    model = SimpleExpSmoothing(s, initialization_method="estimated")
    fit = model.fit(smoothing_level=alpha, optimized=optimized)

    idx = _make_index_like(s, h)
    yhat = fit.forecast(h)
    yhat.index = idx
    fitted = fit.fittedvalues
    residuals = s - fitted

    return ForecastResult(
        yhat=yhat.rename("yhat"),
        fitted=fitted.rename("fitted"),
        residuals=residuals.rename("residuals"),
        info={"model": "SES", "params": dict(fit.params)},
    )


def holt_forecast(
    y,
    h: int,
    damped_trend: bool = False,
    optimized: bool = True,
    smoothing_level: Optional[float] = None,
    smoothing_trend: Optional[float] = None,
) -> ForecastResult:
    """
    Holt (nível + tendência), com opção de damped_trend.
    """
    try:
        from statsmodels.tsa.holtwinters import Holt
    except Exception as e:
        raise ImportError("Para holt_forecast, instale statsmodels: pip install statsmodels") from e

    s = as_series(y, name="y").astype(float).dropna()
    if h < 1:
        raise ValueError("h deve ser >= 1.")

    model = Holt(s, damped_trend=damped_trend, initialization_method="estimated")
    fit = model.fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        optimized=optimized,
    )

    idx = _make_index_like(s, h)
    yhat = fit.forecast(h)
    yhat.index = idx

    fitted = fit.fittedvalues
    residuals = s - fitted

    return ForecastResult(
        yhat=yhat.rename("yhat"),
        fitted=fitted.rename("fitted"),
        residuals=residuals.rename("residuals"),
        info={"model": "Holt", "params": dict(fit.params), "damped_trend": damped_trend},
    )


def holt_winters_forecast(
    y,
    h: int,
    seasonal_periods: int,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = "add",
    damped_trend: bool = False,
    optimized: bool = True,
) -> ForecastResult:
    """
    Holt-Winters (Exponential Smoothing) com tendência e sazonalidade.

    trend: None, 'add', 'mul'
    seasonal: None, 'add', 'mul'
    seasonal_periods: período sazonal (ex.: 12 mensal, 7 diário semanal, etc.)
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception as e:
        raise ImportError("Para holt_winters_forecast, instale statsmodels: pip install statsmodels") from e

    s = as_series(y, name="y").astype(float).dropna()
    if h < 1:
        raise ValueError("h deve ser >= 1.")
    m = int(seasonal_periods)
    if m < 2:
        raise ValueError("seasonal_periods deve ser >= 2.")
    if len(s) < 2 * m:
        # regra prática para estabilizar inicialização sazonal
        raise ValueError("Série curta para Holt-Winters (recomendado >= 2*seasonal_periods).")

    model = ExponentialSmoothing(
        s,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=m,
        damped_trend=damped_trend,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=optimized)

    idx = _make_index_like(s, h)
    yhat = fit.forecast(h)
    yhat.index = idx

    fitted = fit.fittedvalues
    residuals = s - fitted

    return ForecastResult(
        yhat=yhat.rename("yhat"),
        fitted=fitted.rename("fitted"),
        residuals=residuals.rename("residuals"),
        info={
            "model": "Holt-Winters",
            "params": dict(fit.params),
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": m,
            "damped_trend": damped_trend,
        },
    )