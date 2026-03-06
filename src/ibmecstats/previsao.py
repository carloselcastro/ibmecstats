from __future__ import annotations

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


def naive_forecast(y, h: int, seasonal_period: int | None = None) -> ForecastResult:
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
    return ForecastResult(
        yhat=yhat_s,
        fitted=fitted,
        residuals=residuals,
        info={"model": "naive", "seasonal_period": seasonal_period},
    )


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
    Previsão via média móvel simples.

    Usa a média dos últimos `window` valores e repete no horizonte.
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
    return ForecastResult(
        yhat=yhat_s,
        fitted=fitted,
        residuals=residuals,
        info={"model": "moving_average", "window": w},
    )


def trend_projection_forecast(y, h: int, model: str = "linear") -> ForecastResult:
    """
    Previsao por tendencia via minimos quadrados.

    model:
      - 'linear': y_t = b0 + b1*t
      - 'quadratic': y_t = b0 + b1*t + b2*t^2
      - 'exponential': y_t = exp(b0 + b1*t), requer y > 0
    """
    s = as_series(y, name="y").astype(float).dropna()
    if h < 1:
        raise ValueError("h deve ser >= 1.")
    if len(s) < 3:
        raise ValueError("Serie muito curta para projecao de tendencia.")

    t = np.arange(1, len(s) + 1, dtype=float)
    m = model.lower()

    def _linear_fit(tt: np.ndarray, yy: np.ndarray) -> tuple[float, float]:
        t_mean = float(np.mean(tt))
        y_mean = float(np.mean(yy))
        denom = float(np.sum((tt - t_mean) ** 2))
        if denom == 0:
            raise ValueError("Nao foi possivel ajustar tendencia linear.")
        b1 = float(np.sum((tt - t_mean) * (yy - y_mean)) / denom)
        b0 = y_mean - b1 * t_mean
        return b0, b1

    def _det3(
        a11: float,
        a12: float,
        a13: float,
        a21: float,
        a22: float,
        a23: float,
        a31: float,
        a32: float,
        a33: float,
    ) -> float:
        return (
            a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31)
        )

    if m == "linear":
        b0, b1 = _linear_fit(t, s.values)
        fitted_vals = b0 + b1 * t
        tf = np.arange(len(s) + 1, len(s) + h + 1, dtype=float)
        yhat = b0 + b1 * tf
    elif m == "quadratic":
        n = float(len(s))
        y = s.values
        t1 = float(np.sum(t))
        t2 = float(np.sum(t**2))
        t3 = float(np.sum(t**3))
        t4 = float(np.sum(t**4))
        y0 = float(np.sum(y))
        y1 = float(np.sum(t * y))
        y2 = float(np.sum((t**2) * y))

        det_a = _det3(n, t1, t2, t1, t2, t3, t2, t3, t4)
        if abs(det_a) < 1e-12:
            raise ValueError("Nao foi possivel ajustar tendencia quadratica.")

        det_b0 = _det3(y0, t1, t2, y1, t2, t3, y2, t3, t4)
        det_b1 = _det3(n, y0, t2, t1, y1, t3, t2, y2, t4)
        det_b2 = _det3(n, t1, y0, t1, t2, y1, t2, t3, y2)
        b0 = det_b0 / det_a
        b1 = det_b1 / det_a
        b2 = det_b2 / det_a

        fitted_vals = b0 + b1 * t + b2 * (t**2)
        tf = np.arange(len(s) + 1, len(s) + h + 1, dtype=float)
        yhat = b0 + b1 * tf + b2 * (tf**2)
    elif m == "exponential":
        if np.any(s.values <= 0):
            raise ValueError("Para model='exponential', todos os valores de y devem ser positivos.")
        b0, b1 = _linear_fit(t, np.log(s.values))
        fitted_vals = np.exp(b0 + b1 * t)
        tf = np.arange(len(s) + 1, len(s) + h + 1, dtype=float)
        yhat = np.exp(b0 + b1 * tf)
    else:
        raise ValueError("model deve ser 'linear', 'quadratic' ou 'exponential'.")

    idx = _make_index_like(s, h)
    yhat_s = pd.Series(yhat, index=idx, name="yhat")
    fitted = pd.Series(fitted_vals, index=s.index, name="fitted")
    residuals = s - fitted
    return ForecastResult(
        yhat=yhat_s,
        fitted=fitted,
        residuals=residuals,
        info={"model": "trend_projection", "trend_type": m},
    )


# -----------------------------
# Suavização Exponencial (statsmodels)
# -----------------------------
def ses_forecast(y, h: int, alpha: float | None = None, optimized: bool = True) -> ForecastResult:
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
    smoothing_level: float | None = None,
    smoothing_trend: float | None = None,
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
    trend: str | None = "add",
    seasonal: str | None = "add",
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
        raise ImportError(
            "Para holt_winters_forecast, instale statsmodels: pip install statsmodels"
        ) from e

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


def autoregressive_forecast(
    y,
    h: int,
    lags: int = 1,
    trend: str = "c",
    seasonal: bool = False,
    period: int | None = None,
) -> ForecastResult:
    """
    Previsao com modelo autorregressivo AR(p) via estimacao de Yule-Walker.
    """
    s = as_series(y, name="y").astype(float).dropna()
    if h < 1:
        raise ValueError("h deve ser >= 1.")
    p = int(lags)
    if p < 1:
        raise ValueError("lags deve ser >= 1.")
    if len(s) <= p + 1:
        raise ValueError("Serie muito curta para o numero de lags informado.")
    if seasonal:
        raise NotImplementedError("seasonal=True ainda nao esta disponivel para autoregressive_forecast.")
    if period is not None:
        raise NotImplementedError("period nao se aplica na implementacao atual de autoregressive_forecast.")

    yv = s.values.astype(float)
    mu = float(np.mean(yv)) if trend == "c" else 0.0
    yc = yv - mu

    # autocovariancias amostrais gamma(k)
    n = len(yc)
    gamma = np.empty(p + 1, dtype=float)
    for k in range(p + 1):
        gamma[k] = np.sum(yc[k:] * yc[: n - k]) / n

    # Levinson-Durbin para AR(p)
    phi = np.zeros((p + 1, p + 1), dtype=float)
    v = np.zeros(p + 1, dtype=float)
    v[0] = gamma[0]
    if v[0] <= 0:
        raise ValueError("Variancia da serie nao positiva; AR nao pode ser ajustado.")

    for m in range(1, p + 1):
        if m == 1:
            kappa = gamma[1] / v[0]
        else:
            num = gamma[m] - np.sum(phi[m - 1, 1:m] * gamma[1:m][::-1])
            kappa = num / v[m - 1]

        phi[m, m] = kappa
        for j in range(1, m):
            phi[m, j] = phi[m - 1, j] - kappa * phi[m - 1, m - j]
        v[m] = v[m - 1] * (1 - kappa**2)

    ar_coefs = phi[p, 1 : p + 1]
    intercept = mu * (1 - np.sum(ar_coefs)) if trend == "c" else 0.0

    fitted_vals = np.full(n, np.nan, dtype=float)
    for t in range(p, n):
        lag_vec = yv[t - p : t][::-1]
        fitted_vals[t] = intercept + float(np.dot(ar_coefs, lag_vec))
    fitted = pd.Series(fitted_vals, index=s.index, name="fitted")
    residuals = (s - fitted).rename("residuals")

    # previsao recursiva
    history = list(yv)
    pred_vals = []
    for _ in range(h):
        lag_vec = np.array(history[-p:][::-1], dtype=float)
        next_val = intercept + float(np.dot(ar_coefs, lag_vec))
        pred_vals.append(next_val)
        history.append(next_val)

    idx = _make_index_like(s, h)
    yhat = pd.Series(pred_vals, index=idx, name="yhat")

    return ForecastResult(
        yhat=yhat,
        fitted=fitted,
        residuals=residuals,
        info={
            "model": "AR-YuleWalker",
            "lags": p,
            "trend": trend,
            "intercept": float(intercept),
            "phi": ar_coefs.tolist(),
        },
    )
