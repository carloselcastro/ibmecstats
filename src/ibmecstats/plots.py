from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import as_series, dropna_series


def set_theme(
    style: str = "whitegrid",
    context: str = "talk",
    palette: str = "deep",
    font_scale: float = 1.0,
    rc: Optional[dict] = None,
) -> None:
    """
    Define um tema padrão seaborn para gráficos do pacote.
    """
    import seaborn as sns

    sns.set_theme(style=style, context=context, palette=palette, font_scale=font_scale, rc=rc)


def plot_distribution(
    x,
    bins: Union[int, str] = "auto",
    kde: bool = True,
    title: Optional[str] = None,
):
    """
    Histograma + KDE (opcional).
    Retorna Axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    s = dropna_series(x, name="x").astype(float)

    ax = sns.histplot(s, bins=bins, kde=kde)
    ax.set_xlabel(s.name or "x")
    ax.set_ylabel("Frequência")
    if title:
        ax.set_title(title)
    return ax


def plot_boxplot(x, title: Optional[str] = None):
    """
    Boxplot univariado.
    """
    import seaborn as sns

    s = dropna_series(x, name="x")
    ax = sns.boxplot(x=s)
    ax.set_xlabel(s.name or "x")
    if title:
        ax.set_title(title)
    return ax


def plot_qq(x, dist: str = "norm", title: Optional[str] = None):
    """
    Q-Q plot (via scipy.stats.probplot).
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    s = dropna_series(x, name="x").astype(float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    stats.probplot(s, dist=dist, plot=ax)
    if title:
        ax.set_title(title)
    return ax


def plot_pp(x, dist: str = "norm", title: Optional[str] = None):
    """
    P-P plot (empírico vs teórico). Para normal, usa mu/sigma estimados.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    s = dropna_series(x, name="x").astype(float).values
    s_sorted = np.sort(s)
    n = len(s_sorted)
    emp = (np.arange(1, n + 1) - 0.5) / n  # plotting positions

    if dist == "norm":
        mu = float(np.mean(s_sorted))
        sigma = float(np.std(s_sorted, ddof=1))
        theo = stats.norm.cdf(s_sorted, loc=mu, scale=sigma)
    else:
        # fallback genérico: usa CDF com parâmetros padrão
        dist_obj = getattr(stats, dist)
        theo = dist_obj.cdf(s_sorted)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(theo, emp, marker="o", linestyle="None")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("CDF teórica")
    ax.set_ylabel("CDF empírica")
    if title:
        ax.set_title(title)
    return ax


def plot_time_series(y, title: Optional[str] = None):
    """
    Série temporal simples.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    s = as_series(y, name="y").astype(float).dropna()
    ax = sns.lineplot(x=s.index, y=s.values)
    ax.set_xlabel("tempo")
    ax.set_ylabel(s.name or "y")
    if title:
        ax.set_title(title)
    return ax


def plot_acf_pacf(y, lags: int = 40, title: Optional[str] = None):
    """
    ACF e PACF (statsmodels). Retorna tupla (ax_acf, ax_pacf).
    """
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except Exception as e:
        raise ImportError("Para ACF/PACF, instale statsmodels: pip install statsmodels") from e

    import matplotlib.pyplot as plt

    s = as_series(y, name="y").astype(float).dropna()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plot_acf(s, lags=lags, ax=ax1)
    if title:
        ax1.set_title(f"{title} — ACF")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    plot_pacf(s, lags=lags, ax=ax2, method="ywm")
    if title:
        ax2.set_title(f"{title} — PACF")

    return ax1, ax2


def plot_forecast(
    y_train,
    y_test,
    y_pred,
    title: Optional[str] = None,
):
    """
    Plota treino, teste e previsão.
    """
    import seaborn as sns

    tr = as_series(y_train, name="train").astype(float)
    te = as_series(y_test, name="test").astype(float)
    yp = as_series(y_pred, name="pred").astype(float)

    # junta para plot
    df = pd.concat(
        [
            tr.rename("Treino"),
            te.rename("Teste"),
            yp.rename("Previsão"),
        ],
        axis=1,
    )

    ax = sns.lineplot(data=df)
    ax.set_xlabel("tempo")
    ax.set_ylabel("valor")
    if title:
        ax.set_title(title)
    return ax


def plot_residuals(residuals, title: Optional[str] = None):
    """
    Resíduos ao longo do tempo (linha) + hist (novo Axes em figura separada).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    r = as_series(residuals, name="residuals").astype(float).dropna()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    sns.lineplot(x=r.index, y=r.values, ax=ax1)
    ax1.axhline(0, linestyle="--")
    ax1.set_xlabel("tempo")
    ax1.set_ylabel("resíduo")
    if title:
        ax1.set_title(f"{title} — resíduos")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    sns.histplot(r, kde=True, ax=ax2)
    ax2.set_xlabel("resíduo")
    ax2.set_ylabel("frequência")
    if title:
        ax2.set_title(f"{title} — distribuição dos resíduos")

    return ax1, ax2