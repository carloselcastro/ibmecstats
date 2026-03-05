from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .utils import as_series, dropna_series


def summary_stats(
    x,
    percentiles: Sequence[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
    ddof: int = 1,
) -> pd.Series:
    """
    Estatísticas descritivas fundamentais.

    Parâmetros
    ----------
    x : array-like 1D
    percentiles : sequência de quantis (0-1)
    ddof : ddof do desvio-padrão (default=1)

    Retorna
    -------
    pd.Series com n, mean, std, min, quantis, max, skew, kurtosis
    """
    s = dropna_series(x, name="x")
    q = s.quantile(list(percentiles))

    out = {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=ddof)),
        "min": float(s.min()),
        "max": float(s.max()),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurt()),
    }

    for p, val in q.items():
        out[f"q_{p:.2f}"] = float(val)

    return pd.Series(out, name="summary_stats")


def freq_table(
    x,
    normalize: bool = False,
    dropna: bool = False,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Tabela de frequências (absoluta e relativa opcional).

    Retorna DataFrame com colunas:
      - count
      - proportion (se normalize=True)
    """
    s = as_series(x, name="x")
    vc = s.value_counts(dropna=dropna, sort=sort)
    df = vc.to_frame(name="count")
    if normalize:
        df["proportion"] = df["count"] / df["count"].sum()
    return df


def iqr_outliers(x, k: float = 1.5) -> pd.DataFrame:
    """
    Detecta outliers via regra do IQR.

    Retorna DataFrame com:
      - value
      - is_outlier
      - lower_bound
      - upper_bound
    """
    s = dropna_series(x, name="x")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    out = pd.DataFrame({"value": s})
    out["lower_bound"] = lower
    out["upper_bound"] = upper
    out["is_outlier"] = (out["value"] < lower) | (out["value"] > upper)
    return out


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Matriz de correlação (apenas colunas numéricas).
    method: 'pearson', 'spearman', 'kendall'
    """
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        raise ValueError("DataFrame não possui colunas numéricas para correlação.")
    return num.corr(method=method)
