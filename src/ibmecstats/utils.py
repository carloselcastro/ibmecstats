from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ArrayLike1D = pd.Series | np.ndarray | list | tuple


def as_series(x: ArrayLike1D, name: str = "x") -> pd.Series:
    """
    Converte entrada 1D (list/tuple/ndarray/Series) em pd.Series.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
        if s.name is None:
            s.name = name
        return s
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Esperado vetor 1D; recebido array com ndim={arr.ndim}.")
    return pd.Series(arr, name=name)


def dropna_series(x: ArrayLike1D, name: str = "x") -> pd.Series:
    """
    Converte para Series e remove NaNs.
    """
    s = as_series(x, name=name)
    return s.dropna()


def ensure_datetime_index(
    y: pd.Series | pd.DataFrame,
    date_col: str | None = None,
    freq: str | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Garante que y tenha índice datetime. Aceita:
      - Series/DataFrame já com DatetimeIndex
      - DataFrame com uma coluna de data (date_col)
    Se freq for informado, tenta asfreq(freq).
    """
    obj = y.copy()

    if isinstance(obj, (pd.Series, pd.DataFrame)) and isinstance(obj.index, pd.DatetimeIndex):
        out = obj
    elif isinstance(obj, pd.DataFrame) and date_col is not None:
        if date_col not in obj.columns:
            raise ValueError(f"date_col='{date_col}' não está nas colunas do DataFrame.")
        out = obj.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.set_index(date_col).sort_index()
    else:
        raise ValueError(
            "Não foi possível inferir DatetimeIndex. "
            "Passe um Series/DataFrame com DatetimeIndex ou um DataFrame com date_col."
        )

    if freq is not None:
        out = out.asfreq(freq)

    return out


def train_test_split_time(
    y: ArrayLike1D,
    test_size: int | float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    """
    Split temporal (sem shuffle). Retorna (train, test).

    test_size:
      - float (0,1): fração do tamanho total
      - int >= 1   : número de observações no teste
    """
    s = as_series(y, name="y").dropna()
    n = len(s)
    if n < 2:
        raise ValueError("Série muito curta para split temporal.")

    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size float deve estar em (0,1).")
        n_test = int(np.ceil(n * test_size))
    else:
        n_test = int(test_size)

    if n_test <= 0 or n_test >= n:
        raise ValueError(
            "test_size inválido; precisa deixar pelo menos 1 ponto no treino e 1 no teste."
        )

    train = s.iloc[: n - n_test]
    test = s.iloc[n - n_test :]
    return train, test


@dataclass(frozen=True)
class ForecastResult:
    """
    Resultado padronizado para previsões.
    """

    yhat: pd.Series
    fitted: pd.Series | None = None
    residuals: pd.Series | None = None
    info: dict | None = None
