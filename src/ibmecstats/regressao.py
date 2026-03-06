from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .utils import as_series


@dataclass(frozen=True)
class OLSModel:
    params: pd.Series
    feature_names: list[str]
    include_constant: bool
    nobs: int
    df_model: int
    df_resid: int
    rsquared: float
    rsquared_adj: float
    fvalue: float
    f_pvalue: float
    sigma_hat: float
    aic: float
    bic: float
    residuals: pd.Series
    fittedvalues: pd.Series


def _to_dataframe_x(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, pd.Series):
        return x.to_frame(name=x.name or "x1")

    arr = np.asarray(x)
    if arr.ndim == 1:
        return pd.DataFrame({"x1": arr})
    if arr.ndim == 2:
        return pd.DataFrame(arr, columns=[f"x{i + 1}" for i in range(arr.shape[1])])
    raise ValueError("X deve ser array-like 1D/2D, Series ou DataFrame.")


def _prepare_xy(y, x) -> tuple[pd.Series, pd.DataFrame]:
    ys = as_series(y, name="y").astype(float)
    xd = _to_dataframe_x(x).copy()
    df = pd.concat([ys, xd], axis=1).dropna()
    y_clean = df.iloc[:, 0].astype(float)
    x_clean = df.iloc[:, 1:].astype(float)
    if x_clean.shape[1] == 0:
        raise ValueError("X nao possui colunas validas apos alinhamento/dropna.")
    return y_clean, x_clean


def _compute_xtx_xty(x_mat: np.ndarray, y_vec: np.ndarray) -> tuple[list[list[float]], list[float]]:
    n, p = x_mat.shape
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]

    for r in range(n):
        xr = x_mat[r]
        yr = float(y_vec[r])
        for i in range(p):
            xi = float(xr[i])
            xty[i] += xi * yr
            for j in range(p):
                xtx[i][j] += xi * float(xr[j])

    return xtx, xty


def _solve_beta(x_mat: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
    xtx, xty = _compute_xtx_xty(x_mat, y_vec)

    # Solucao por eliminacao de Gauss em Python puro para evitar falhas de backend BLAS/LAPACK.
    a = [row[:] for row in xtx]
    b = xty[:]
    n = len(a)

    # regularizacao leve na diagonal para melhorar estabilidade numerica
    for i in range(n):
        a[i][i] += 1e-8

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(a[r][k]))
        if abs(a[pivot_row][k]) < 1e-14:
            raise ValueError("Matriz singular ao estimar OLS.")
        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]

        pivot = a[k][k]
        for j in range(k, n):
            a[k][j] /= pivot
        b[k] /= pivot

        for i in range(n):
            if i == k:
                continue
            factor = a[i][k]
            if factor == 0.0:
                continue
            for j in range(k, n):
                a[i][j] -= factor * a[k][j]
            b[i] -= factor * b[k]

    return np.asarray(b, dtype=float)


def add_dummy_variables(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Cria variaveis dummy para colunas categoricas.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas nao encontradas: {missing}")
    return pd.get_dummies(df, columns=list(columns), drop_first=drop_first, dtype=float)


def ols_fit(y, x, add_constant: bool = True) -> OLSModel:
    """
    Ajuste de regressao linear por minimos quadrados ordinarios.
    """
    y_clean, x_clean = _prepare_xy(y, x)

    if add_constant:
        x_design = pd.concat([pd.Series(1.0, index=x_clean.index, name="const"), x_clean], axis=1)
    else:
        x_design = x_clean.copy()

    x_mat = x_design.values.astype(float)
    y_vec = y_clean.values.astype(float)
    beta = _solve_beta(x_mat, y_vec)

    fitted = np.empty(len(y_vec), dtype=float)
    for r in range(len(y_vec)):
        fitted[r] = float(sum(float(x_mat[r, c]) * float(beta[c]) for c in range(x_mat.shape[1])))
    resid = y_vec - fitted
    n = len(y_vec)
    p = x_mat.shape[1]
    if n <= p:
        raise ValueError("Numero de observacoes insuficiente para estimar OLS.")

    sse = float(np.sum(resid**2))
    y_mean = float(np.mean(y_vec))
    sst = float(np.sum((y_vec - y_mean) ** 2))
    rsq = 1.0 - (sse / sst) if sst > 0 else 0.0
    rsq_adj = 1.0 - ((1.0 - rsq) * (n - 1) / (n - p))

    df_model = p - 1 if add_constant else p
    df_resid = n - p
    mse_resid = sse / df_resid
    sigma_hat = float(np.sqrt(mse_resid))

    if df_model > 0:
        ssr = sst - sse
        msr = ssr / df_model
        fval = float(msr / mse_resid) if mse_resid > 0 else np.inf
        fp = float(1 - stats.f.cdf(fval, df_model, df_resid)) if np.isfinite(fval) else 0.0
    else:
        fval, fp = np.nan, np.nan

    sse_safe = max(sse, 1e-12)
    aic = float(n * np.log(sse_safe / n) + 2 * p)
    bic = float(n * np.log(sse_safe / n) + np.log(n) * p)

    params = pd.Series(beta, index=list(x_design.columns), name="coef")
    fitted_s = pd.Series(fitted, index=y_clean.index, name="fitted")
    resid_s = pd.Series(resid, index=y_clean.index, name="residuals")

    return OLSModel(
        params=params,
        feature_names=list(x_clean.columns),
        include_constant=add_constant,
        nobs=n,
        df_model=df_model,
        df_resid=df_resid,
        rsquared=float(rsq),
        rsquared_adj=float(rsq_adj),
        fvalue=float(fval),
        f_pvalue=float(fp),
        sigma_hat=sigma_hat,
        aic=aic,
        bic=bic,
        residuals=resid_s,
        fittedvalues=fitted_s,
    )


def _prepare_exog_for_model(model: OLSModel, x_new) -> pd.DataFrame:
    xdf = _to_dataframe_x(x_new).copy()
    for col in model.feature_names:
        if col not in xdf.columns:
            xdf[col] = 0.0
    xdf = xdf[model.feature_names]
    if model.include_constant:
        xdf = pd.concat([pd.Series(1.0, index=xdf.index, name="const"), xdf], axis=1)
    return xdf


def ols_predict(model: OLSModel, x_new) -> pd.Series:
    """
    Previsao para novos valores de X em um modelo OLS ajustado.
    """
    exog = _prepare_exog_for_model(model, x_new)
    x_mat = exog.values.astype(float)
    b = model.params.values.astype(float)
    yhat = np.empty(x_mat.shape[0], dtype=float)
    for r in range(x_mat.shape[0]):
        yhat[r] = float(sum(float(x_mat[r, c]) * float(b[c]) for c in range(x_mat.shape[1])))
    return pd.Series(yhat, index=exog.index, name="yhat")


def ols_diagnostics(model: OLSModel) -> dict[str, float]:
    """
    Principais metricas de diagnostico para regressao linear.
    """
    return {
        "r2": float(model.rsquared),
        "r2_adj": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "sigma_hat": float(model.sigma_hat),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "nobs": float(model.nobs),
    }


def _fit_with_features(
    y: pd.Series,
    x: pd.DataFrame,
    features: list[str],
    add_constant: bool = True,
):
    x_sub = x[features].copy()
    return ols_fit(y, x_sub, add_constant=add_constant)


def best_subset_selection(
    y,
    x,
    criterion: str = "aic",
    max_features: int | None = None,
    add_constant: bool = True,
) -> dict[str, object]:
    """
    Seleciona o melhor subconjunto por AIC ou BIC.
    """
    y_clean, x_clean = _prepare_xy(y, x)
    cols = list(x_clean.columns)
    if max_features is None:
        max_features = len(cols)
    max_features = max(1, min(int(max_features), len(cols)))

    crit = criterion.lower()
    if crit not in {"aic", "bic"}:
        raise ValueError("criterion deve ser 'aic' ou 'bic'.")

    best_model = None
    best_features: list[str] = []
    best_value = np.inf

    for r in range(1, max_features + 1):
        for comb in combinations(cols, r):
            mdl = _fit_with_features(y_clean, x_clean, list(comb), add_constant=add_constant)
            val = float(getattr(mdl, crit))
            if val < best_value:
                best_value = val
                best_model = mdl
                best_features = list(comb)

    return {"model": best_model, "features": best_features, criterion: float(best_value)}


def model_selection(
    y,
    x,
    method: str = "forward",
    criterion: str = "aic",
    add_constant: bool = True,
) -> dict[str, object]:
    """
    Selecao de modelos: forward, backward, stepwise ou bestsubset.
    """
    y_clean, x_clean = _prepare_xy(y, x)
    cols = list(x_clean.columns)
    if len(cols) == 0:
        raise ValueError("Nao ha variaveis explicativas em X.")

    meth = method.lower()
    crit = criterion.lower()
    if crit not in {"aic", "bic"}:
        raise ValueError("criterion deve ser 'aic' ou 'bic'.")
    if meth == "bestsubset":
        return best_subset_selection(
            y_clean, x_clean, criterion=criterion, max_features=len(cols), add_constant=add_constant
        )

    selected: list[str]
    if meth == "backward":
        selected = cols.copy()
        best_model = _fit_with_features(y_clean, x_clean, selected, add_constant=add_constant)
        best_value = float(getattr(best_model, crit))

        improved = True
        while improved and len(selected) > 1:
            improved = False
            candidates = []
            for c in selected:
                subset = [v for v in selected if v != c]
                mdl = _fit_with_features(y_clean, x_clean, subset, add_constant=add_constant)
                candidates.append((float(getattr(mdl, crit)), subset, mdl))
            cand_val, cand_subset, cand_model = min(candidates, key=lambda z: z[0])
            if cand_val < best_value:
                best_value = cand_val
                selected = cand_subset
                best_model = cand_model
                improved = True
        return {"model": best_model, "features": selected, criterion: best_value}

    selected = []
    remaining = cols.copy()
    best_model = None
    best_value = np.inf

    while remaining:
        candidates = []
        for c in remaining:
            subset = selected + [c]
            mdl = _fit_with_features(y_clean, x_clean, subset, add_constant=add_constant)
            candidates.append((float(getattr(mdl, crit)), c, mdl))
        cand_val, cand_col, cand_model = min(candidates, key=lambda z: z[0])

        if cand_val < best_value:
            best_value = cand_val
            selected.append(cand_col)
            remaining.remove(cand_col)
            best_model = cand_model
        else:
            break

        if meth == "stepwise" and len(selected) > 1:
            improved = True
            while improved and len(selected) > 1:
                improved = False
                remove_candidates = []
                for c in selected:
                    subset = [v for v in selected if v != c]
                    mdl = _fit_with_features(y_clean, x_clean, subset, add_constant=add_constant)
                    remove_candidates.append((float(getattr(mdl, crit)), c, subset, mdl))
                rem_val, rem_col, rem_subset, rem_model = min(remove_candidates, key=lambda z: z[0])
                if rem_val < best_value:
                    best_value = rem_val
                    selected = rem_subset
                    if rem_col not in remaining:
                        remaining.append(rem_col)
                    best_model = rem_model
                    improved = True

    if meth not in {"forward", "stepwise"}:
        raise ValueError("method deve ser 'forward', 'backward', 'stepwise' ou 'bestsubset'.")

    if best_model is None:
        raise ValueError("Nao foi possivel ajustar nenhum modelo.")
    return {"model": best_model, "features": selected, criterion: float(best_value)}
