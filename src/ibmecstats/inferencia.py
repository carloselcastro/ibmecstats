from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .utils import dropna_series


# -----------------------------
# Intervalos de confiança
# -----------------------------
def ci_mean(
    x,
    alpha: float = 0.05,
    method: str = "t",
) -> tuple[float, float]:
    """
    IC para a média.

    method:
      - 't' : usa t-student (sigma desconhecido)  [recomendado]
      - 'z' : usa normal padrão (sigma conhecido/assumido)
    """
    s = dropna_series(x, name="x")
    n = s.size
    if n < 2:
        raise ValueError("Amostra muito pequena para IC de média.")
    xbar = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(n))

    if method.lower() == "t":
        crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    elif method.lower() == "z":
        crit = stats.norm.ppf(1 - alpha / 2)
    else:
        raise ValueError("method deve ser 't' ou 'z'.")

    return xbar - crit * se, xbar + crit * se


def ci_proportion(
    k: int,
    n: int,
    alpha: float = 0.05,
    method: str = "wilson",
) -> tuple[float, float]:
    """
    IC para proporção p = k/n.

    method:
      - 'wald'   : clássico (p +/- z*sqrt(p(1-p)/n)) (pode ser ruim em amostras pequenas)
      - 'wilson' : recomendado
      - 'agresti-coull' : alternativa robusta
    """
    if n <= 0:
        raise ValueError("n deve ser > 0.")
    if not (0 <= k <= n):
        raise ValueError("k deve estar em [0,n].")
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)

    m = method.lower()
    if m == "wald":
        se = np.sqrt(p * (1 - p) / n)
        return max(0.0, p - z * se), min(1.0, p + z * se)

    if m == "wilson":
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
        return max(0.0, center - half), min(1.0, center + half)

    if m in {"agresti-coull", "ac"}:
        n_tilde = n + z**2
        p_tilde = (k + z**2 / 2) / n_tilde
        se = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        return max(0.0, p_tilde - z * se), min(1.0, p_tilde + z * se)

    raise ValueError("method deve ser 'wald', 'wilson' ou 'agresti-coull'.")


# -----------------------------
# Testes clássicos
# -----------------------------
def t_test_1samp(x, mu0: float, alternative: str = "two-sided") -> dict[str, float]:
    """
    Teste t de 1 amostra.

    alternative: 'two-sided', 'less', 'greater'
    Retorna dict com statistic, pvalue, df
    """
    s = dropna_series(x, name="x")
    stat, p = stats.ttest_1samp(s, popmean=mu0, alternative=alternative)
    return {"statistic": float(stat), "pvalue": float(p), "df": float(len(s) - 1)}


def t_test_ind(
    x1,
    x2,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Teste t para duas amostras independentes.
    equal_var=False => Welch (recomendado em geral).
    """
    s1 = dropna_series(x1, name="x1")
    s2 = dropna_series(x2, name="x2")
    stat, p = stats.ttest_ind(s1, s2, equal_var=equal_var, alternative=alternative)
    return {"statistic": float(stat), "pvalue": float(p)}


def t_test_paired(x1, x2, alternative: str = "two-sided") -> dict[str, float]:
    """
    Teste t pareado.
    """
    s1 = dropna_series(x1, name="x1")
    s2 = dropna_series(x2, name="x2")
    if len(s1) != len(s2):
        raise ValueError("Para teste pareado, x1 e x2 devem ter o mesmo tamanho (após dropna).")
    stat, p = stats.ttest_rel(s1, s2, alternative=alternative)
    return {"statistic": float(stat), "pvalue": float(p), "df": float(len(s1) - 1)}


def z_test_proportion(
    k: int,
    n: int,
    p0: float,
    alternative: str = "two-sided",
    continuity: bool = True,
) -> dict[str, float]:
    """
    Teste z para proporção.

    H0: p = p0
    alternative: 'two-sided', 'less', 'greater'
    continuity: correção de continuidade (aprox. conservadora)
    """
    if n <= 0:
        raise ValueError("n deve ser > 0.")
    if not (0 <= k <= n):
        raise ValueError("k deve estar em [0,n].")
    if not (0.0 < p0 < 1.0):
        raise ValueError("p0 deve estar em (0,1).")

    phat = k / n
    se0 = np.sqrt(p0 * (1 - p0) / n)

    # correção de continuidade simples (ajusta phat levemente)
    adj = (0.5 / n) if continuity else 0.0
    if alternative == "greater":
        z = (phat - p0 - adj) / se0
        p = 1 - stats.norm.cdf(z)
    elif alternative == "less":
        z = (phat - p0 + adj) / se0
        p = stats.norm.cdf(z)
    elif alternative == "two-sided":
        z = (phat - p0) / se0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError("alternative deve ser 'two-sided', 'less' ou 'greater'.")

    return {"z": float(z), "pvalue": float(p), "phat": float(phat)}


def chi2_gof(observed, expected=None) -> dict[str, float]:
    """
    Qui-quadrado de aderência (GOF).
    observed: contagens observadas
    expected: contagens esperadas (mesmo shape) ou None => assume equiprovável
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 1:
        raise ValueError("observed deve ser 1D.")
    if expected is None:
        exp = np.repeat(obs.sum() / obs.size, obs.size)
    else:
        exp = np.asarray(expected, dtype=float)
        if exp.shape != obs.shape:
            raise ValueError("expected deve ter o mesmo shape de observed.")

    stat, p = stats.chisquare(f_obs=obs, f_exp=exp)
    df = obs.size - 1
    return {"statistic": float(stat), "pvalue": float(p), "df": float(df)}


def chi2_independence(table, correction: bool = True) -> dict[str, float]:
    """
    Qui-quadrado de independência em tabela de contingência.
    """
    tab = np.asarray(table, dtype=float)
    stat, p, df, expected = stats.chi2_contingency(tab, correction=correction)
    return {"statistic": float(stat), "pvalue": float(p), "df": float(df)}


def anova_oneway(*groups) -> dict[str, float]:
    """
    ANOVA one-way (um fator).
    """
    clean = [dropna_series(g).values for g in groups]
    stat, p = stats.f_oneway(*clean)
    return {"statistic": float(stat), "pvalue": float(p)}


# -----------------------------
# Normalidade (diagnóstico)
# -----------------------------
def normality_tests(x) -> pd.DataFrame:
    """
    Retorna uma tabela com testes comuns de normalidade:
      - Shapiro-Wilk
      - Anderson-Darling (normal)
      - KS (contra normal com média/desvio estimados) [observação: não é Lilliefors]
    """
    s = dropna_series(x, name="x").astype(float)
    if len(s) < 3:
        raise ValueError("Amostra muito pequena para testes de normalidade.")

    rows = []

    # Shapiro
    w, p = stats.shapiro(s)
    rows.append({"test": "Shapiro-Wilk", "statistic": float(w), "pvalue": float(p)})

    # Anderson
    ad = stats.anderson(s, dist="norm")
    # Anderson não fornece pvalue direto; reportamos stat e níveis críticos
    rows.append(
        {
            "test": "Anderson-Darling (normal)",
            "statistic": float(ad.statistic),
            "pvalue": np.nan,
        }
    )

    # KS (com parâmetros estimados) - cuidado conceitual
    mu, sigma = float(s.mean()), float(s.std(ddof=1))
    if sigma <= 0:
        ks_stat, ks_p = np.nan, np.nan
    else:
        ks_stat, ks_p = stats.kstest(s, "norm", args=(mu, sigma))
    rows.append({"test": "KS vs N(mu, sigma)", "statistic": float(ks_stat), "pvalue": float(ks_p)})

    return pd.DataFrame(rows)


# -----------------------------
# Bootstrap CI genérico
# -----------------------------
def bootstrap_ci(
    x,
    statfunc: Callable[[np.ndarray], float] = np.mean,
    alpha: float = 0.05,
    n_boot: int = 10000,
    random_state: int | None = None,
) -> dict[str, float]:
    """
    IC bootstrap percentil para uma estatística (ex.: média, mediana).

    Retorna dict com estimate, lower, upper
    """
    s = dropna_series(x, name="x").values.astype(float)
    n = len(s)
    if n < 2:
        raise ValueError("Amostra muito pequena para bootstrap.")

    rng = np.random.default_rng(random_state)
    stats_boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(s, size=n, replace=True)
        stats_boot[b] = float(statfunc(sample))

    est = float(statfunc(s))
    lo = float(np.quantile(stats_boot, alpha / 2))
    hi = float(np.quantile(stats_boot, 1 - alpha / 2))
    return {"estimate": est, "lower": lo, "upper": hi}
