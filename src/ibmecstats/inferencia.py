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


def chi2_homogeneity(table, correction: bool = False) -> dict[str, float]:
    """
    Qui-quadrado de homogeneidade (mesma estatística do teste de independência).
    """
    tab = np.asarray(table, dtype=float)
    stat, p, df, _ = stats.chi2_contingency(tab, correction=correction)
    return {"statistic": float(stat), "pvalue": float(p), "df": float(df)}


def anova_oneway(*groups) -> dict[str, float]:
    """
    ANOVA one-way (um fator).
    """
    clean = [dropna_series(g).values for g in groups]
    stat, p = stats.f_oneway(*clean)
    return {"statistic": float(stat), "pvalue": float(p)}


def bartlett_homoscedasticity(*groups) -> dict[str, float]:
    """
    Teste de Bartlett para homocedasticidade entre grupos.
    """
    clean = [dropna_series(g).values for g in groups]
    if len(clean) < 2:
        raise ValueError("Informe pelo menos dois grupos para o teste de Bartlett.")
    stat, p = stats.bartlett(*clean)
    return {"statistic": float(stat), "pvalue": float(p)}


def cochran_c_test(
    *groups,
    n_sim: int = 20000,
    random_state: int | None = None,
) -> dict[str, float]:
    """
    Teste C de Cochran para homocedasticidade.

    O p-valor e aproximado por simulacao Monte Carlo sob normalidade.
    """
    clean = [dropna_series(g).values.astype(float) for g in groups]
    if len(clean) < 2:
        raise ValueError("Informe pelo menos dois grupos para o teste C de Cochran.")
    if any(len(g) < 2 for g in clean):
        raise ValueError("Cada grupo deve ter pelo menos 2 observacoes.")

    variances = np.array([np.var(g, ddof=1) for g in clean], dtype=float)
    c_stat = float(np.max(variances) / np.sum(variances))

    rng = np.random.default_rng(random_state)
    sizes = [len(g) for g in clean]
    sim_stats = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        sim_vars = np.array(
            [np.var(rng.normal(loc=0.0, scale=1.0, size=n), ddof=1) for n in sizes],
            dtype=float,
        )
        sim_stats[i] = np.max(sim_vars) / np.sum(sim_vars)

    pvalue = float(np.mean(sim_stats >= c_stat))
    return {"statistic": c_stat, "pvalue": pvalue, "k_groups": float(len(clean))}


def correlation_test(x, y, method: str = "pearson") -> dict[str, float]:
    """
    Teste de correlacao para Pearson, Spearman ou Kendall.
    """
    s1 = dropna_series(x, name="x").astype(float)
    s2 = dropna_series(y, name="y").astype(float)
    df = pd.concat([s1, s2], axis=1).dropna()
    if len(df) < 3:
        raise ValueError("Amostra conjunta muito pequena para teste de correlacao.")

    m = method.lower()
    x_clean, y_clean = df.iloc[:, 0].values, df.iloc[:, 1].values
    if m == "pearson":
        stat, p = stats.pearsonr(x_clean, y_clean)
    elif m == "spearman":
        stat, p = stats.spearmanr(x_clean, y_clean)
    elif m == "kendall":
        stat, p = stats.kendalltau(x_clean, y_clean)
    else:
        raise ValueError("method deve ser 'pearson', 'spearman' ou 'kendall'.")

    return {"statistic": float(stat), "pvalue": float(p)}


def jarque_bera_test(x) -> dict[str, float]:
    """
    Teste de normalidade de Jarque-Bera.
    """
    s = dropna_series(x, name="x").astype(float)
    stat, p = stats.jarque_bera(s)
    return {"statistic": float(stat), "pvalue": float(p)}


def f_test_variances(x1, x2, alternative: str = "two-sided") -> dict[str, float]:
    """
    Teste F para comparacao de variancias de duas amostras.
    """
    s1 = dropna_series(x1, name="x1").astype(float).values
    s2 = dropna_series(x2, name="x2").astype(float).values
    if len(s1) < 2 or len(s2) < 2:
        raise ValueError("Cada amostra precisa de pelo menos 2 observacoes.")

    v1 = np.var(s1, ddof=1)
    v2 = np.var(s2, ddof=1)
    f_stat = float(v1 / v2)
    df1, df2 = len(s1) - 1, len(s2) - 1

    if alternative == "greater":
        p = 1 - stats.f.cdf(f_stat, df1, df2)
    elif alternative == "less":
        p = stats.f.cdf(f_stat, df1, df2)
    elif alternative == "two-sided":
        p = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
        p = min(1.0, p)
    else:
        raise ValueError("alternative deve ser 'two-sided', 'less' ou 'greater'.")

    return {"statistic": f_stat, "pvalue": float(p), "df1": float(df1), "df2": float(df2)}


def ci_variance(x, alpha: float = 0.05) -> tuple[float, float]:
    """
    IC para variancia populacional via distribuicao qui-quadrado.
    """
    s = dropna_series(x, name="x").astype(float).values
    n = len(s)
    if n < 2:
        raise ValueError("Amostra muito pequena para IC de variancia.")

    s2 = np.var(s, ddof=1)
    df = n - 1
    chi2_lo = stats.chi2.ppf(alpha / 2, df=df)
    chi2_hi = stats.chi2.ppf(1 - alpha / 2, df=df)
    lower = (df * s2) / chi2_hi
    upper = (df * s2) / chi2_lo
    return float(lower), float(upper)


def ci_mean_diff(
    x1,
    x2,
    alpha: float = 0.05,
    paired: bool = False,
    equal_var: bool = False,
) -> tuple[float, float]:
    """
    IC para diferenca de medias (mu1 - mu2), pareada ou independente.
    """
    s1 = dropna_series(x1, name="x1").astype(float).values
    s2 = dropna_series(x2, name="x2").astype(float).values

    if paired:
        if len(s1) != len(s2):
            raise ValueError("Para IC pareado, x1 e x2 devem ter mesmo tamanho.")
        d = s1 - s2
        n = len(d)
        if n < 2:
            raise ValueError("Amostra pareada muito pequena.")
        dbar = np.mean(d)
        se = np.std(d, ddof=1) / np.sqrt(n)
        crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        return float(dbar - crit * se), float(dbar + crit * se)

    n1, n2 = len(s1), len(s2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Cada amostra precisa de pelo menos 2 observacoes.")

    m1, m2 = np.mean(s1), np.mean(s2)
    v1, v2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
    diff = m1 - m2

    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        num = (v1 / n1 + v2 / n2) ** 2
        den = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
        df = num / den

    crit = stats.t.ppf(1 - alpha / 2, df=df)
    return float(diff - crit * se), float(diff + crit * se)


def ci_proportion_diff(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    IC de Wald para diferenca de proporcoes (p1 - p2).
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 e n2 devem ser > 0.")
    if not (0 <= k1 <= n1 and 0 <= k2 <= n2):
        raise ValueError("k1 e k2 devem estar nos intervalos [0,n1] e [0,n2].")

    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    return float(d - z * se), float(d + z * se)


def z_test_2proportions(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Teste z para H0: p1 = p2 com estimativa combinada.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 e n2 devem ser > 0.")
    if not (0 <= k1 <= n1 and 0 <= k2 <= n2):
        raise ValueError("k1 e k2 devem estar nos intervalos [0,n1] e [0,n2].")

    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * ((1 / n1) + (1 / n2)))
    if se == 0:
        raise ValueError("Erro padrao nulo; dados degenerados.")

    z = (p1 - p2) / se
    if alternative == "greater":
        p = 1 - stats.norm.cdf(z)
    elif alternative == "less":
        p = stats.norm.cdf(z)
    elif alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError("alternative deve ser 'two-sided', 'less' ou 'greater'.")

    return {"z": float(z), "pvalue": float(p), "p1": float(p1), "p2": float(p2)}


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

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(s)
    rows.append({"test": "Jarque-Bera", "statistic": float(jb_stat), "pvalue": float(jb_p)})

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
