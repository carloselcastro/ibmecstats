<p align="center">
  <img
    src="https://raw.githubusercontent.com/carloselcastro/carloselcastro.github.io/master/images/ibmecstats.png"
    alt="IbmecStats"
    width="260"
  />
</p>

<p align="center">
  <a href="https://pypi.org/project/ibmecstats/"><img alt="PyPI" src="https://img.shields.io/pypi/v/ibmecstats"></a>
  <a href="https://pypi.org/project/ibmecstats/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/ibmecstats"></a>
  <a href="https://github.com/carloselcastro/ibmecstats/actions"><img alt="CI" src="https://github.com/carloselcastro/ibmecstats/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/carloselcastro/ibmecstats/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

# Biblioteca IBMEC Stats

Um pacote Python com funções fundamentais para **Estatística**, **Inferência Estatística** e **Métodos de Previsão**, com **visualizações prontas** usando **seaborn**.

Este projeto contém implementações de funções estatísticas para uso na disciplina de Inferência Estatística Métodos de Previsão do IBMEC/SP.

## Instalação

```bash
pip install ibmecstats
```
---

## Requisitos

Este pacote utiliza:

* numpy, pandas, scipy
* matplotlib, seaborn
* statsmodels (para suavização exponencial e ACF/PACF)

---

## Quickstart

```python
import pandas as pd
import ibmecstats as ibs

ibs.set_theme()

x = [10, 11, 10, 12, 9, 10, 11, 10, 200]  # tem outlier :)

# Descritiva
print(ibs.summary_stats(x))
print(ibs.freq_table(["A","A","B","C","A"], normalize=True))
out = ibs.iqr_outliers(x)
print(out[out["is_outlier"]])

# Inferência
print(ibs.ci_mean(x, alpha=0.05, method="t"))
print(ibs.t_test_1samp(x, mu0=10, alternative="two-sided"))

# Gráficos
ibs.plot_distribution(x, kde=True, title="Distribuição de x")
ibs.plot_qq(x, title="Q-Q plot")
```

---

## API (Referência)

### 1) Utilitários (`ibmecstats.utils`)

* `ForecastResult`: estrutura padrão de retorno para previsão (`yhat`, `fitted`, `residuals`, `info`).
* `ensure_datetime_index(y, date_col=None, freq=None)`: garante índice temporal em `Series`/`DataFrame`.
* `train_test_split_time(y, test_size=0.2)`: separa série temporal em treino/teste sem embaralhar.

---

### 2) Estatística Descritiva (`ibmecstats.descritiva`)

* `summary_stats(x, percentiles=(...), ddof=1)`: resumo estatístico (n, média, desvio, quantis, skew, curtose).
* `freq_table(x, normalize=False, dropna=False, sort=True)`: tabela de frequências absolutas e proporcionais.
* `iqr_outliers(x, k=1.5)`: identifica outliers pela regra do IQR.
* `correlation_matrix(df, method="pearson")`: matriz de correlação numérica (`pearson`, `spearman`, `kendall`).

---

### 3) Inferência Estatística (`ibmecstats.inferencia`)

#### Intervalos de confiança

* `ci_mean(x, alpha=0.05, method="t")`: IC da média (t ou z).
* `ci_mean_diff(x1, x2, alpha=0.05, paired=False, equal_var=False)`: IC da diferença entre médias.
* `ci_proportion(k, n, alpha=0.05, method="wilson")`: IC de uma proporção.
* `ci_proportion_diff(k1, n1, k2, n2, alpha=0.05)`: IC da diferença entre duas proporções.
* `ci_variance(x, alpha=0.05)`: IC da variância usando qui-quadrado.

#### Testes de hipótese

* `t_test_1samp(x, mu0, alternative="two-sided")`: teste t de 1 amostra.
* `t_test_ind(x1, x2, equal_var=False, alternative="two-sided")`: teste t para amostras independentes (Welch por padrão).
* `t_test_paired(x1, x2, alternative="two-sided")`: teste t pareado.
* `z_test_proportion(k, n, p0, alternative="two-sided", continuity=True)`: teste z para uma proporção.
* `z_test_2proportions(k1, n1, k2, n2, alternative="two-sided")`: teste z para comparar duas proporções.
* `chi2_gof(observed, expected=None)`: qui-quadrado de aderência.
* `chi2_independence(table, correction=True)`: qui-quadrado de independência.
* `chi2_homogeneity(table, correction=False)`: qui-quadrado de homogeneidade.
* `anova_oneway(*groups)`: ANOVA de um fator.
* `bartlett_homoscedasticity(*groups)`: teste de Bartlett para igualdade de variâncias.
* `cochran_c_test(*groups, n_sim=20000, random_state=None)`: teste C de Cochran para homocedasticidade.
* `f_test_variances(x1, x2, alternative="two-sided")`: teste F para comparação de duas variâncias.
* `correlation_test(x, y, method="pearson")`: teste de correlação (Pearson, Spearman, Kendall).
* `jarque_bera_test(x)`: teste de normalidade Jarque-Bera.
* `normality_tests(x)`: tabela de testes de normalidade (Shapiro, Anderson-Darling, KS e Jarque-Bera).

#### Reamostragem

* `bootstrap_ci(x, statfunc=np.mean, alpha=0.05, n_boot=10000, random_state=None)`: IC bootstrap percentil para estatística arbitrária.

**Exemplo rápido (`cochran_c_test`)**

```python
import ibmecstats as ibs

g1 = [10, 11, 9, 12, 10]
g2 = [8, 7, 9, 8, 10]
g3 = [12, 15, 13, 14, 16]

out = ibs.cochran_c_test(g1, g2, g3, n_sim=5000, random_state=42)
print(out)  # {'statistic': ..., 'pvalue': ..., 'k_groups': ...}
```

---

### 4) Regressão Linear (`ibmecstats.regressao`)

* `add_dummy_variables(df, columns, drop_first=True)`: cria variáveis dummy para colunas categóricas.
* `ols_fit(y, x, add_constant=True)`: ajusta regressão linear por mínimos quadrados ordinários.
* `ols_predict(model, x_new)`: gera previsões com modelo OLS ajustado.
* `ols_diagnostics(model)`: retorna métricas de ajuste (R², R² ajustado, F, AIC, BIC, sigma).
* `best_subset_selection(y, x, criterion="aic", max_features=None, add_constant=True)`: seleciona melhor subconjunto de variáveis (AIC/BIC).
* `model_selection(y, x, method="forward", criterion="aic", add_constant=True)`: seleção de modelos (`forward`, `backward`, `stepwise`, `bestsubset`).

**Exemplo rápido (`ols_fit` e `model_selection`)**

```python
import pandas as pd
import ibmecstats as ibs

df = pd.DataFrame({
    "y": [10, 11, 13, 14, 16, 18],
    "x1": [1, 2, 3, 4, 5, 6],
    "x2": [2, 1, 2, 3, 2, 4],
    "grupo": ["A", "A", "B", "B", "A", "B"],
})

x = ibs.add_dummy_variables(df[["x1", "x2", "grupo"]], columns=["grupo"], drop_first=True)
model = ibs.ols_fit(df["y"], x, add_constant=True)
print(ibs.ols_diagnostics(model))

sel = ibs.model_selection(df["y"], x, method="stepwise", criterion="aic")
print(sel["features"])
```

---

### 5) Métodos de Previsão (`ibmecstats.previsao`)

Todos retornam `ForecastResult`.

* `naive_forecast(y, h, seasonal_period=None)`: previsão ingênua (último valor ou sazonal ingênua).
* `drift_forecast(y, h)`: projeção com tendência linear entre primeiro e último ponto.
* `moving_average_forecast(y, h, window=3)`: previsão por média móvel simples.
* `trend_projection_forecast(y, h, model="linear")`: projeção por tendência via MQ (`linear`, `quadratic`, `exponential`).
* `autoregressive_forecast(y, h, lags=1, trend="c", seasonal=False, period=None)`: previsão AR(p) por Yule-Walker.
* `ses_forecast(y, h, alpha=None, optimized=True)`: suavização exponencial simples (statsmodels).
* `holt_forecast(...)`: suavização de Holt (nível + tendência) via statsmodels.
* `holt_winters_forecast(...)`: Holt-Winters (tendência + sazonalidade) via statsmodels.

**Exemplo rápido (`trend_projection_forecast`)**

```python
import ibmecstats as ibs

y = [120, 124, 129, 133, 140, 147, 155]
res = ibs.trend_projection_forecast(y, h=3, model="quadratic")
print(res.yhat)
```

---

### 6) Métricas de Forecast (`ibmecstats.metrics`)

* `mae(y_true, y_pred)`: erro absoluto médio.
* `mse(y_true, y_pred)`: erro quadrático médio.
* `rmse(y_true, y_pred)`: raiz do erro quadrático médio.
* `mape(y_true, y_pred, eps=1e-9)`: erro percentual absoluto médio.
* `smape(y_true, y_pred, eps=1e-9)`: erro percentual absoluto médio simétrico.
* `wape(y_true, y_pred, eps=1e-9)`: erro percentual absoluto ponderado.
* `mase(y_true, y_pred, y_train, seasonal_period=1, eps=1e-9)`: erro absoluto escalonado.
* `forecast_accuracy(y_true, y_pred, y_train=None, seasonal_period=1)`: tabela com métricas resumidas.

---

### 7) Gráficos (`ibmecstats.plots`) — seaborn

* `set_theme(...)`: define estilo visual padrão.
* `plot_distribution(x, bins="auto", kde=True, title=None)`: histograma com opção de KDE.
* `plot_boxplot(x, title=None)`: boxplot.
* `plot_qq(x, dist="norm", title=None)`: gráfico QQ.
* `plot_pp(x, dist="norm", title=None)`: gráfico PP.
* `plot_time_series(y, title=None)`: série temporal.
* `plot_acf_pacf(y, lags=40, title=None)`: ACF/PACF (requer statsmodels).
* `plot_forecast(y_train, y_test, y_pred, title=None)`: gráfico de treino, teste e previsão.
* `plot_residuals(residuals, title=None)`: diagnóstico visual de resíduos.
