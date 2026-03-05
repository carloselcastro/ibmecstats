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

### 1) Estatística Descritiva (`ibmecstats.descritiva`)

#### `summary_stats(x, percentiles=(...), ddof=1) -> pd.Series`

Resumo de estatísticas:

* n, média, desvio-padrão, min/max
* quantis informados
* assimetria (skew) e curtose

**Exemplo**

```python
ibs.summary_stats(x, percentiles=(0.25, 0.5, 0.75))
```

---

#### `freq_table(x, normalize=False, dropna=False, sort=True) -> pd.DataFrame`

Tabela de frequência para variáveis categóricas.

**Parâmetros**

* `normalize`: adiciona proporção
* `dropna`: inclui/exclui NaN
* `sort`: ordena por frequência

**Exemplo**

```python
ibs.freq_table(["A","A","B"], normalize=True)
```

---

#### `iqr_outliers(x, k=1.5) -> pd.DataFrame`

Detecta outliers pela regra do IQR.

**Parâmetros**

* `k`: multiplicador do IQR (1.5 padrão)

**Exemplo**

```python
ibs.iqr_outliers(x, k=1.5)
```

---

#### `correlation_matrix(df, method="pearson") -> pd.DataFrame`

Matriz de correlação para colunas numéricas.

**method**

* `"pearson"`, `"spearman"`, `"kendall"`

**Exemplo**

```python
ibs.correlation_matrix(df, method="spearman")
```

---

### 2) Inferência Estatística (`ibmecstats.inferencia`)

#### `ci_mean(x, alpha=0.05, method="t") -> (lower, upper)`

IC para a média.

* `method="t"`: t-student (recomendado)
* `method="z"`: normal

**Exemplo**

```python
ibs.ci_mean(x, alpha=0.01, method="t")
```

---

#### `ci_proportion(k, n, alpha=0.05, method="wilson") -> (lower, upper)`

IC para proporção.

**method**

* `"wald"` (clássico)
* `"wilson"` (recomendado)
* `"agresti-coull"`

**Exemplo**

```python
ibs.ci_proportion(k=42, n=100, method="wilson")
```

---

#### Testes t

##### `t_test_1samp(x, mu0, alternative="two-sided") -> dict`

Teste t de 1 amostra.

**alternative**

* `"two-sided"`, `"less"`, `"greater"`

**Exemplo**

```python
ibs.t_test_1samp(x, mu0=10, alternative="greater")
```

##### `t_test_ind(x1, x2, equal_var=False, alternative="two-sided") -> dict`

Teste t de 2 amostras independentes.

* `equal_var=False` usa Welch (recomendado)

##### `t_test_paired(x1, x2, alternative="two-sided") -> dict`

Teste t pareado.

---

#### Proporção (teste z)

##### `z_test_proportion(k, n, p0, alternative="two-sided", continuity=True) -> dict`

Teste z para proporção.

---

#### Qui-quadrado

##### `chi2_gof(observed, expected=None) -> dict`

Teste de aderência (GOF).

* `expected=None` assume equiprovável.

##### `chi2_independence(table, correction=True) -> dict`

Teste de independência em tabela de contingência.

---

#### ANOVA

##### `anova_oneway(*groups) -> dict`

ANOVA de um fator.

---

#### Normalidade

##### `normality_tests(x) -> pd.DataFrame`

Retorna tabela com:

* Shapiro-Wilk
* Anderson-Darling (normal)
* KS vs N(mu, sigma) *(observação: KS com parâmetros estimados não é Lilliefors)*

---

#### Bootstrap

##### `bootstrap_ci(x, statfunc=np.mean, alpha=0.05, n_boot=10000, random_state=None) -> dict`

IC bootstrap percentil para qualquer estatística.

---

### 3) Métodos de Previsão (`ibmecstats.previsao`)

Todos retornam um `ForecastResult` com:

* `yhat`: previsão (pd.Series)
* `fitted` (quando aplicável)
* `residuals` (quando aplicável)
* `info`: metadados do modelo

#### `naive_forecast(y, h, seasonal_period=None) -> ForecastResult`

* sem sazonalidade: repete último valor
* com sazonalidade: repete o último ciclo sazonal

#### `drift_forecast(y, h) -> ForecastResult`

Método drift (tendência linear entre o primeiro e último ponto).

#### `moving_average_forecast(y, h, window=3) -> ForecastResult`

Média móvel simples.

#### `ses_forecast(y, h, alpha=None, optimized=True) -> ForecastResult`

SES (Simple Exponential Smoothing) via statsmodels.

#### `holt_forecast(y, h, damped_trend=False, optimized=True, smoothing_level=None, smoothing_trend=None) -> ForecastResult`

Holt (nível + tendência) via statsmodels.

#### `holt_winters_forecast(y, h, seasonal_periods, trend="add", seasonal="add", damped_trend=False, optimized=True) -> ForecastResult`

Holt-Winters (tendência + sazonalidade) via statsmodels.

---

### 4) Métricas de Forecast (`ibmecstats.metrics`)

* `mae(y_true, y_pred)`
* `mse(y_true, y_pred)`
* `rmse(y_true, y_pred)`
* `mape(y_true, y_pred, eps=1e-9)`
* `smape(y_true, y_pred, eps=1e-9)`
* `wape(y_true, y_pred, eps=1e-9)`
* `mase(y_true, y_pred, y_train, seasonal_period=1, eps=1e-9)`
* `forecast_accuracy(y_true, y_pred, y_train=None, seasonal_period=1) -> pd.DataFrame`

**Exemplo**

```python
train, test = ibs.train_test_split_time(y, test_size=0.2)
res = ibs.naive_forecast(train, h=len(test))
ibs.forecast_accuracy(test, res.yhat, y_train=train, seasonal_period=12)
```

---

### 5) Gráficos (`ibmecstats.plots`) — seaborn

* `set_theme(style="whitegrid", context="talk", palette="deep", font_scale=1.0, rc=None)`
* `plot_distribution(x, bins="auto", kde=True, title=None)`
* `plot_boxplot(x, title=None)`
* `plot_qq(x, dist="norm", title=None)`
* `plot_pp(x, dist="norm", title=None)`
* `plot_time_series(y, title=None)`
* `plot_acf_pacf(y, lags=40, title=None)` *(requer statsmodels)*
* `plot_forecast(y_train, y_test, y_pred, title=None)`
* `plot_residuals(residuals, title=None)`


