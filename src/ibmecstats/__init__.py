from .descritiva import correlation_matrix, freq_table, iqr_outliers, summary_stats
from .inferencia import (
    anova_oneway,
    bootstrap_ci,
    chi2_gof,
    chi2_independence,
    ci_mean,
    ci_proportion,
    normality_tests,
    t_test_1samp,
    t_test_ind,
    t_test_paired,
    z_test_proportion,
)
from .metrics import forecast_accuracy, mae, mase, mape, mse, rmse, smape, wape
from .previsao import (
    drift_forecast,
    holt_forecast,
    holt_winters_forecast,
    moving_average_forecast,
    naive_forecast,
    ses_forecast,
)
from .plots import (
    plot_acf_pacf,
    plot_boxplot,
    plot_distribution,
    plot_forecast,
    plot_pp,
    plot_qq,
    plot_residuals,
    plot_time_series,
    set_theme,
)
from .utils import ForecastResult, ensure_datetime_index, train_test_split_time

__all__ = [
    # utils
    "ForecastResult",
    "ensure_datetime_index",
    "train_test_split_time",
    # descritiva
    "summary_stats",
    "freq_table",
    "iqr_outliers",
    "correlation_matrix",
    # inferencia
    "ci_mean",
    "ci_proportion",
    "t_test_1samp",
    "t_test_ind",
    "t_test_paired",
    "z_test_proportion",
    "chi2_gof",
    "chi2_independence",
    "anova_oneway",
    "normality_tests",
    "bootstrap_ci",
    # previsao
    "naive_forecast",
    "drift_forecast",
    "moving_average_forecast",
    "ses_forecast",
    "holt_forecast",
    "holt_winters_forecast",
    # metrics
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "wape",
    "mase",
    "forecast_accuracy",
    # plots
    "set_theme",
    "plot_distribution",
    "plot_boxplot",
    "plot_qq",
    "plot_pp",
    "plot_time_series",
    "plot_acf_pacf",
    "plot_forecast",
    "plot_residuals",
]