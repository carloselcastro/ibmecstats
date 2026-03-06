import ibmecstats as ibs
import numpy as np
import pytest


def test_ci_mean():
    lo, hi = ibs.ci_mean([1, 2, 3, 4, 5], alpha=0.05, method="t")
    assert lo < hi


def test_t_test_1samp():
    out = ibs.t_test_1samp([1, 2, 3, 4, 5], mu0=3.0)
    assert "statistic" in out and "pvalue" in out
    assert 0.0 <= out["pvalue"] <= 1.0


def test_normality_tests():
    df = ibs.normality_tests([1, 2, 3, 4, 5, 6, 7])
    assert "test" in df.columns
    assert df.shape[0] >= 2
    assert "Jarque-Bera" in set(df["test"].tolist())


def test_inferencia_additional_methods():
    x1 = [10, 12, 11, 13, 12, 14]
    x2 = [9, 10, 11, 10, 9, 11]

    out_corr = ibs.correlation_test(x1, x2, method="pearson")
    assert "statistic" in out_corr and "pvalue" in out_corr

    out_jb = ibs.jarque_bera_test(x1)
    assert 0.0 <= out_jb["pvalue"] <= 1.0

    out_f = ibs.f_test_variances(x1, x2)
    assert out_f["df1"] > 0 and out_f["df2"] > 0

    lo_v, hi_v = ibs.ci_variance(x1, alpha=0.05)
    assert lo_v < hi_v

    lo_d, hi_d = ibs.ci_mean_diff(x1, x2, alpha=0.05, paired=False)
    assert lo_d < hi_d

    lo_p, hi_p = ibs.ci_proportion_diff(30, 100, 20, 100)
    assert lo_p < hi_p

    out_2p = ibs.z_test_2proportions(30, 100, 20, 100)
    assert "z" in out_2p and 0.0 <= out_2p["pvalue"] <= 1.0

    out_h = ibs.chi2_homogeneity([[20, 30], [25, 25]])
    assert out_h["df"] > 0

    out_b = ibs.bartlett_homoscedasticity(x1, x2)
    assert "statistic" in out_b and "pvalue" in out_b

    out_c = ibs.cochran_c_test(x1, x2, n_sim=1000, random_state=42)
    assert "statistic" in out_c and 0.0 <= out_c["pvalue"] <= 1.0


def test_inferencia_validation_errors():
    with pytest.raises(ValueError):
        ibs.ci_mean([1], method="t")
    with pytest.raises(ValueError):
        ibs.ci_mean([1, 2, 3], method="bad")

    with pytest.raises(ValueError):
        ibs.ci_proportion(k=5, n=0)
    with pytest.raises(ValueError):
        ibs.ci_proportion(k=5, n=10, method="bad")

    with pytest.raises(ValueError):
        ibs.z_test_proportion(k=2, n=10, p0=1.0)
    with pytest.raises(ValueError):
        ibs.z_test_2proportions(k1=1, n1=10, k2=1, n2=10, alternative="bad")

    with pytest.raises(ValueError):
        ibs.f_test_variances([1], [1, 2])
    with pytest.raises(ValueError):
        ibs.f_test_variances([1, 2], [1, 2], alternative="bad")

    with pytest.raises(ValueError):
        ibs.correlation_test([1, 2, 3], [1, 2, 3], method="bad")

    with pytest.raises(ValueError):
        ibs.ci_mean_diff([1, 2, 3], [1, 2], paired=True)
    with pytest.raises(ValueError):
        ibs.ci_proportion_diff(1, 0, 2, 10)

    with pytest.raises(ValueError):
        ibs.cochran_c_test([1, 2], n_sim=100)


def test_inferencia_additional_branches_and_modes():
    x1 = [10, 12, 11, 13, 12, 14]
    x2 = [9, 10, 11, 10, 9, 11]

    # ci_proportion modos faltantes
    lo_w, hi_w = ibs.ci_proportion(30, 100, method="wald")
    lo_ac, hi_ac = ibs.ci_proportion(30, 100, method="ac")
    assert lo_w < hi_w
    assert lo_ac < hi_ac

    # t tests branches
    out_ind = ibs.t_test_ind(x1, x2, equal_var=True, alternative="greater")
    out_pair = ibs.t_test_paired([1, 2, 3], [1, 2, 2], alternative="greater")
    assert 0.0 <= out_ind["pvalue"] <= 1.0
    assert 0.0 <= out_pair["pvalue"] <= 1.0

    # z proportion alternatives
    out_zg = ibs.z_test_proportion(30, 100, 0.2, alternative="greater", continuity=False)
    out_zl = ibs.z_test_proportion(30, 100, 0.4, alternative="less", continuity=True)
    assert 0.0 <= out_zg["pvalue"] <= 1.0
    assert 0.0 <= out_zl["pvalue"] <= 1.0

    # chi2_gof expected explícito
    out_gof = ibs.chi2_gof([10, 20, 30], expected=[12, 18, 30])
    assert out_gof["df"] == 2.0

    # f_test branches
    out_fg = ibs.f_test_variances(x1, x2, alternative="greater")
    out_fl = ibs.f_test_variances(x1, x2, alternative="less")
    assert 0.0 <= out_fg["pvalue"] <= 1.0
    assert 0.0 <= out_fl["pvalue"] <= 1.0

    # ci_mean_diff branches
    lo_p, hi_p = ibs.ci_mean_diff([2, 3, 4], [1, 2, 3], paired=True)
    lo_ev, hi_ev = ibs.ci_mean_diff(x1, x2, paired=False, equal_var=True)
    assert lo_p <= hi_p
    assert lo_ev < hi_ev

    # z_test_2proportions branches
    out_2g = ibs.z_test_2proportions(35, 100, 20, 100, alternative="greater")
    out_2l = ibs.z_test_2proportions(15, 100, 20, 100, alternative="less")
    assert 0.0 <= out_2g["pvalue"] <= 1.0
    assert 0.0 <= out_2l["pvalue"] <= 1.0

    # bootstrap path
    out_boot = ibs.bootstrap_ci([1, 2, 3, 4, 5], n_boot=200, random_state=42)
    assert out_boot["lower"] <= out_boot["estimate"] <= out_boot["upper"]


def test_inferencia_error_edges():
    with pytest.raises(ValueError):
        ibs.t_test_paired([1, 2], [1, 2, 3])

    with pytest.raises(ValueError):
        ibs.chi2_gof([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        ibs.chi2_gof([1, 2, 3], expected=[1, 2])

    with pytest.raises(ValueError):
        ibs.bartlett_homoscedasticity([1, 2, 3])

    with pytest.raises(ValueError):
        ibs.ci_variance([1])
    with pytest.raises(ValueError):
        ibs.ci_mean_diff([1], [2], paired=False)

    with pytest.raises(ValueError):
        ibs.z_test_2proportions(0, 10, 0, 10)

    with pytest.raises(ValueError):
        ibs.normality_tests([1, 2])
    out_const = ibs.normality_tests([5, 5, 5, 5, 5])
    row_ks = out_const.loc[out_const["test"] == "KS vs N(mu, sigma)"].iloc[0]
    assert np.isnan(row_ks["statistic"])

    with pytest.raises(ValueError):
        ibs.bootstrap_ci([1])
