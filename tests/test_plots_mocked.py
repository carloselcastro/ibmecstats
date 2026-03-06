import pytest

import ibmecstats.plots as pl


class _DummyAxes:
    def __init__(self):
        self.title = None

    def set_xlabel(self, *_args, **_kwargs):
        return None

    def set_ylabel(self, *_args, **_kwargs):
        return None

    def set_title(self, title):
        self.title = title

    def plot(self, *_args, **_kwargs):
        return None

    def axhline(self, *_args, **_kwargs):
        return None


class _DummyFig:
    def add_subplot(self, *_args, **_kwargs):
        return _DummyAxes()


def test_plots_module_with_mocks(monkeypatch):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.graphics.tsaplots as tsaplots
    from scipy import stats

    monkeypatch.setattr(sns, "set_theme", lambda **kwargs: None)
    monkeypatch.setattr(sns, "histplot", lambda *args, **kwargs: _DummyAxes())
    monkeypatch.setattr(sns, "boxplot", lambda *args, **kwargs: _DummyAxes())
    monkeypatch.setattr(sns, "lineplot", lambda *args, **kwargs: _DummyAxes())
    monkeypatch.setattr(plt, "figure", lambda: _DummyFig())
    monkeypatch.setattr(stats, "probplot", lambda *args, **kwargs: None)
    monkeypatch.setattr(tsaplots, "plot_acf", lambda *args, **kwargs: None)
    monkeypatch.setattr(tsaplots, "plot_pacf", lambda *args, **kwargs: None)

    pl.set_theme()
    ax1 = pl.plot_distribution([1, 2, 3], title="d")
    ax2 = pl.plot_boxplot([1, 2, 3], title="b")
    ax3 = pl.plot_qq([1, 2, 3, 4], title="q")
    ax4 = pl.plot_pp([1, 2, 3, 4], title="p")
    ax5 = pl.plot_pp([1, 2, 3, 4], dist="expon")
    ax6 = pl.plot_time_series([1, 2, 3], title="t")
    a1, a2 = pl.plot_acf_pacf([1, 2, 3, 4, 5], lags=2, title="acf")
    ax7 = pl.plot_forecast([1, 2], [3], [3], title="f")
    r1, r2 = pl.plot_residuals([1, -1, 0], title="r")

    assert isinstance(ax1, _DummyAxes)
    assert isinstance(ax2, _DummyAxes)
    assert isinstance(ax3, _DummyAxes)
    assert isinstance(ax4, _DummyAxes)
    assert isinstance(ax5, _DummyAxes)
    assert isinstance(ax6, _DummyAxes)
    assert isinstance(a1, _DummyAxes)
    assert isinstance(a2, _DummyAxes)
    assert isinstance(ax7, _DummyAxes)
    assert isinstance(r1, _DummyAxes)
    assert isinstance(r2, _DummyAxes)


def test_plot_acf_pacf_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "statsmodels.graphics.tsaplots":
            raise ImportError("mocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        pl.plot_acf_pacf([1, 2, 3, 4])
