import pytest

from summer2.examples import EXAMPLES


@pytest.mark.parametrize("name, module", EXAMPLES.items())
def test_summer_examples(name, module, monkeypatch):
    model = module.build_model()
    monkeypatch.setattr(module, "plot_timeseries", _plot_timeseries)
    module.plot_outputs(model)


def _plot_timeseries(title, times, values):
    assert title
    assert all([len(v) == len(times) for v in values.values()])
