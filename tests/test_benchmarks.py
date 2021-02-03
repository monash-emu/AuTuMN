import pytest

from apps import covid_19


@pytest.mark.benchmark
@pytest.mark.github_only
@pytest.mark.parametrize("region", covid_19.app.region_names)
def test_benchmark_covid_models(region, benchmark):
    """
    Performance benchmark: check how long our models take to run.
    See: https://pytest-benchmark.readthedocs.io/en/stable/
    Run these with pytest -vv -m benchmark --benchmark-json benchmark.json
    """
    benchmark(_run_covid_model, region=region)


def _run_covid_model(region):
    region_app = covid_19.app.get_region(region)
    model = region_app.build_model(region_app.params["default"])
    model.run()
