import numpy as np
from numpy.testing import assert_allclose

from apps.covid_19.model.parameters import Country
from apps.covid_19.model.preprocess.mixing_matrix.mixing_adjusters import LocationMixingAdjuster
from apps.covid_19.model.preprocess.mixing_matrix.mixing_adjusters import location_adjuster


MM = np.ones([16, 16])
HOME_MM = MM * 0.1
OTHER_LOCATIONS_MM = MM * 0.2
SCHOOL_MM = MM * 0.3
WORK_MM = MM * 0.6


def test_location_adjuster__with_no_data():
    """
    Ensure there is no change if no mixing data has been suplied.
    """
    country = Country(iso3="AUS")
    mobility_funcs = {}
    microdistancing_funcs = {}
    adjuster = LocationMixingAdjuster(country, mobility_funcs, microdistancing_funcs)
    mm = np.ones([16, 16])
    adj_mm = adjuster.get_adjustment(0, mm)
    assert_allclose(mm, adj_mm, atol=0.01, verbose=True)


def test_location_adjuster__with_only_mobility_data(monkeypatch):
    monkeypatch.setattr(location_adjuster, "get_country_mixing_matrix", _get_country_mixing_matrix)
    country = Country(iso3="AUS")
    mobility_funcs = {"work": lambda t: 0.3 * t, "school": lambda t: 0.2 * t}
    microdistancing_funcs = {}
    adjuster = LocationMixingAdjuster(country, mobility_funcs, microdistancing_funcs)
    mm = np.ones([16, 16])
    adj_mm = adjuster.get_adjustment(1, mm)
    work_component = WORK_MM * (0.3 - 1)
    school_component = SCHOOL_MM * (0.2 - 1)
    expect_mm = MM + work_component + school_component
    assert_allclose(expect_mm, adj_mm, atol=0.01, verbose=True)


def test_location_adjuster__with_only_microdistancing_data(monkeypatch):
    monkeypatch.setattr(location_adjuster, "get_country_mixing_matrix", _get_country_mixing_matrix)
    country = Country(iso3="AUS")
    mobility_funcs = {}
    microdistancing_funcs = {"work": lambda t: 0.3 * t, "school": lambda t: 0.2 * t}
    adjuster = LocationMixingAdjuster(country, mobility_funcs, microdistancing_funcs)
    mm = np.ones([16, 16])
    adj_mm = adjuster.get_adjustment(1, mm)
    work_component = WORK_MM * (0.3 - 1)
    school_component = SCHOOL_MM * (0.2 - 1)
    expect_mm = MM + work_component + school_component
    assert_allclose(expect_mm, adj_mm, atol=0.01, verbose=True)


def test_location_adjuster__with_microdistancing_and_mobility_data(monkeypatch):
    monkeypatch.setattr(location_adjuster, "get_country_mixing_matrix", _get_country_mixing_matrix)
    country = Country(iso3="AUS")
    mobility_funcs = {"work": lambda t: 0.3 * t, "home": lambda t: 0.5}
    microdistancing_funcs = {"school": lambda t: 0.2 * t, "home": lambda t: 0.7}
    adjuster = LocationMixingAdjuster(country, mobility_funcs, microdistancing_funcs)
    mm = np.ones([16, 16])
    adj_mm = adjuster.get_adjustment(1, mm)
    work_component = WORK_MM * (0.3 - 1)
    school_component = SCHOOL_MM * (0.2 - 1)
    home_component = HOME_MM * (0.5 * 0.7 - 1)
    expect_mm = MM + work_component + school_component + home_component
    assert_allclose(expect_mm, adj_mm, atol=0.01, verbose=True)


def _get_country_mixing_matrix(sheet_type, iso3):
    if sheet_type == "home":
        return HOME_MM
    if sheet_type == "other_locations":
        return OTHER_LOCATIONS_MM
    if sheet_type == "school":
        return SCHOOL_MM
    if sheet_type == "work":
        return WORK_MM
    else:
        return MM