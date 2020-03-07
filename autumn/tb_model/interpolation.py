"""
Utilities for generating interpolated functions from input data.
"""
from ..curve import scale_up_function
from ..db import get_bcg_coverage, get_crude_birth_rate


def get_bcg_functions(_tb_model, _input_database, _country_iso3, start_year=1955):
    """
    function to obtain the bcg coverage from the input database and add the appropriate functions to the tb model

    :param _tb_model: StratifiedModel class
        SUMMER model object to be assigned bcg vaccination coverage functions
    :param _input_database: sql database
        database containing the TB data to extract the bcg coverage from
    :param _country_iso3: string
        iso3 code for country of interest
    :param start_year: int
        year in which bcg vaccination was assumed to have started at a significant programmatic level for the country
    :return: StratifiedModel class
        SUMMER model object with bcg vaccination functions added
    """

    # create dictionary of data
    bcg_coverage = get_bcg_coverage(_input_database, _country_iso3)
    bcg_coverage[start_year] = 0.0

    # fit function
    bcg_coverage_function = scale_up_function(
        bcg_coverage.keys(), bcg_coverage.values(), smoothness=0.2, method=5
    )

    # add to model object and return
    _tb_model.time_variants["bcg_coverage"] = bcg_coverage_function
    _tb_model.time_variants["bcg_coverage_complement"] = lambda value: 1.0 - bcg_coverage_function(
        value
    )
    return _tb_model


def add_birth_rate_functions(_tb_model, _input_database, _country_iso3):
    """
    Add crude birth rate function to existing epidemiological model

    :param _tb_model: EpiModel or StratifiedModel class
        SUMMER model object to be assigned bcg vaccination coverage functions
    :param _input_database: sql database
        database containing the TB data to extract the bcg coverage from
    :param _country_iso3: string
        iso3 code for country of interest
    :return: EpiModel or StratifiedModel class
        SUMMER model object with birth rate function added
    """
    crude_birth_rate_data = get_crude_birth_rate(_input_database, _country_iso3)
    if _country_iso3 == "MNG":  # provisional patch
        for year in crude_birth_rate_data.keys():
            if year > 1990.0:
                crude_birth_rate_data[year] = 0.04

    _tb_model.time_variants["crude_birth_rate"] = scale_up_function(
        crude_birth_rate_data.keys(), crude_birth_rate_data.values(), smoothness=0.2, method=5
    )
    return _tb_model
