"""
Type definition for model parameters
"""
from pydantic import BaseModel, Extra, root_validator, validator
from pydantic.dataclasses import dataclass

from datetime import date
from typing import Any, Dict, List, Optional, Union

from autumn.settings.constants import COVID_BASE_DATETIME, GOOGLE_MOBILITY_LOCATIONS, COVID_BASE_AGEGROUPS
from autumn.core.inputs.social_mixing.constants import LOCATIONS

BASE_DATE = COVID_BASE_DATETIME.date()

# Forbid additional arguments to prevent extraneous parameter specification
BaseModel.Config.extra = Extra.forbid

"""
Commonly used checking processes
"""


def get_check_prop(name):

    msg = f"Parameter '{name}' not in domain [0, 1], but is intended as a proportion"

    def check_prop(value: float) -> float:
        assert 0.0 <= value <= 1.0, msg
        return value

    return check_prop


def get_check_non_neg(name):

    msg = f"Parameter '{name}' is negative, but is intended to be non-negative"

    def check_non_neg(value: float) -> float:
        assert 0.0 <= value, msg
        return value

    return check_non_neg


def get_check_all_prop(name):

    msg = f"Parameter '{name}' contains values outside [0, 1], but is intended as a list of proportions"

    def check_all_pos(values: list) -> float:
        assert all([0.0 <= i_value <= 1.0 for i_value in values]), msg
        return values

    return check_all_pos


def get_check_all_non_neg(name):

    msg = f"Parameter '{name}' contains negative values, but is intended as a list of proportions"

    def check_all_non_neg(values: list) -> float:
        assert all([0.0 <= i_value for i_value in values]), msg
        return values

    return check_all_non_neg


def get_check_all_dict_values_non_neg(name):

    msg = f"Dictionary parameter '{name}' contains negative values, but is intended as a list of proportions"

    def check_non_neg_values(dict_param: dict) -> float:
        assert all([0.0 <= i_value for i_value in dict_param.values()]), msg
        return dict_param

    return check_non_neg_values


def get_check_all_non_neg_if_present(name):

    msg = f"Parameter '{name}' contains negative values, but is intended as a list of proportions"

    def check_all_non_neg(values: float) -> float:
        if values:
            assert all([0.0 <= i_value for i_value in values]), msg
        return values

    return check_all_non_neg


"""
Parameter validation models
"""


class Time(BaseModel):
    """
    Parameters to define the model time period and evaluation steps.
    """

    start: float
    end: float
    step: float

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        start, end = values.get("start"), values.get("end")
        assert end >= start, f"End time: {end} before start: {start}"
        return values


class TimeSeries(BaseModel):
    """
    A set of values with associated time points.
    """

    times: List[float]
    values: List[float]

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, inputs):
        value_series, time_series = inputs.get("values"), inputs.get("times")
        msg = f"TimeSeries length mismatch, times length: {len(time_series)}, values length: {len(value_series)}"
        assert len(time_series) == len(value_series), msg
        return inputs

    @validator("times", pre=True, allow_reuse=True)
    def parse_dates_to_days(dates):
        return [(d - BASE_DATE).days if isinstance(d, date) else d for d in dates]


class Country(BaseModel):
    """
    The country that the model is based in. (The country may be, and often is, the same as the region.)
    """

    iso3: str

    @validator("iso3", pre=True, allow_reuse=True)
    def check_length(iso3):
        assert len(iso3) == 3, f"ISO3 codes should have three digits, code is: {iso3}"
        return iso3


class Population(BaseModel):
    """
    Model population parameters.
    """

    region: Optional[str]  # None/null means default to parent country
    year: int  # Year to use to find the population data in the database

    @validator("year", pre=True, allow_reuse=True)
    def check_year(year):
        msg = f"Year before 1900 or after 2050: {year}"
        assert 1900 <= year <= 2050, msg
        return year


class CompartmentSojourn(BaseModel):
    """
    Compartment sojourn times, i.e. the mean period of time spent in a compartment.
    """

    total_time: float
    proportion_early: Optional[float]

    check_total_positive = validator("total_time", allow_reuse=True)(get_check_non_neg("total_time"))
    check_prop_early = validator("proportion_early", allow_reuse=True)(get_check_prop("proportion_early"))


class Sojourns(BaseModel):
    """
    Parameters for determining how long a person stays in a given compartment.
    """

    active: float
    latent: float

class MixingLocation(BaseModel):

    append: bool  # Whether to append or overwrite times / values
    times: List[int]  # Times for dynamic mixing func
    values: List[Any]  # Values for dynamic mixing func

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        value_series, time_series = values.get("values"), values.get("times")
        assert len(time_series) == len(value_series), f"Mixing series length mismatch."
        return values

    @validator("times", pre=True, allow_reuse=True)
    def parse_dates_to_days(dates):
        return [(d - BASE_DATE).days if isinstance(d, date) else d for d in dates]


class Mobility(BaseModel):

    region: Optional[str]  # None/null means default to parent country
    mixing: Dict[str, MixingLocation]
    age_mixing: Optional[Dict[str, TimeSeries]]
    smooth_google_data: bool
    square_mobility_effect: bool
    google_mobility_locations: Dict[str, Dict[str, float]]

    @validator("google_mobility_locations", allow_reuse=True)
    def check_location_weights(val):
        for location in val:
            location_total = sum(val[location].values())
            msg = f"Mobility weights don't sum to one: {location_total}"
            assert abs(location_total - 1.0) < 1e-6, msg
            msg = "Google mobility key not recognised"
            assert all(
                [key in GOOGLE_MOBILITY_LOCATIONS for key in val[location].keys()]
            ), msg
        return val


class AgeSpecificProps(BaseModel):

    values: Dict[int, float]
    source_immunity_distribution: Dict[str, float]
    source_immunity_protection: Dict[str, float]
    multiplier: float

    @validator("source_immunity_distribution", allow_reuse=True)
    def check_source_dist(vals):
        sum_of_values = sum(vals.values())
        msg = f"Proportions by immunity status in source for parameters does not sum to one: {sum_of_values}"
        assert sum_of_values == 1.0, msg
        return vals

    @validator("source_immunity_protection", allow_reuse=True)
    def check_source_dist(protection_params):
        msg = "Source protection estimates not proportions"
        assert (
            all([0.0 <= val <= 1.0 for val in protection_params.values()]) == 1.0
        ), msg
        return protection_params

    check_props = validator("source_immunity_distribution", allow_reuse=True)(
        get_check_all_dict_values_non_neg("source_immunity_distribution")
    )
    check_protections = validator("source_immunity_protection", allow_reuse=True)(
        get_check_all_dict_values_non_neg("source_immunity_protection")
    )


class AgeStratification(BaseModel):
    """
    Parameters used in age based stratification.
    """

    susceptibility: Optional[
        Union[Dict[int, float], float]
    ]  # Dictionary that represents each age group, single float or None
    prop_symptomatic: Optional[Union[Dict[int, float], float]]  # As for susceptibility
    prop_hospital: AgeSpecificProps
    cfr: AgeSpecificProps


class ImmunityRiskReduction(BaseModel):

    none: float
    low: float
    high: float

    check_none = validator("none", allow_reuse=True)(get_check_prop("none"))
    check_low = validator("low", allow_reuse=True)(get_check_prop("low"))
    check_high = validator("high", allow_reuse=True)(get_check_prop("high"))

    @root_validator(pre=True, allow_reuse=True)
    def check_progression(cls, values):
        msg = "Immunity stratification effects are not increasing"
        assert values["none"] <= values["low"] <= values["high"], msg
        return values


class ImmunityStratification(BaseModel):

    prop_immune: float
    prop_high_among_immune: float
    infection_risk_reduction: ImmunityRiskReduction

    check_prop_immune = validator("prop_immune", allow_reuse=True)(get_check_prop("prop_immune"))
    check_high_immune = validator("prop_high_among_immune", allow_reuse=True)(
        get_check_prop("prop_high_among_immune")
    )
    

class TimeDistribution(BaseModel):

    distribution: str
    parameters: dict

    @validator("distribution", allow_reuse=True)
    def check_distribution(distribution):
        supported_distributions = ("gamma",)
        msg = f"Requested time distribution not supported: {distribution}"
        assert distribution in supported_distributions, msg
        return distribution


class TimeToEvent(BaseModel):

    hospitalisation: TimeDistribution
    icu_admission: TimeDistribution
    death: TimeDistribution


class HospitalStay(BaseModel):

    hospital_all: TimeDistribution
    icu: TimeDistribution


class RandomProcessParams(BaseModel):

    coefficients: Optional[List[float]]
    noise_sd: Optional[float]
    values: Optional[List[float]]
    order: int
    time: Time


class ParamConfig:
    """
    Config for parameter models.
    """

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable


@dataclass(config=ParamConfig)
class Parameters:
    # Metadata
    description: Optional[str]
    country: Country
    population: Population
    ref_mixing_iso3: str
    age_groups: List[int]
    time: Time
    # Values
    contact_rate: float
    infectious_seed: float
    sojourns: Sojourns
    is_dynamic_mixing_matrix: bool
    mobility: Mobility
   
    time_from_onset_to_event: TimeToEvent
    hospital_stay: HospitalStay
    prop_icu_among_hospitalised: float

    age_stratification: AgeStratification
    immunity_stratification: ImmunityStratification

    # Random process
    activate_random_process: bool
    random_process: Optional[RandomProcessParams] 

    # Output-related
    requested_cumulative_outputs: List[str]
    cumulative_start_time: Optional[float]
    request_incidence_by_age: bool

    @validator("age_groups", allow_reuse=True)
    def validate_age_groups(age_groups):
        msg = "Not all requested age groups in the available age groups of 5-year increments from zero to 75"
        int_age_groups = [int(i_group) for i_group in COVID_BASE_AGEGROUPS]
        assert all([i_group in int_age_groups for i_group in age_groups]), msg
        return age_groups