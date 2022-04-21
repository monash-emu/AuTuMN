"""
Type definition for model parameters
"""
from pydantic import BaseModel, Extra, root_validator, validator
from pydantic.dataclasses import dataclass

from datetime import date
from typing import Any, Dict, List, Optional, Union

from autumn.models.covid_19.constants import GOOGLE_MOBILITY_LOCATIONS
from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.tools.inputs.social_mixing.constants import LOCATIONS

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
    A set of values with associated time points
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
    Compartment sojourn times, meaning the mean period of time spent in a compartment.
    """

    total_time: float
    proportion_early: Optional[float]

    check_total_positive = validator("total_time", allow_reuse=True)(
        get_check_non_neg("total_time")
    )
    check_prop_early = validator("proportion_early", allow_reuse=True)(
        get_check_prop("proportion_early")
    )


class Sojourns(BaseModel):
    """
    Parameters for determining how long a person stays in a given compartment.
    """

    active: CompartmentSojourn
    latent: CompartmentSojourn
    recovered: Optional[float]  # Doesn't have an early and late

    check_recovered_positive = validator("recovered", allow_reuse=True)(
        get_check_non_neg("recovered")
    )


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


class EmpiricMicrodistancingParams(BaseModel):

    max_effect: float
    times: List[float]
    values: List[float]

    check_max_effect = validator("max_effect", allow_reuse=True)(
        get_check_prop("max_effect")
    )

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        value, time_series = values.get("values"), values.get("times")
        msg = f"TimeSeries length mismatch, times length: {len(time_series)}, values length: {len(value)}"
        assert len(time_series) == len(value), msg
        return values


class TanhMicrodistancingParams(BaseModel):

    shape: float
    inflection_time: float
    lower_asymptote: float
    upper_asymptote: float

    check_lower_asymptote = validator("lower_asymptote", allow_reuse=True)(
        get_check_prop("lower_asymptote")
    )
    check_upper_asymptote = validator("upper_asymptote", allow_reuse=True)(
        get_check_prop("upper_asymptote")
    )

    @validator("shape", allow_reuse=True)
    def shape_is_positive(shape):
        assert (
            shape >= 0.0
        ), "Shape parameter for tanh-microdistancing function must be non-negative"

    @root_validator(pre=True, allow_reuse=True)
    def check_asymptotes(cls, values):
        lower, upper = values.get("lower_asymptote"), values.get("upper_asymptote")
        assert (
            lower <= upper
        ), f"Asymptotes specified upside-down, lower: {'lower'}, upper: {'upper'}"
        return values


class ConstantMicrodistancingParams(BaseModel):

    effect: float

    check_effect_domain = validator("effect", allow_reuse=True)(
        get_check_prop("effect")
    )


class MicroDistancingFunc(BaseModel):

    function_type: str
    parameters: Union[
        EmpiricMicrodistancingParams,
        TanhMicrodistancingParams,
        ConstantMicrodistancingParams,
    ]
    locations: List[str]

    @validator("locations", allow_reuse=True)
    def effect_domain(locations):
        assert all([loc in LOCATIONS for loc in locations])
        return locations


class Mobility(BaseModel):

    region: Optional[str]  # None/null means default to parent country
    mixing: Dict[str, MixingLocation]
    age_mixing: Optional[Dict[str, TimeSeries]]
    microdistancing: Dict[str, MicroDistancingFunc]
    smooth_google_data: bool
    square_mobility_effect: bool
    npi_effectiveness: Dict[str, float]
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

    check_none = validator("multiplier", allow_reuse=True)(get_check_prop("multiplier"))
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
    ]  # Dictionary to represent age groups, single float or None
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

    check_prop_immune = validator("prop_immune", allow_reuse=True)(
        get_check_prop("prop_immune")
    )
    check_high_immune = validator("prop_high_among_immune", allow_reuse=True)(
        get_check_prop("prop_high_among_immune")
    )


class TestingToDetection(BaseModel):
    """
    Empiric approach to building the case detection rate that is based on per capita testing rates.
    """

    assumed_tests_parameter: float
    assumed_cdr_parameter: float
    smoothing_period: int
    test_multiplier: Optional[TimeSeries]

    check_tests = validator("assumed_tests_parameter", allow_reuse=True)(
        get_check_non_neg("assumed_tests_parameter")
    )
    check_cdr = validator("assumed_cdr_parameter", allow_reuse=True)(
        get_check_prop("assumed_cdr_parameter")
    )

    @validator("smoothing_period", allow_reuse=True)
    def check_smoothing_period(val):
        assert 1 < val, f"Smoothing period must be greater than 1: {val}"
        return val


class CrossImmunity(BaseModel):

    early_reinfection: float
    late_reinfection: float

    check_early_reinfect = validator("early_reinfection", allow_reuse=True)(
        get_check_prop("early_reinfection")
    )
    check_late_reinfect = validator("late_reinfection", allow_reuse=True)(
        get_check_prop("late_reinfection")
    )


class VocSeed(BaseModel):

    start_time: float
    entry_rate: float
    seed_duration: float

    check_seed_time = validator("seed_duration", allow_reuse=True)(
        get_check_non_neg("seed_duration")
    )
    check_entry_rate = validator("entry_rate", allow_reuse=True)(
        get_check_non_neg("entry_rate")
    )


class VocComponent(BaseModel):
    """
    Parameters defining the emergence profile of the Variants of Concerns
    """

    starting_strain: bool
    seed_prop: float
    new_voc_seed: Optional[VocSeed]
    contact_rate_multiplier: float
    relative_latency: Optional[float]
    relative_active_period: Optional[float]
    immune_escape: float
    cross_protection: Dict[str, CrossImmunity]
    hosp_protection: Optional[float]
    death_protection: Optional[float]
    icu_multiplier: Optional[float]

    @root_validator(pre=True, allow_reuse=True)
    def check_starting_strain_multiplier(cls, values):
        if values["starting_strain"]:
            multiplier = values["contact_rate_multiplier"]
            msg = f"Starting or 'wild type' strain must have a contact rate multiplier of one: {multiplier}"
            assert multiplier == 1.0, msg
        return values

    @validator("icu_multiplier", pre=True, allow_reuse=True)
    def check_times(multiplier):
        if multiplier:
            assert 0.0 <= multiplier, "ICU multiplier negative"
        return multiplier

    check_immune_escape = validator("immune_escape", allow_reuse=True)(
        get_check_prop("immune_escape")
    )
    check_hosp_protection = validator("hosp_protection", allow_reuse=True)(
        get_check_prop("hosp_protection")
    )
    check_relative_latency = validator("relative_latency", allow_reuse=True)(
        get_check_non_neg("relative_latency")
    )
    check_relative_active_period = validator(
        "relative_active_period", allow_reuse=True
    )(get_check_non_neg("relative_active_period"))
    check_death_protection = validator("death_protection", allow_reuse=True)(
        get_check_prop("death_protection")
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

    notification: TimeDistribution
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


class ParamConfig:
    """
    Config for parameter models
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
    detect_prop: float  # Not optional, so as always to have a back-up value available if testing to detection not used
    testing_to_detection: Optional[TestingToDetection]
    asympt_infectiousness_effect: Optional[float]
    isolate_infectiousness_effect: Optional[float]

    time_from_onset_to_event: TimeToEvent
    hospital_stay: HospitalStay
    prop_icu_among_hospitalised: float

    age_stratification: AgeStratification
    immunity_stratification: ImmunityStratification
    voc_emergence: Optional[Dict[str, VocComponent]]

    # Random process
    activate_random_process: bool
    random_process: Optional[RandomProcessParams]

    @validator("age_groups", allow_reuse=True)
    def validate_age_groups(age_groups):
        assert all(
            [i_group % 5 == 0 for i_group in age_groups]
        ), "Not all age groups are multiples of 5"
        assert all(
            [0 <= i_group <= 75 for i_group in age_groups]
        ), "Age breakpoints must be from zero to 75"
        return age_groups
