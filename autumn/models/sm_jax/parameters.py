"""
Type definition for model parameters
"""
from functools import partial

from datetime import date
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel as _BaseModel, Extra, root_validator, validator
from pydantic.dataclasses import dataclass


from autumn.settings.constants import (
    COVID_BASE_DATETIME,
    GOOGLE_MOBILITY_LOCATIONS,
    COVID_BASE_AGEGROUPS,
)
from autumn.core.inputs.social_mixing.constants import LOCATIONS

from summer2.experimental.model_builder import ParamStruct, parameter_class as pclass

from numpyro.distributions import constraints
from numbers import Real

from math import inf

# Mysterious missing constraint in numpyro...
constraints.non_negative = constraints.interval(0.0, inf)

BASE_DATE = COVID_BASE_DATETIME.date()

# Forbid additional arguments to prevent extraneous parameter specification
_BaseModel.Config.extra = Extra.forbid

# ModelBuilder requires all parameters to be embedded in ParamStruct objects
class BaseModel(_BaseModel, ParamStruct):
    pass


"""
Commonly used checking processes
"""


def validate_expected(field: str, expected: str):
    """Returns a validator that asserts that the member field {field}
    has value {expected}

    Args:
        field: Member field to validate
        expected: Expected value
    """

    def check_field_value(value_to_check):
        assert value_to_check == expected, f"Invalid {field}: {value_to_check}"
        return value_to_check

    return validator(field)(check_field_value)


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

    total_time: pclass(constraints.non_negative, "total_time", "Total mean time in compartment")
    proportion_early: Optional[
        pclass(constraints.unit_interval, "proportion_early", "Proportion early")
    ]


class Sojourns(BaseModel):
    """
    Parameters for determining how long a person stays in a given compartment.
    """

    active: CompartmentSojourn
    latent: CompartmentSojourn
    recovered: Optional[
        pclass(constraints.non_negative)
    ]  # If there is a sojourn time for recovered, then there will be two compartments


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

    check_max_effect = validator("max_effect", allow_reuse=True)(get_check_prop("max_effect"))

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        value_series, time_series = values.get("values"), values.get("times")
        msg = f"TimeSeries length mismatch, times length: {len(time_series)}, values length: {len(value_series)}"
        assert len(time_series) == len(value_series), msg
        return values


class TanhMicrodistancingParams(BaseModel):

    shape: float
    inflection_time: float
    start_asymptote: float
    end_asymptote: float

    check_lower_asymptote = validator("start_asymptote", allow_reuse=True)(
        get_check_prop("start_asymptote")
    )
    check_upper_asymptote = validator("end_asymptote", allow_reuse=True)(
        get_check_prop("end_asymptote")
    )

    @validator("shape", allow_reuse=True)
    def shape_is_positive(shape):
        msg = "Shape parameter for tanh-microdistancing function must be non-negative"
        assert shape >= 0.0, msg
        return shape


class ConstantMicrodistancingParams(BaseModel):

    effect: float

    check_effect_domain = validator("effect", allow_reuse=True)(get_check_prop("effect"))


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
    google_mobility_locations: Dict[str, Dict[str, float]]

    @validator("google_mobility_locations", allow_reuse=True)
    def check_location_weights(val):
        for location in val:
            location_total = sum(val[location].values())
            msg = f"Mobility weights don't sum to one: {location_total}"
            assert abs(location_total - 1.0) < 1e-6, msg
            msg = "Google mobility key not recognised"
            assert all([key in GOOGLE_MOBILITY_LOCATIONS for key in val[location].keys()]), msg
        return val


class AgeSpecificProps(BaseModel):

    values: Dict[int, float]
    source_immunity_distribution: Dict[str, float]
    source_immunity_protection: Dict[str, float]
    multiplier: pclass(constraints.non_negative)

    @validator("source_immunity_distribution", allow_reuse=True)
    def check_source_dist(vals):
        sum_of_values = sum(vals.values())
        msg = f"Proportions by immunity status in source for parameters does not sum to one: {sum_of_values}"
        assert sum_of_values == 1.0, msg
        return vals

    @validator("source_immunity_protection", allow_reuse=True)
    def check_source_dist(protection_params):
        msg = "Source protection estimates not proportions"
        assert all([0.0 <= val <= 1.0 for val in protection_params.values()]) == 1.0, msg
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

    none: pclass(constraints.unit_interval)
    low: pclass(constraints.unit_interval)
    high: pclass(constraints.unit_interval)

    @root_validator(pre=True, allow_reuse=True)
    def check_progression(cls, values):
        msg = "Immunity stratification effects are not increasing"
        assert values["none"] <= values["low"] <= values["high"], msg
        return values


class ImmunityStratification(BaseModel):

    prop_immune: pclass(constraints.unit_interval)
    prop_high_among_immune: pclass(constraints.unit_interval)
    infection_risk_reduction: ImmunityRiskReduction


class TestingToDetection(BaseModel):
    """
    Empiric approach to building the case detection rate that is based on per capita testing rates.
    """

    assumed_tests_parameter: pclass(constraints.non_negative)
    assumed_cdr_parameter: pclass(constraints.unit_interval)
    smoothing_period: int
    floor_value: pclass(constraints.non_negative)

    @validator("smoothing_period", allow_reuse=True)
    def check_smoothing_period(val):
        assert 1 <= val, f"Smoothing period must be greater than or equal to one: {val}"
        return val

    @root_validator(pre=True, allow_reuse=True)
    def check_floor_request(cls, values):
        floor_value, assumed_cdr = values["floor_value"], values["assumed_cdr_parameter"]
        msg = f"Requested value for the CDR floor does not fall between zero and the assumed CDR parameter of {assumed_cdr}, value is: {floor_value}"
        assert 0.0 <= floor_value <= assumed_cdr, msg
        return values


class CrossImmunity(BaseModel):

    early_reinfection: pclass(constraints.unit_interval)
    late_reinfection: pclass(constraints.unit_interval)


class VocSeed(BaseModel):

    start_time: pclass()
    entry_rate: pclass(constraints.non_negative)
    seed_duration: pclass(constraints.non_negative)


class VocComponent(BaseModel):
    """
    Parameters defining the emergence profile of the Variants of Concerns
    """

    starting_strain: bool
    seed_prop: pclass()
    new_voc_seed: Optional[VocSeed]
    contact_rate_multiplier: pclass(constraints.non_negative)
    relative_latency: Optional[pclass(constraints.non_negative)]
    relative_active_period: Optional[pclass(constraints.non_negative)]
    immune_escape: pclass()
    cross_protection: Dict[str, CrossImmunity]
    hosp_protection: Optional[pclass(constraints.unit_interval)]
    death_protection: Optional[pclass(constraints.unit_interval)]
    icu_multiplier: Optional[pclass(constraints.non_negative)]


validate_dist = partial(validate_expected, "distribution")


@dataclass
class GammaDistribution(ParamStruct):
    distribution: str
    shape: pclass(constraints.positive, desc="shape")
    mean: pclass(desc="mean")

    _check_dist = validate_dist("gamma")

    def __repr__(self):
        return f"Gamma: {self.shape},{self.mean}"


TimeDistribution = GammaDistribution


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
    delta_values: Optional[List[float]]
    order: int
    time: Time


class ParamConfig:
    """
    Config for parameter models.
    """

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable


@dataclass(config=ParamConfig)
class Parameters(ParamStruct):
    # Metadata
    description: Optional[str]
    country: Country
    population: Population
    ref_mixing_iso3: str
    age_groups: List[int]
    time: Time
    # Values
    contact_rate: pclass(constraints.positive)
    infectious_seed: pclass(constraints.non_negative)
    sojourns: Sojourns
    is_dynamic_mixing_matrix: bool
    mobility: Mobility

    is_undetected: bool  # If True, then case detection rate is effectively 1.0
    detect_prop: pclass(
        constraints.unit_interval
    )  # Not optional, so as always to have a back-up value available if testing to detection not used
    testing_to_detection: Optional[TestingToDetection]
    asympt_infectiousness_effect: Optional[pclass()]
    isolate_infectiousness_effect: Optional[pclass()]

    time_from_onset_to_event: TimeToEvent
    hospital_stay: HospitalStay
    prop_icu_among_hospitalised: pclass(constraints.unit_interval)

    age_stratification: AgeStratification
    immunity_stratification: ImmunityStratification
    voc_emergence: Optional[Dict[str, VocComponent]]

    # Random process
    activate_random_process: bool
    random_process: Optional[RandomProcessParams]

    # Vaccination/immunity-related
    booster_effect_duration: float
    additional_immunity: Optional[TimeSeries]
    future_monthly_booster_rate: Optional[float]
    future_booster_age_allocation: Optional[
        Union[
            Dict[
                int, float
            ],  # to specify allocation proportions by age group (e.g. {70: .8, 50: .2})
            List[int],  # to specify a prioritisation order (e.g. [70, 50, 25, 15])
        ]
    ]

    # Output-related
    requested_cumulative_outputs: List[str]
    cumulative_start_time: Optional[float]
    request_incidence_by_age: bool
    request_immune_prop_by_age: bool
    request_hospital_admissions_by_age: bool
    request_hospital_occupancy_by_age: bool
    request_icu_admissions_by_age: bool
    request_icu_occupancy_by_age: bool
    request_infection_deaths_by_age: bool

    @validator("age_groups", allow_reuse=True)
    def validate_age_groups(age_groups):
        msg = "Not all requested age groups in the available age groups of 5-year increments from zero to 75"
        int_age_groups = [int(i_group) for i_group in COVID_BASE_AGEGROUPS]
        assert all([i_group in int_age_groups for i_group in age_groups]), msg
        return age_groups
