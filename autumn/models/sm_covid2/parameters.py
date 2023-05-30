"""
Type definition for model parameters
"""
from pydantic import BaseModel as _BaseModel, Extra, root_validator, validator
from pydantic.dataclasses import dataclass

from functools import partial

from datetime import date
from typing import Any, Dict, List, Optional, Union

from autumn.settings.constants import (
    COVID_BASE_DATETIME,
    GOOGLE_MOBILITY_LOCATIONS,
    COVID_BASE_AGEGROUPS,
)
from autumn.core.inputs.social_mixing.constants import LOCATIONS


from summer2.experimental.model_builder import (
    ParamStruct,
    parameter_class as pclass,
    parameter_array_class as parray,
)

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

    total_time: pclass(constraints.non_negative)
    proportion_early: Optional[pclass(constraints.non_negative)]


class Sojourns(BaseModel):
    """
    Parameters for determining how long a person stays in a given compartment.
    """

    active: pclass(constraints.non_negative)
    latent: pclass(constraints.non_negative)


class LatencyInfectiousness(BaseModel):
    """
    Parameters to define how many latent compartments are infectious and their relative
    infectiousness compared to the active disease compartments
    """

    n_infectious_comps: int
    rel_infectiousness: pclass(constraints.non_negative)


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
    microdistancing: Optional[
        dict
    ]  # this is not used for the sm_covid model. Still included to prevent crash in mixing matrix code
    apply_unesco_school_data: bool
    unesco_partial_opening_value: pclass()
    unesco_full_closure_value: Optional[float]

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
    multiplier: pclass(constraints.non_negative)


class AgeStratification(BaseModel):
    """
    Parameters used in age based stratification.
    """

    susceptibility: Optional[
        Union[Dict[int, float], float]
    ]  # Dictionary that represents each age group, single float or None
    prop_symptomatic: Optional[Union[Dict[int, float], float]]  # As for susceptibility
    prop_hospital: AgeSpecificProps
    ifr: AgeSpecificProps


class VaccineEffects(BaseModel):
    ve_infection: pclass()
    ve_hospitalisation: pclass()
    ve_death: pclass()


class VocSeed(BaseModel):

    time_from_gisaid_report: pclass()
    seed_duration: pclass(constraints.non_negative)


class VocComponent(BaseModel):
    """
    Parameters defining the emergence profile of the Variants of Concerns
    """

    starting_strain: bool
    seed_prop: float
    new_voc_seed: Optional[VocSeed]
    contact_rate_multiplier: pclass()
    incubation_overwrite_value: Optional[float]
    vacc_immune_escape: pclass(constraints.unit_interval)
    cross_protection: Dict[str, pclass()]
    hosp_risk_adjuster: Optional[pclass(constraints.non_negative)]
    death_risk_adjuster: Optional[pclass(constraints.non_negative)]
    icu_risk_adjuster: Optional[pclass(constraints.non_negative)]

    @root_validator(pre=True, allow_reuse=True)
    def check_starting_strain_multiplier(cls, values):
        if values["starting_strain"]:
            multiplier = values["contact_rate_multiplier"]
            msg = f"Starting or 'wild type' strain must have a contact rate multiplier of one: {multiplier}"
            assert multiplier == 1.0, msg
        return values


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

    hospitalisation: TimeDistribution
    icu_admission: TimeDistribution
    death: TimeDistribution


class HospitalStay(BaseModel):

    hospital_all: TimeDistribution
    icu: TimeDistribution


class RandomProcessParams(BaseModel):

    coefficients: Optional[List[float]]
    noise_sd: Optional[float]
    delta_values: Optional[parray()]
    order: int
    time: Time
    affected_locations: List[str]


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
    age_groups: List[int]
    time: Time
    infectious_seed_time: pclass()
    seed_duration: float

    serodata_age: dict

    # Values
    contact_rate: pclass()
    sojourns: Sojourns
    is_dynamic_mixing_matrix: bool
    mobility: Mobility
    school_multiplier: pclass()
    hh_contact_increase: pclass()

    compartment_replicates: Dict[str, int]
    latency_infectiousness: LatencyInfectiousness

    time_from_onset_to_event: TimeToEvent
    hospital_stay: HospitalStay
    prop_icu_among_hospitalised: float

    age_stratification: AgeStratification
    vaccine_effects: VaccineEffects

    voc_emergence: Optional[Dict[str, VocComponent]]

    # Random process
    activate_random_process: bool
    random_process: Optional[RandomProcessParams]

    # Output-related
    requested_cumulative_outputs: List[str]
    cumulative_start_time: Optional[float]
    request_incidence_by_age: bool
    request_immune_prop_by_age: bool

    @validator("age_groups", allow_reuse=True)
    def validate_age_groups(age_groups):
        msg = "Not all requested age groups in the available age groups of 5-year increments from zero to 75"
        int_age_groups = [int(i_group) for i_group in COVID_BASE_AGEGROUPS]
        assert all([i_group in int_age_groups for i_group in age_groups]), msg
        return age_groups

    @validator("compartment_replicates", allow_reuse=True)
    def validate_comp_replicates(compartment_replicates):
        replicated_comps = list(compartment_replicates.keys())
        msg = "Replicated compartments must be latent and infectious"
        assert replicated_comps == ["latent", "infectious"], msg

        n_replicates = list(compartment_replicates.values())
        msg = "Number of requested replicates should be positive"
        assert all([n > 0 for n in n_replicates]), msg

        return compartment_replicates
