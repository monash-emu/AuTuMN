"""
Type definition for model parameters
"""
from datetime import date
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, root_validator, validator
from pydantic.dataclasses import dataclass

from apps.covid_19.constants import BASE_DATE


class Time(BaseModel):
    """
    Model time period
    Running time-related - for COVID-19 model
    All times are assumed to be in days and reference time is 31st Dec 2019
    """

    start: float
    end: float
    step: float


class TimeSeries(BaseModel):
    """A set of values with associated time points"""

    times: List[float]
    values: List[float]

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        vs, ts = values.get("values"), values.get("times")
        assert len(ts) == len(vs), f"TimeSeries length mismatch."
        return values

    @validator("times", pre=True, allow_reuse=True)
    def parse_dates_to_days(dates):
        return [(d - BASE_DATE).days if type(d) is date else d for d in dates]


class Country(BaseModel):
    """The country that the model is based in."""

    iso3: str


class Population(BaseModel):
    """Model population parameters"""

    region: Optional[str]  # None/null means default to parent country
    year: int


class Sojourn(BaseModel):
    """Parameters for determining how long a person stays in a given compartment."""

    class CalcPeriod(BaseModel):
        total_period: float
        proportions: Dict[str, float]

    # Mean time in days spent in each compartment
    compartment_periods: Dict[str, float]
    # Mean time spent in each compartment, defined via proportions
    compartment_periods_calculated: Dict[str, CalcPeriod]


class MixingLocation(BaseModel):
    # Whether to append or overwrite times / values
    append: bool
    # Times for dynamic mixing func.
    times: List[int]
    # Values for dynamic mixing func.
    values: List[Any]

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        vs, ts = values.get("values"), values.get("times")
        assert len(ts) == len(vs), f"Mixing series length mismatch."
        return values

    @validator("times", pre=True, allow_reuse=True)
    def parse_dates_to_days(dates):
        return [(d - BASE_DATE).days if type(d) is date else d for d in dates]


class EmpiricMicrodistancingParams(BaseModel):
    max_effect: float
    times: List[float]
    values: List[float]


class TanhMicrodistancingParams(BaseModel):
    shape: float
    inflection_time: float
    lower_asymptote: float
    upper_asymptote: float


class ConstantMicrodistancingParams(BaseModel):
    effect: float


class MicroDistancingFunc(BaseModel):
    function_type: str
    parameters: Union[
        EmpiricMicrodistancingParams, TanhMicrodistancingParams, ConstantMicrodistancingParams
    ]
    locations: List[str]


class Mobility(BaseModel):
    """Google mobility params"""

    region: Optional[str]  # None/null means default to parent country
    mixing: Dict[str, MixingLocation]
    age_mixing: Optional[Dict[str, TimeSeries]]
    microdistancing: Dict[str, MicroDistancingFunc]
    smooth_google_data: bool
    square_mobility_effect: bool
    npi_effectiveness: Dict[str, float]
    google_mobility_locations: Dict[str, List[str]]


class AgeStratification(BaseModel):
    """Parameters used in age based stratification"""

    # Susceptibility by age
    susceptibility: Dict[str, float]


class StrataProps(BaseModel):
    props: List[float]
    multiplier: float


class ClinicalProportions(BaseModel):
    hospital: StrataProps
    symptomatic: StrataProps


class ClinicalStratification(BaseModel):
    """Parameters used in clinical status based stratification"""

    props: ClinicalProportions
    icu_prop: float  # Proportion of those hospitalised that are admitted to ICU
    icu_mortality_prop: float  # Death proportion ceiling for ICU mortality
    late_infect_multiplier: Dict[str, float]
    non_sympt_infect_multiplier: float


class InfectionFatality(BaseModel):
    """Parameters relating to death from infection"""

    # Calibrated multiplier for props.
    multiplier: float
    # Proportion of people dying / total infected by age.
    props: List[float]


class CaseDetection(BaseModel):
    """Time variant detection of cases"""

    shape: float  # The shape parameter to the tanh-based curve
    inflection_time: float  # Point at which curve inflects
    lower_asymptote: float  # Starting value - lower asymptote for increasing function
    upper_asymptote: float  # End value - upper asymptote for increasing function


class TestingToDetection(BaseModel):
    """
    More empiric approach based on per capita testing rates
    An alternative to CaseDetection.
    """

    assumed_tests_parameter: float
    assumed_cdr_parameter: float
    smoothing_period: int


class SusceptibilityHeterogeneity(BaseModel):
    """Specifies heterogeneity in susceptibility"""

    bins: int
    tail_cut: float
    coeff_var: float


class MetroClusterStratification(BaseModel):
    mobility: Mobility


class RegionalClusterStratification(BaseModel):
    mobility: Mobility


class VictorianClusterStratification(BaseModel):
    intercluster_mixing: float
    contact_rate_multiplier_north_metro: float
    contact_rate_multiplier_west_metro: float
    contact_rate_multiplier_south_metro: float
    contact_rate_multiplier_south_east_metro: float
    contact_rate_multiplier_barwon_south_west: float
    contact_rate_multiplier_regional: float
    metro: MetroClusterStratification
    regional: RegionalClusterStratification


class VaccCoveragePeriod(BaseModel):
    """
    Parameters to pass when desired behaviour is vaccinating a proportion of the population over a period of time
    """

    coverage: float
    start_time: float
    end_time: float


class VocEmmergence(BaseModel):
    """
    Parameters defining the emergence profile of Variants of Concerns
    """

    final_proportion: float
    start_time: float
    end_time: float
    contact_rate_multiplier: float


class Vaccination(BaseModel):
    infection_efficacy: float
    severity_efficacy: float
    roll_out_function: VaccCoveragePeriod


class ParamConfig:
    """Config for parameter models"""

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable


@dataclass(config=ParamConfig)
class Parameters:
    description: str  # Scenario description, used by PowerBI.
    contact_rate: float
    infect_death: float
    universal_death_rate: float
    infectious_seed: float
    seasonal_force: Optional[float]  # Seasonal forcing factor
    voc_emmergence: Optional[VocEmmergence]
    elderly_mixing_reduction: Optional[dict]
    waning_immunity_duration: Optional[float]
    stratify_by_immunity: bool
    vaccination: Optional[Vaccination]
    stratify_by_infection_history: bool
    rel_prop_symptomatic_experienced: Optional[float]
    haario_scaling_factor: float
    metropolis_init_rel_step_size: float
    n_steps_fixed_proposal: int
    metropolis_initialisation_type: str
    # Modular parameters.
    time: Time
    country: Country
    population: Population
    sojourn: Sojourn
    mobility: Mobility
    infection_fatality: InfectionFatality
    age_stratification: AgeStratification
    clinical_stratification: ClinicalStratification
    case_detection: CaseDetection
    testing_to_detection: Optional[TestingToDetection]
    victorian_clusters: Optional[VictorianClusterStratification]
    # Dummy parameters - not used
    notifications_dispersion_param: float
    icu_occupancy_dispersion_param: float
    proportion_seropositive_dispersion_param: float
    hospital_occupancy_dispersion_param: float
    new_hospital_admissions_dispersion_param: float
    new_icu_admissions_dispersion_param: float
    infection_deaths_dispersion_param: float
    accum_deaths_dispersion_param: float
    target_output_ratio: Optional[float]
