"""
Type definition for model parameters
"""
from datetime import date
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Extra, root_validator, validator
from pydantic.dataclasses import dataclass

from autumn.models.covid_19.constants import BASE_DATE

# Forbid additional arguments to prevent extraneous parameter specification
BaseModel.Config.extra = Extra.forbid


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


class MixingMatrices(BaseModel):
    type: Optional[str]  # None defaults to Prem matrices, otherwise 'prem' or 'extrapolated' - see build_model
    source_iso3: Optional[str]
    age_adjust: bool  # Only relevant if 'extrapolated' selected


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

    # Calibrated multiplier for props
    multiplier: float
    # Alternative approach to adjusting the IFR during calibration - over-write the oldest age bracket
    top_bracket_overwrite: Optional[float]
    # Proportion of people dying / total infected by age
    props: List[float]


class TestingToDetection(BaseModel):
    """
    More empiric approach based on per capita testing rates
    An alternative to CaseDetection.
    """

    assumed_tests_parameter: float
    assumed_cdr_parameter: float
    smoothing_period: int
    test_multiplier: Optional[TimeSeries]


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
    contact_rate_multiplier_south_metro: float
    contact_rate_multiplier_barwon_south_west: float
    contact_rate_multiplier_regional: float
    metro: MetroClusterStratification
    regional: RegionalClusterStratification


class VocComponent(BaseModel):
    """
        Parameters defining the emergence profile of the Variants of Concerns
    """
    start_time: Optional[float]
    entry_rate: Optional[float]
    seed_duration: Optional[float]
    contact_rate_multiplier: Optional[float]


class VaccCoveragePeriod(BaseModel):
    """
    Parameters to pass when desired behaviour is vaccinating a proportion of the population over a period of time
    """

    coverage: Optional[float]
    start_time: float
    end_time: float


class RollOutFunc(BaseModel):
    age_min: Optional[float]
    age_max: Optional[float]
    supply_period_coverage: Optional[VaccCoveragePeriod]
    supply_timeseries: Optional[TimeSeries]

    @root_validator(pre=True, allow_reuse=True)
    def check_suppy(cls, values):
        p, ts = values.get("supply_period_coverage"), values.get("supply_timeseries")
        has_supply = bool(p) != bool(ts)
        assert has_supply, "Roll out function must have a period or timeseries for supply."
        return values


class VaccEffectiveness(BaseModel):
    overall_efficacy: float
    vacc_prop_prevent_infection: float
    vacc_reduce_infectiousness: float

    @validator("overall_efficacy", pre=True, allow_reuse=True)
    def check_overall_efficacy(val):
        assert 0 <= val <= 1, "Overall efficacy should be in [0, 1]"
        return val

    @validator("vacc_prop_prevent_infection", pre=True, allow_reuse=True)
    def check_vacc_prop_prevent_infection(val):
        assert 0 <= val <= 1, "Proportion of vaccine effect attributable to preventing infection should be in [0, 1]"
        return val

    @validator("vacc_reduce_infectiousness", pre=True, allow_reuse=True)
    def check_overall_efficacy(val):
        assert 0 <= val <= 1, "Reduction in infectousness should be in [0, 1]"
        return val


class Vaccination(BaseModel):
    second_dose_delay: float
    one_dose: Optional[VaccEffectiveness]
    fully_vaccinated: VaccEffectiveness

    roll_out_components: List[RollOutFunc]
    coverage_override: Optional[float]


class VaccinationRisk(BaseModel):
    prop_astrazeneca: float
    prop_mrna: float

    tts_rate: Dict[str, float]
    tts_fatality_ratio: Dict[str, float]

    myocarditis_rate: Dict[str, float]


class ContactTracing(BaseModel):
    """

    """
    assumed_trace_prop: float
    assumed_prev: float
    quarantine_infect_multiplier: float


class AgeSpecificRiskMultiplier(BaseModel):
    age_categories: List[str]
    adjustment_start_time: Optional[int]
    adjustment_end_time: Optional[int]
    contact_rate_multiplier: float


class ParamConfig:
    """Config for parameter models"""

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable


@dataclass(config=ParamConfig)
class Parameters:
    # Metadata
    description: Optional[str]
    # Values
    contact_rate: float
    infect_death: float
    universal_death_rate: float
    infectious_seed: float
    voc_emergence: Optional[Dict[str, VocComponent]]
    age_specific_risk_multiplier: Optional[AgeSpecificRiskMultiplier]
    stratify_by_infection_history: bool
    waning_immunity_duration: Optional[float]
    vaccination: Optional[Vaccination]
    vaccination_risk: Optional[VaccinationRisk]
    rel_prop_symptomatic_experienced: Optional[float]
    haario_scaling_factor: float
    metropolis_init_rel_step_size: float
    n_steps_fixed_proposal: int
    cumul_incidence_start_time: Optional[float]
    # Modular parameters
    time: Time
    country: Country
    population: Population
    sojourn: Sojourn
    mobility: Mobility
    mixing_matrices: Optional[MixingMatrices]
    infection_fatality: InfectionFatality
    age_stratification: AgeStratification
    clinical_stratification: ClinicalStratification
    testing_to_detection: Optional[TestingToDetection]
    contact_tracing: Optional[ContactTracing]
    victorian_clusters: Optional[VictorianClusterStratification]
    # Non-epidemiological parameters
    target_output_ratio: Optional[float]
