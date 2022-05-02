"""
Type definition for model parameters
"""
from pydantic import BaseModel, Extra, root_validator, validator
from pydantic.dataclasses import dataclass

from datetime import date
from typing import Any, Dict, List, Optional, Union

from autumn.models.covid_19.constants import COVID_BASE_DATETIME, VACCINATION_STRATA, GOOGLE_MOBILITY_LOCATIONS, Strain
from autumn.settings.region import Region
from autumn.tools.inputs.social_mixing.constants import LOCATIONS

# Forbid additional arguments to prevent extraneous parameter specification
BaseModel.Config.extra = Extra.forbid

BASE_DATE = COVID_BASE_DATETIME.date()

class Time(BaseModel):
    """
    Parameters to define the model time period and evaluation steps.
    For the COVID-19 model, all times are assumed to be in days and reference time is 31st Dec 2019.
    The medium term plan is to replace this structure with standard Python date structures.
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
        assert len(time_series) == \
               len(value_series), \
        f"TimeSeries length mismatch, times length: {len(time_series)}, values length: {len(value_series)}"
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
    year: int


class Sojourn(BaseModel):
    """
    Parameters for determining how long a person stays in a given compartment.
    """

    class CalcPeriod(BaseModel):
        total_period: float
        proportions: Dict[str, float]

        @validator("proportions", allow_reuse=True)
        def check_props(props):
            prop_sum = sum(props.values())
            assert prop_sum == 1., f"Requested period proportions do not sum to one: {prop_sum}"
            return props

    # Mean time in days spent in each compartment
    compartment_periods: Dict[str, float]
    # Mean time spent in each compartment, defined via proportions
    compartment_periods_calculated: Dict[str, CalcPeriod]

    @validator("compartment_periods", allow_reuse=True)
    def check_positive(periods):
        assert all(val >= 0. for val in periods.values()), f"Sojourn times must be non-negative, times are: {periods}"
        return periods


class TanhScaleup(BaseModel):
    shape: float
    inflection_time: float
    start_asymptote: float
    end_asymptote: float

    @validator("shape", allow_reuse=True)
    def check_shape(val):
        msg = f"Shape parameter negative: {val}, change order of asymptotes if desired gradient is the reversed"
        assert 0. <= val, msg
        return val


class MixingLocation(BaseModel):
    # Whether to append or overwrite times / values
    append: bool
    # Times for dynamic mixing func.
    times: List[int]
    # Values for dynamic mixing func.
    values: List[Any]

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

    @validator("max_effect", allow_reuse=True)
    def effect_domain(effect):
        assert 0. <= effect <= 1.
        return effect

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        value, time_series = values.get("values"), values.get("times")
        assert len(time_series) == len(value), f"TimeSeries length mismatch, times length: {len(time_series)}, values length: {len(value)}"
        return values


class ConstantMicrodistancingParams(BaseModel):
    effect: float

    @validator("effect", allow_reuse=True)
    def effect_domain(effect):
        assert 0. <= effect <= 1., "Microdistancing effect not in domain [0, 1]"
        return effect


class MicroDistancingFunc(BaseModel):
    function_type: str
    parameters: Union[
        EmpiricMicrodistancingParams, TanhScaleup, ConstantMicrodistancingParams
    ]
    locations: List[str]

    @validator("locations", allow_reuse=True)
    def effect_domain(locations):
        assert all([loc in LOCATIONS for loc in locations])
        return locations


class Mobility(BaseModel):
    """Google mobility params"""

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
            assert abs(location_total - 1.) < 1e-6, msg
            msg = "Google mobility key not recognised"
            assert all([key in GOOGLE_MOBILITY_LOCATIONS for key in val[location].keys()]), msg
        return val


class AgeStratification(BaseModel):
    """
    Parameters used in age based stratification.
    """

    # Susceptibility by age
    susceptibility: Dict[str, float]


class StrataProps(BaseModel):
    props: List[float]
    multiplier: float

    @validator("props", allow_reuse=True)
    def check_props(val):
        msg = f"Not all of list of proportions is in [0, 1]: {val}"
        assert all([0. <= prop <= 1. for prop in val]), msg
        return val


class ClinicalProportions(BaseModel):
    hospital: StrataProps
    symptomatic: StrataProps


class ClinicalStratification(BaseModel):
    """
    Parameters used in clinical status based stratification.
    """

    props: ClinicalProportions
    icu_prop: float  # Proportion of those hospitalised that are admitted to ICU
    icu_mortality_prop: float  # Death proportion ceiling for ICU mortality
    late_infect_multiplier: Dict[str, float]
    non_sympt_infect_multiplier: float

    @validator("icu_prop", allow_reuse=True)
    def check_icu_prop(val):
        assert 0. <= val <= 1., f"Proportion of hospitalised patients admitted to ICU is not in [0, 1]: {val}"
        return val

    @validator("icu_mortality_prop", allow_reuse=True)
    def check_icu_ceiling(val):
        assert 0. <= val <= 1., f"Ceiling for proportion of ICU patients dying is not in [0, 1]: {val}"
        return val


class InfectionFatality(BaseModel):
    """
    Parameters relating to death from infection.
    """

    # Calibrated multiplier for props
    multiplier: float
    # Proportion of people dying / total infected by age
    props: List[float]

    @validator("multiplier", allow_reuse=True)
    def check_multiplier(val):
        assert 0. <= val, f"Multiplier applied to IFRs must be in range [0, 1]: {val}"
        return val


class TestingToDetection(BaseModel):
    """
    Empiric approach to building the case detection rate that is based on per capita testing rates.
    """

    assumed_tests_parameter: float
    assumed_cdr_parameter: float
    smoothing_period: int
    test_multiplier: Optional[TimeSeries]

    @validator("assumed_tests_parameter", allow_reuse=True)
    def check_assumed_tests_positive(val):
        assert 0. <= val, f"Assumed tests is negative: {val}"
        return val

    @validator("assumed_cdr_parameter", allow_reuse=True)
    def check_assumed_cdr_is_proportion(val):
        assert 0. <= val <= 1., f"Assumed CDR parameter is not in range [0, 1]: {val}"
        return val

    @validator("smoothing_period", allow_reuse=True)
    def check_smoothing_period(val):
        assert 1 < val, f"Smoothing period must be greater than 1: {val}"
        return val


class MetroClusterStratification(BaseModel):
    mobility: Mobility


class RegionalClusterStratification(BaseModel):
    mobility: Mobility


class VocComponent(BaseModel):
    """
    Parameters defining the emergence profile of the Variants of Concerns
    """

    start_time: Optional[float]
    entry_rate: Optional[float]
    seed_duration: Optional[float]
    contact_rate_multiplier: Optional[float]
    ifr_multiplier: Optional[float]
    hosp_multiplier: Optional[float]

    @root_validator(pre=True, allow_reuse=True)
    def check_times(cls, values):
        if "seed_duration" in values:
            assert 0. <= values["seed_duration"], "Seed duration negative"
        if "contact_rate_multiplier" in values:
            assert 0. <= values["contact_rate_multiplier"], "Contact rate multiplier negative"
        if "entry_rate" in values:
            assert 0. <= values["entry_rate"], "Entry rate negative"
        if "ifr_multiplier" in values:
            assert 0. <= values["ifr_multiplier"], "VoC effect on mortality negative"
        else:
            values["ifr_multiplier"] = 1.
        if "hosp_multiplier" in values:
            hosp_multiplier = values["hosp_multiplier"]
            assert 0. <= hosp_multiplier, f"VoC effect on hospitalisation negative"
        else:
            values["hosp_multiplier"] = 1.
        return values


class VaccCoveragePeriod(BaseModel):
    """
    Parameters to pass when desired behaviour is vaccinating a proportion of the population over a period of time.
    """

    coverage: Optional[float]
    start_time: float
    end_time: float

    @validator("coverage", allow_reuse=True)
    def check_coverage(val):
        if val:
            assert 0. <= val <= 1., f"Requested coverage for phase of vaccination program is not in [0, 1]: {val}"
        return val

    @root_validator(allow_reuse=True)
    def check_times(cls, values):
        msg = f"End time: {values['start_time']} before start time: {values['end_time']}"
        assert values["start_time"] <= values["end_time"], msg
        return values


class RollOutFunc(BaseModel):
    """
    Provides the parameters needed to construct a phase of vaccination roll-out.
    """

    age_min: Optional[float]
    age_max: Optional[float]
    supply_period_coverage: Optional[VaccCoveragePeriod]

    @root_validator(pre=True, allow_reuse=True)
    def check_suppy(cls, values):
        if "age_min" in values:
            assert 0. <= values["age_min"], f"Minimum age is negative: {values['age_min']}"
        if "age_max" in values:
            assert 0. <= values["age_max"], f"Minimum age is negative: {values['age_max']}"
        if "age_min" in values and "age_max" in values:
            msg = f"Maximum age: {values['age_max']} is less than minimum age: {values['age_max']}"
            assert values["age_min"] <= values["age_max"], msg
        return values


class VaccEffectiveness(BaseModel):
    ve_sympt_covid: float
    ve_prop_prevent_infection: Optional[float]
    ve_prop_prevent_infection_ratio: Optional[float]
    ve_infectiousness: Optional[float]
    ve_infectiousness_ratio: Optional[float]
    ve_hospitalisation: Optional[float]
    ve_death: Optional[float]
    doses: Optional[TimeSeries]
    coverage: Optional[TimeSeries]

    @validator("ve_sympt_covid", pre=True, allow_reuse=True)
    def check_ve_sympt_covid(val):
        assert 0. <= val <= 1., f"Overall efficacy should be in [0, 1]: {val}"
        return val

    @validator("ve_prop_prevent_infection", pre=True, allow_reuse=True)
    def check_ve_prop_prevent_infection(val):
        assert 0. <= val <= 1., f"Proportion of vaccine effect preventing infection should be in [0, 1]: {val}"
        return val

    @validator("ve_infectiousness", pre=True, allow_reuse=True)
    def check_ve_infectiousness(val):
        assert 0. <= val <= 1., f"Reduction in infectiousness should be in [0, 1]: {val}"
        return val

    @root_validator(pre=True, allow_reuse=True)
    def check_single_requests(cls, values):
        n_requests = sum(
            [int(bool(values[option])) for option in ["ve_infectiousness", "ve_infectiousness_ratio"]]
        )
        msg = f"Both ve_infectiousness and ve_infectiousness_ratio cannot be requested together"
        assert n_requests < 2, msg

        n_requests = sum(
            [int(bool(values[option])) for option in ["ve_prop_prevent_infection", "ve_prop_prevent_infection_ratio"]]
        )
        msg = f"Both ve_prop_prevent_infection and ve_prop_prevent_infection_ratio cannot be requested together"
        assert n_requests < 2, msg
        return values

    @validator("ve_hospitalisation", pre=True, allow_reuse=True)
    def check_ve_hospitalisation(val):
        if val:
            assert 0. <= val <= 1., f"Reduction in hospitalisation risk should be in [0, 1]: {val}"
        return val

    @validator("ve_death", pre=True, allow_reuse=True)
    def check_ve_death(val):
        if val:
            assert 0. <= val <= 1., f"Reduction in risk of death should be in [0, 1]: {val}"
        return val

    @root_validator(pre=True, allow_reuse=True)
    def check_effect_ratios(cls, values):
        overall_effect = values["ve_sympt_covid"]
        if values["ve_hospitalisation"]:
            hospital_effect = values["ve_hospitalisation"]
            msg = f"Symptomatic Covid effect: {overall_effect} exceeds hospitalisation effect: {hospital_effect}"
            assert hospital_effect >= overall_effect, msg
        if values["ve_death"]:
            death_effect = values["ve_death"]
            msg = f"Symptomatic Covid effect: {overall_effect} exceeds death effect: {death_effect}"
            assert death_effect >= overall_effect, msg
        return values


class Vaccination(BaseModel):

    # *** This parameter determines whether the model is stratified into three rather than two vaccination strata
    second_dose_delay: Optional[Union[float, TanhScaleup]]
    boost_delay: Optional[float]

    # *** This first parameter (vacc_full_effect_duration) determines whether waning immunity is applied
    vacc_full_effect_duration: Optional[Union[float, None]]
    vacc_part_effect_duration: Optional[float]

    one_dose: VaccEffectiveness
    fully_vaccinated: Optional[VaccEffectiveness]
    part_waned: Optional[VaccEffectiveness]
    fully_waned: Optional[VaccEffectiveness]
    boosted: Optional[VaccEffectiveness]
    lag: float
    program_start_time: Optional[float]

    standard_supply: bool

    roll_out_components: List[RollOutFunc]
    coverage_override: Optional[float]

    @root_validator(pre=True, allow_reuse=True)
    def check_vacc_range(cls, values):

        second_dose_delay = values["second_dose_delay"]
        msg = f"Days to second dose is less than one"
        if isinstance(second_dose_delay, (float, int)):
            assert second_dose_delay > 1., msg
        elif type(second_dose_delay) == TanhScaleup:
            assert second_dose_delay["start_asymptote"] > 1., msg
            assert second_dose_delay["end_asymptote"] > 1., msg
        return values

    @root_validator(pre=True, allow_reuse=True)
    def apply_ratio_adjustment(cls, values):

        strata_to_adjust = VACCINATION_STRATA[1: 2]
        for stratum in strata_to_adjust:
            for key in values["fully_vaccinated"]:
                ratio_key = f"{key}_ratio"
                if ratio_key in values[stratum] and values[stratum][ratio_key]:
                    values[stratum][key] = values["fully_vaccinated"][key] * values[stratum][ratio_key]
                    values[stratum][ratio_key] = None

        return values

    @validator("lag", allow_reuse=True)
    def check_lag(val):
        msg = f"Vaccination lag period is negative: {val}"
        assert val >= 0., msg
        return val


class VaccinationRisk(BaseModel):
    calculate: bool
    cumul_start_time: Optional[float]
    prop_astrazeneca: float
    prop_mrna: float
    tts_rate: Dict[str, float]
    tts_fatality_ratio: Dict[str, float]
    myocarditis_rate: Dict[str, float]
    risk_multiplier: float

    @root_validator(pre=True, allow_reuse=True)
    def check_vacc_risk_ranges(cls, values):
        msg = f"Proportion Astra-Zeneca not in range [0, 1]: {values['prop_astrazeneca']}"
        assert 0. <= values["prop_astrazeneca"] <= 1., msg
        msg = f"Proportion mRNA not in range [0, 1]: {values['prop_mrna']}"
        assert 0. <= values["prop_mrna"] <= 1., msg
        msg = f"At least one TTS rate is negative: {values['tts_rate']}"
        assert all([0. <= val for val in values["tts_rate"].values()]), msg
        msg = f"TTS fatality ratio is negative: {values['tts_fatality_ratio']}"
        assert all([0. <= val for val in values["tts_fatality_ratio"].values()]), msg
        msg = f"Myocarditis rate is negative: {values['myocarditis_rate']}"
        assert all([0. <= val for val in values["myocarditis_rate"].values()]), msg
        return values


class History(BaseModel):

    experienced: Optional[VaccEffectiveness]
    waned: Optional[VaccEffectiveness]

    natural_immunity_duration: Optional[float]

    @validator("natural_immunity_duration", allow_reuse=True)
    def check_immunity_duration(val):
        if type(val) == float:
            assert val > 0., f"Waning immunity duration request is not positive: {val}"
        return val


class ContactTracing(BaseModel):
    """
    Contact tracing effectiveness that scales with disease burden parameters.
    """
    floor: float
    assumed_trace_prop: float
    assumed_prev: float
    quarantine_infect_multiplier: float

    @validator("floor", allow_reuse=True)
    def check_floor(val):
        assert 0. <= val <= 1., f"Contact tracing floor must be in range [0, 1]: {val}"
        return val

    @validator("quarantine_infect_multiplier", allow_reuse=True)
    def check_multiplier(val):
        assert 0. <= val <= 1., f"Contact tracing infectiousness multiplier must be in range [0, 1]: {val}"
        return val

    @validator("assumed_prev", allow_reuse=True)
    def check_prevalence(val):
        assert 0. <= val <= 1., f"Contact tracing assumed prevalence must be in range [0, 1]: {val}"
        return val

    @validator("assumed_trace_prop", allow_reuse=True)
    def check_prevalence(val):
        assert 0. <= val <= 1., f"Contact tracing assumed tracing proportion must be in range [0, 1]: {val}"
        return val

    @root_validator(allow_reuse=True)
    def check_floor(cls, values):
        if "floor" in values:
            trace_prop = values["assumed_trace_prop"]
            floor_prop = values["floor"]
            msg = f"Contact tracing assumed_trace_prop must be >= floor: {trace_prop} < {floor_prop}"
            assert trace_prop >= floor_prop, msg
        return values


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
    # Values
    contact_rate: float
    seasonal_force: Optional[float]
    infect_death: float
    universal_death_rate: float
    infectious_seed: float
    voc_emergence: Optional[Dict[str, VocComponent]]
    vaccination: Optional[Vaccination]
    history: Optional[History]
    vaccination_risk: Optional[VaccinationRisk]
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
    ref_mixing_iso3: str
    infection_fatality: InfectionFatality
    age_stratification: AgeStratification
    clinical_stratification: ClinicalStratification
    testing_to_detection: Optional[TestingToDetection]
    contact_tracing: Optional[ContactTracing]
    hospital_reporting: float
    # Non_epidemiological parameters
    target_output_ratio: Optional[float]

    @validator("voc_emergence", allow_reuse=True)
    def check_voc_names(val):
        if val:
            msg = "Requested names for VoCs are not unique"
            assert len(set(val.keys())) == len(val.keys()), msg
            msg = f"Strain name {Strain.WILD_TYPE} reserved for the wild-type non-VoC strain"
            assert Strain.WILD_TYPE not in val.keys(), msg
        return val

    @validator("hospital_reporting", allow_reuse=True)
    def check_hospital_reporting(val):
        assert 0. <= val <= 1., f"Hospital reporting fraction must be in range [0, 1]: {val}"
        return val
