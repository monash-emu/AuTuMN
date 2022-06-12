
# All available compartments
class Compartment:
    SUSCEPTIBLE = "susceptible"
    LATENT = "latent"
    LATENT_LATE = "latent_late"
    INFECTIOUS = "infectious"
    INFECTIOUS_LATE = "infectious_late"
    RECOVERED = "recovered"
    WANED = "waned"


# All available flows
class FlowName:
    INFECTION = "infection"
    WITHIN_LATENT = "within_latent"
    PROGRESSION = "progression"
    WITHIN_INFECTIOUS = "within_infectious"
    RECOVERY = "recovery"
    WANING = "waning"
    EARLY_REINFECTION = "early_reinfection"
    LATE_REINFECTION = "late_reinfection"


# Routinely implemented compartments
BASE_COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.LATENT,
    Compartment.INFECTIOUS,
    Compartment.RECOVERED,
]


WILD_TYPE = "wild_type"


# Available strata for the clinical stratification
class ClinicalStratum:
    ASYMPT = "asympt"
    SYMPT_NON_DETECT = "sympt_non_detect"
    DETECT = "detect"


CLINICAL_STRATA = [
    ClinicalStratum.ASYMPT,
    ClinicalStratum.SYMPT_NON_DETECT,
    ClinicalStratum.DETECT,
]


# Available strata for the immunity stratification
class ImmunityStratum:
    NONE = "none"
    LOW = "low"
    HIGH = "high"


IMMUNITY_STRATA = [
    ImmunityStratum.NONE,
    ImmunityStratum.LOW,
    ImmunityStratum.HIGH
]

LOCATIONS = ["home", "other_locations", "school", "work"]

PARAMETER_NAMES = {
    "age_stratification.cfr.source_immunity_distribution.high": 
        "proportion of population received booster in evidence source setting",
    "age_stratification.cfr.source_immunity_distribution.low":
        "proportion of population received two doses but no booster",
    "age_stratification.cfr.source_immunity_distribution.none":
        "proportion of population not received primary course",
    "age_stratification.cfr.source_immunity_protection.high":
        "protection against severe disease from booster assumed for evidence source setting",
    "age_stratification.cfr.source_immunity_protection.low":
        "protection against severe disease from primary course for evidence source setting",
    "asympt_infectiousness_effect":
        "relative infectiousness of asymptomatic persons",
    "booster_effect_duration":
        "duration of booster effect",
    "contact_rate":
        "probability of transmission per contact",
    "hospital_stay.hospital_all.distribution":
        "distribution type for hospital stay",
    "hospital_stay.hospital_all.parameters.mean":
        "mean hospital stay",
    "hospital_stay.hospital_all.parameters.shape":
        "shape parameter for hospital stay",
    "hospital_stay.icu.distribution":
        "distribution type for ICU stay",
    "hospital_stay.icu.parameters.mean":
        "mean ICU stay",
    "hospital_stay.icu.parameters.shape":
        "shape parameter for ICU stay",
    "immunity_stratification.infection_risk_reduction.high":
        "reduction in transmission risk for boosted",
    "immunity_stratification.infection_risk_reduction.low":
        "reduction in transmission risk for primary course",
    "infectious_seed":
        "starting infectious seed",
    "isolate_infectiousness_effect":
        "relative infectiousness of isolated cases",
    "prop_icu_among_hospitalised":
        "proportion of hospitalised persons admitted to ICU",
    "ref_mixing_iso3":
        "ISO3 code for source country for mixing matrix",
    "sojourns.active.proportion_early":
        "proportion of active period before isolation",
    "sojourns.latent.proportion_early":
        "proportion of latent period in first compartment",
    "testing_to_detection.assumed_tests_parameter":
        "index testing rate (\(tests(t)\))",
    "testing_to_detection.assumed_cdr_parameter":
        "CDR reached at index testing rate (\(CDR(t)\))",
}

PARAMETER_EXPLANATIONS = {
}
