
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
