
class Compartment:
    SUSCEPTIBLE = "susceptible"
    LATENT = "latent"
    LATENT_LATE = "latent_late"
    INFECTIOUS = "infectious"
    INFECTIOUS_LATE = "infectious_late"
    RECOVERED = "recovered"


class FlowName:
    INFECTION = "infection"
    WITHIN_LATENT = "within_latent"
    PROGRESSION = "progression"
    WITHIN_INFECTIOUS = "within_infectious"
    RECOVERY = "recovery"


BASE_COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.INFECTIOUS,
    Compartment.RECOVERED,
]


WILD_TYPE = "wild_type"


class ClinicalStratum:
    ASYMPT = "asympt"
    SYMPT_NON_DETECT = "sympt_non_detect"
    DETECT = "detect"


CLINICAL_STRATA = [
    ClinicalStratum.ASYMPT,
    ClinicalStratum.SYMPT_NON_DETECT,
    ClinicalStratum.DETECT,
]


class ImmunityStratum:
    NONE = "none"
    HIGH = "high"
    LOW = "low"


IMMUNITY_STRATA = [ImmunityStratum.NONE, ImmunityStratum.HIGH, ImmunityStratum.LOW]
