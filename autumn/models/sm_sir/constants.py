
class Compartment:
    SUSCEPTIBLE = "susceptible"
    EXPOSED = "exposed"
    EXPOSED_LATE = "exposed_late"
    INFECTIOUS = "infectious"
    INFECTIOUS_LATE = "infectious_late"
    RECOVERED = "recovered"


class FlowName:
    INFECTION = "infection"
    PROGRESSION = "progression"
    WITHIN_EXPOSED = "within_exposed"
    WITHIN_INFECTIOUS = "within_infectious"
    RECOVERY = "recovery"


BASE_COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.INFECTIOUS,
    Compartment.RECOVERED,
]

AGEGROUP_STRATA = ['0', '15', '25', '50', '70']


class ImmunityStratum:
    NONE = 'none'
    HIGH = 'high'
    LOW = 'low'


IMMUNITY_STRATA = [ImmunityStratum.NONE, ImmunityStratum.HIGH, ImmunityStratum.LOW]
