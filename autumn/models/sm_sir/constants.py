
class Compartment:
    SUSCEPTIBLE = "susceptible"
    EXPOSED = "exposed"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"


class FlowName:
    INFECTION = "infection"
    PROGRESSION = "progression"
    RECOVERY = "recovery"


COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.EXPOSED,
    Compartment.INFECTIOUS,
    Compartment.RECOVERED,
]

AGEGROUP_STRATA = ['0', '15', '25', '50', '70']


class ImmunityStratum:
    NONE = 'none'
    HIGH = 'high'
    LOW = 'low'


IMMUNITY_STRATA = [ImmunityStratum.NONE, ImmunityStratum.HIGH, ImmunityStratum.LOW]
