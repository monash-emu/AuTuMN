
class Compartment:
    SUSCEPTIBLE = "susceptible"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"


class FlowName:
    INFECTION = "infection"
    RECOVERY = "recovery"
    INFECTION_DEATH = "infection_death"


COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.INFECTIOUS,
    Compartment.RECOVERED,
]

AGEGROUP_STRATA = ['0', '15', '25', '50', '70']
