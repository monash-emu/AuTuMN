
class Compartment:
    """
    A COVID-19 model compartment
    """
    SUSCEPTIBLE = "susceptible"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"


COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.INFECTIOUS,
    Compartment.RECOVERED,
]

AGEGROUP_STRATA = ['0', '15', '25', '50', '70']
