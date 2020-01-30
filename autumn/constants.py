"""
Constants used in building the AuTuMN / SUMMER models.
"""


class Compartment:
    """
    A model compartment.
    """

    SUSCEPTIBLE = "susceptible"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"


class Flow:
    """
    A type of flow between model compartments
    """

    INFECTION_FREQUENCY = "infection_frequency"
    INFECTION_DENSITY = "infection_density"
    DEATH = "compartment_death"


class BirthApproach:
    """
    Options for birth rate settings in model
    """

    NO_BIRTH = "no_birth"
    ADD_CRUDE = "add_crude_birth_rate"
    REPLACE_DEATHS = "replace_deaths"


class IntegrationType:
    """
    Options for ODE solver used by model
    """

    ODE_INT = "odeint"
    SOLVE_IVP = "solve_ivp"
