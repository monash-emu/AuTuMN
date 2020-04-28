"""
Constants used in building SUMMER models.
"""


class Compartment:
    """
    A model compartment.
    """

    SUSCEPTIBLE = "susceptible"
    EARLY_LATENT = "early_latent"
    LATE_LATENT = "late_latent"
    EXPOSED = "exposed"
    INFECTIOUS = "infectious"
    EARLY_INFECTIOUS = "infectious"
    RECOVERED = "recovered"
    LATE_INFECTIOUS = "late"
    PRESYMPTOMATIC = "presympt"
    ON_TREATMENT = "on_treatment"
    LTBI_TREATED = "ltbi_treated"


class Flow:
    """
    A type of flow between model compartments
    """

    CUSTOM = "customised_flows"
    STANDARD = "standard_flows"
    INFECTION_FREQUENCY = "infection_frequency"
    INFECTION_DENSITY = "infection_density"
    COMPARTMENT_DEATH = "compartment_death"
    STRATA_CHANGE = "strata_change"


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
    EULER = "euler"
    RUNGE_KUTTA = "rk4"


class Stratification:
    """
    Attribute used to stratify the population within compartments. 
    """

    AGE = "age"
    STRAIN = "strain"
    LOCATION = "location"
