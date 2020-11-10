"""
Constants used in building SUMMER models.
"""


class FlowAdjustment:
    """
    A type of adjustment to a stratified flow.
    """

    OVERWRITE = "OVERWRITE"  # Overwrite with a value
    MULTIPLY = "MULTIPLY"  # Multiply with a value
    COMPOSE = "COMPOSE"  # Compose with a function


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
    ON_TREATMENT = "on_treatment"
    LTBI_TREATED = "ltbi_treated"
    DETECTED = "detected"


class Flow:
    """
    A type of flow between model compartments
    """

    # Transition flows: where people are moving from one compartment to another
    STANDARD = "standard"
    INFECTION_FREQUENCY = "infection_frequency"
    INFECTION_DENSITY = "infection_density"
    TRANSITION_FLOWS = (
        STANDARD,
        INFECTION_FREQUENCY,
        INFECTION_DENSITY,
    )
    INFECTION_FLOWS = (INFECTION_DENSITY, INFECTION_FREQUENCY)

    # Entry flows: where people are entering the population (births, imports)
    BIRTH = "birth"
    IMPORT = "import"
    ENTRY_FLOWS = (BIRTH, IMPORT)

    # Exit flows: where people are leaving the population (deaths, exports)
    DEATH = "compartment_death"
    UNIVERSAL_DEATH = "universal_death"
    DEATH_FLOWS = (DEATH, UNIVERSAL_DEATH)
    EXIT_FLOWS = (DEATH, UNIVERSAL_DEATH)


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
