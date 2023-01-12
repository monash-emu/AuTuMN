
# All available compartments
class Compartment:
    """
    A tuberculosis model compartment.
    """
    SUSCEPTIBLE = "susceptible"
    EARLY_LATENT = "early_latent"
    LATE_LATENT = "late_latent"
    INFECTIOUS = "infectious"
    ON_TREATMENT = "on_treatment"
    RECOVERED = "recovered"



#REPLICABLE_COMPARTMENTS = [Compartment.LATENT, Compartment.INFECTIOUS]

BASE_COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.EARLY_LATENT,
    Compartment.LATE_LATENT,
    Compartment.INFECTIOUS,
    Compartment.ON_TREATMENT,
    Compartment.RECOVERED,
]
INFECTIOUS_COMPS = [
    Compartment.INFECTIOUS,
    Compartment.ON_TREATMENT,
]

LATENT_COMPS = [
    Compartment.EARLY_LATENT,
    Compartment.LATE_LATENT,
]


LOCATIONS = ["home", "other_locations", "school", "work"]

class FlowName:
    INFECTION = "infection"
    WITHIN_LATENT = "within_latent"
    PROGRESSION = "progression"
    WITHIN_INFECTIOUS = "within_infectious"
    RECOVERY = "recovery"
    WANING = "waning"
    EARLY_REINFECTION = "early_reinfection"
    LATE_REINFECTION = "late_reinfection"
