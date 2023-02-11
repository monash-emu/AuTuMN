
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

class OrganStratum:
    """
    A classification of TB active disease, based on the organ affected and smear-status
    """

    SMEAR_POSITIVE = "smear_positive"
    SMEAR_NEGATIVE = "smear_negative"
    EXTRAPULMONARY = "extrapulmonary"


