
# All available compartments
class Compartment:
    SUSCEPTIBLE = "susceptible"
    LATENT = "latent"
    LATENT_LATE = "latent_late"
    INFECTIOUS = "infectious"
    INFECTIOUS_LATE = "infectious_late"
    RECOVERED = "recovered"
    WANED = "waned"


# All available flows
class FlowName:
    INFECTION = "infection"
    PROGRESSION = "progression"
    RECOVERY = "recovery"
    VACCINATION = "vaccination"


# Routinely implemented compartments
BASE_COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.LATENT,
    Compartment.INFECTIOUS,
    Compartment.RECOVERED,
]


# Available strata for the immunity stratification
class ImmunityStratum:
    UNVACCINATED = "unvaccinated"
    VACCINATED = "vaccinated"


IMMUNITY_STRATA = [
    ImmunityStratum.UNVACCINATED,
    ImmunityStratum.VACCINATED
]

LOCATIONS = ["home", "other_locations", "school", "work"]
