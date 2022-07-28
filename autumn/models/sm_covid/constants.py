
# All available compartments
class Compartment:
    SUSCEPTIBLE = "susceptible"
    LATENT = "latent"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"

REPLICABLE_COMPARTMENTS = [Compartment.LATENT, Compartment.INFECTIOUS]

# All available flows
class FlowName:
    PRIMARY_INFECTIOUS_SEEDING = "primary_infectious_seeding"
    INFECTION = "infection"
    PROGRESSION = "progression"
    RECOVERY = "recovery"
    VACCINATION = "vaccination"
    REINFECTION = "reinfection"

# Available strata for the immunity stratification
class ImmunityStratum:
    UNVACCINATED = "unvaccinated"
    VACCINATED = "vaccinated"


IMMUNITY_STRATA = [
    ImmunityStratum.UNVACCINATED,
    ImmunityStratum.VACCINATED
]

LOCATIONS = ["home", "other_locations", "school", "work"]
