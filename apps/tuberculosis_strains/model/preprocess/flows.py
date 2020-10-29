from autumn.constants import Flow
from apps.tuberculosis_strains.constants import Compartment


DEFAULT_FLOWS = [
    # Infection flows.
    {
        "type": Flow.INFECTION_FREQUENCY,
        "origin": Compartment.SUSCEPTIBLE,
        "to": Compartment.EARLY_LATENT,
        "parameter": "contact_rate",
    },
    {
        "type": Flow.INFECTION_FREQUENCY,
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.EARLY_LATENT,
        "parameter": "contact_rate_from_latent",
    },
    {
        "type": Flow.INFECTION_FREQUENCY,
        "origin": Compartment.RECOVERED,
        "to": Compartment.EARLY_LATENT,
        "parameter": "contact_rate_from_recovered",
    },
    # Transition flows.
    {
        "type": Flow.STANDARD,
        "origin": Compartment.EARLY_LATENT,
        "to": Compartment.LATE_LATENT,
        "parameter": "stabilisation_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.EARLY_LATENT,
        "to": Compartment.INFECTIOUS,
        "parameter": "early_activation_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.INFECTIOUS,
        "parameter": "late_activation_rate",
    },
    # Post-active-disease flows
    {
        "type": Flow.STANDARD,
        "origin": Compartment.INFECTIOUS,
        "to": Compartment.LATE_LATENT,
        "parameter": "self_recovery_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.INFECTIOUS,
        "to": Compartment.ON_TREATMENT,
        "parameter": "detection_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.ON_TREATMENT,
        "to": Compartment.RECOVERED,
        "parameter": "treatment_recovery_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.ON_TREATMENT,
        "to": Compartment.INFECTIOUS,
        "parameter": "relapse_rate",
    },
    # Infection death
    {"type": Flow.DEATH, "parameter": "infect_death_rate", "origin": Compartment.INFECTIOUS},
    {"type": Flow.DEATH, "parameter": "treatment_death_rate", "origin": Compartment.ON_TREATMENT},
]