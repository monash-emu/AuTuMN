from autumn.constants import Flow
from apps.covid_19.constants import Compartment

DEFAULT_FLOWS = [
    # Infection flows.
    {
        "type": Flow.INFECTION_FREQUENCY,
        "origin": Compartment.SUSCEPTIBLE,
        "to": Compartment.EARLY_EXPOSED,
        "parameter": "contact_rate",
    },
    # Transition flows.
    {
        "type": Flow.STANDARD,
        "origin": Compartment.EARLY_EXPOSED,
        "to": Compartment.LATE_EXPOSED,
        "parameter": f"within_{Compartment.EARLY_EXPOSED}",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.LATE_EXPOSED,
        "to": Compartment.EARLY_INFECTIOUS,
        "parameter": f"to_infectious",  # FIXME: Rename to "within_presympt"
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.EARLY_INFECTIOUS,
        "to": Compartment.LATE_INFECTIOUS,
        "parameter": f"within_{Compartment.EARLY_INFECTIOUS}",
    },
    # Recovery flows
    {
        "type": Flow.STANDARD,
        "to": Compartment.RECOVERED,
        "origin": Compartment.LATE_INFECTIOUS,
        "parameter": f"within_{Compartment.LATE_INFECTIOUS}",
    },
    # Infection death
    {
        "type": Flow.COMPARTMENT_DEATH,
        "parameter": "infect_death",
        "origin": Compartment.LATE_INFECTIOUS,
    },
]
