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
        "to": Compartment.EARLY_ACTIVE,
        "parameter": f"within_{Compartment.LATE_EXPOSED}",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.EARLY_ACTIVE,
        "to": Compartment.LATE_ACTIVE,
        "parameter": f"within_{Compartment.EARLY_ACTIVE}",
    },
    # Recovery flows
    {
        "type": Flow.STANDARD,
        "to": Compartment.RECOVERED,
        "origin": Compartment.LATE_ACTIVE,
        "parameter": f"within_{Compartment.LATE_ACTIVE}",
    },
    # Infection death
    {"type": Flow.DEATH, "parameter": "infect_death", "origin": Compartment.LATE_ACTIVE},
]
