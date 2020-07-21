from autumn.constants import Flow, Compartment


DEFAULT_FLOWS = [
    # Infection flows.
    {
        "type": Flow.INFECTION_FREQUENCY,
        "origin": Compartment.SUSCEPTIBLE,
        "to": Compartment.EXPOSED,
        "parameter": "contact_rate",
    },
    # Transition flows.
    {
        "type": Flow.STANDARD,
        "origin": Compartment.EXPOSED,
        "to": Compartment.PRESYMPTOMATIC,
        "parameter": f"within_{Compartment.EXPOSED}",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.PRESYMPTOMATIC,
        "to": Compartment.EARLY_ACTIVE,
        "parameter": f"to_infectious",  # FIXME: Rename to "within_presympt"
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
    {
        "type": Flow.COMPARTMENT_DEATH,
        "parameter": "infect_death",
        "origin": Compartment.LATE_ACTIVE,
    },
]
