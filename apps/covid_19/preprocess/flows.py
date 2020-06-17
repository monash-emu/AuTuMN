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
