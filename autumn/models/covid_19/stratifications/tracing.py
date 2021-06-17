from summer import Stratification, Multiply

from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS


def get_tracing_strat(contact_params) -> Stratification:
    tracing_strat = Stratification(
        "tracing",
        ["traced", "untraced"],
        DISEASE_COMPARTMENTS
    )

    # Current default for everyone to start out untraced
    tracing_strat.set_population_split(
        {
            "traced": 0.,
            "untraced": 1.,
        }
    )

    # Apply the contact tracing
    tracing_strat.add_flow_adjustments(
        "infection",
        {
            "traced": Multiply(0.),
            "untraced": Multiply(1.),
        }
    )

    # Adjust infectiousness of those traced
    for comp in DISEASE_COMPARTMENTS:
        tracing_strat.add_infectiousness_adjustments(
            comp,
            {
                "traced": Multiply(contact_params.quarantine_infect_multiplier),
                "untraced": None,
            },
        )

    return tracing_strat
