from summer import Overwrite, Stratification, Multiply

from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS


def get_tracing_strat(quarantine_infect_multiplier, other_infect_multipliers) -> Stratification:
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

    # Check the effect of isolation is the same as quarantine and hospitalisation
    for infect_multiplier in other_infect_multipliers:
        assert quarantine_infect_multiplier == other_infect_multipliers[infect_multiplier]

    for compartment in DISEASE_COMPARTMENTS:
        tracing_strat.add_infectiousness_adjustments(
            compartment,
            {
                "traced": Overwrite(quarantine_infect_multiplier),
                "untraced": None,
            }
        )

    return tracing_strat
