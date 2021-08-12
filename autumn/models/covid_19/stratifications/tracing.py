from summer import Overwrite, Stratification, Multiply

from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS, Tracing


def get_tracing_strat(quarantine_infect_multiplier, other_infect_multipliers) -> Stratification:
    tracing_strat = Stratification(
        "tracing",
        [Tracing.TRACED, Tracing.UNTRACED],
        DISEASE_COMPARTMENTS
    )

    # Current default for everyone to start out untraced
    tracing_strat.set_population_split(
        {
            Tracing.TRACED: 0.,
            Tracing.UNTRACED: 1.,
        }
    )

    # Apply the contact tracing
    tracing_strat.add_flow_adjustments(
        "infection",
        {
            Tracing.TRACED: Multiply(0.),
            Tracing.UNTRACED: Multiply(1.),
        }
    )

    # Check the effect of isolation is the same as quarantine and hospitalisation
    for infect_multiplier in other_infect_multipliers:
        assert quarantine_infect_multiplier == other_infect_multipliers[infect_multiplier]

    for compartment in DISEASE_COMPARTMENTS:
        tracing_strat.add_infectiousness_adjustments(
            compartment,
            {
                Tracing.TRACED: Overwrite(quarantine_infect_multiplier),
                Tracing.UNTRACED: None,
            }
        )

    return tracing_strat
