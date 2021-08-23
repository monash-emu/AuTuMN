from summer import Overwrite, Stratification, Multiply

from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS, Tracing, INFECTION


def get_tracing_strat(quarantine_infect_multiplier, other_infect_multipliers) -> Stratification:
    """
    Contact tracing stratification to represent those detected actively through screening of first order contacts of
    symptomatic COVID-19 patients presenting passively.
    """

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
        INFECTION,
        {
            Tracing.TRACED: Multiply(0.),
            Tracing.UNTRACED: Multiply(1.),
        }
    )

    # Ensure the effects of isolation, quarantine and admission are the same
    # (to ensure the stratification order doesn't matter)
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
