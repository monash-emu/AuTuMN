from summer import Stratification, Multiply

from autumn.models.covid_19.constants import INFECTIOUS_COMPARTMENTS
from autumn.models.covid_19.parameters import Parameters


def get_tracing_strat(params: Parameters) -> Stratification:
    tracing_strat = Stratification(
        "tracing",
        ["traced", "untraced"],
        INFECTIOUS_COMPARTMENTS
    )

    # Current default for everyone to start out untraced
    tracing_strat.set_population_split(
        {
            "traced": 0.,
            "untraced": 1.,
        }
    )

    # This thing needs to be a function that depends on incidence (as it emerges from the model) and CDR
    prop_traced = lambda t: 0.
    prop_not_traced = lambda t: 1. - prop_traced(t)

    # Apply the contact tracing
    tracing_strat.add_flow_adjustments(
        "infect_onset",
        {
            "traced": Multiply(prop_traced),
            "untraced": Multiply(prop_not_traced),
        }
    )

    # This will just be a parameter
    rel_infectiousness_traced = 0.2

    # Adjust infectiousness of those traced
    for comp in INFECTIOUS_COMPARTMENTS:
        tracing_strat.add_infectiousness_adjustments(
            comp,
            {
                "traced": Multiply(rel_infectiousness_traced),
                "untraced": None,
            },
        )

    return tracing_strat
