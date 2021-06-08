from summer import Stratification, Multiply

from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS
from autumn.models.covid_19.parameters import Parameters


def get_tracing_strat(params: Parameters) -> Stratification:
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

    # This will just be a parameter
    rel_infectiousness_traced = 0.2

    # Adjust infectiousness of those traced
    for comp in DISEASE_COMPARTMENTS:
        tracing_strat.add_infectiousness_adjustments(
            comp,
            {
                "traced": Multiply(rel_infectiousness_traced),
                "untraced": None,
            },
        )

    return tracing_strat
