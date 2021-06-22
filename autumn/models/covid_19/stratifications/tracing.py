from summer import Stratification, Multiply

from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS, NOTIFICATION_CLINICAL_STRATA


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

    return tracing_strat


def make_hack_infectiousness_func(quarantine_infect_multiplier):
    """
    So far the infectiousness of some compartments has been adjusted twice. This concerns the compartment that are both
    traced and detected. This hack will fix this by dividing the infectiousness of the relevant compartments by the
    multiplier that was applied twice.
    """
    def hack_infectiousness_adjustments(model):
        # Find indices of compartments for which infectiousness was adjusted twice
        over_adjusted_compartment_indices = \
            [idx for idx, comp in enumerate(model.compartments) if comp.has_stratum("tracing", "traced")]
        for strain in model._disease_strains:
            for idx in over_adjusted_compartment_indices:
                model._backend._compartment_infectiousness[strain][idx] = quarantine_infect_multiplier

    return hack_infectiousness_adjustments
