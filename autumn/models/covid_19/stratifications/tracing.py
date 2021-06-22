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
    Because infectiousness adjustments are set as multipliers and we don't want to adjust twice for contact tracing and
    for hospitalisation or detection, we set infectiousness as a hack at the end for the traced compartments.
    This would generally assume that the infectiousness modification would be the same for tracing and for
    hospitalisation/detection.
    """
    def hack_infectiousness_adjustments(model):
        tracing_comps = [idx for idx, comp in enumerate(model.compartments) if comp.has_stratum("tracing", "traced")]
        for strain in model._disease_strains:
            for idx in tracing_comps:
                model._backend._compartment_infectiousness[strain][idx] = quarantine_infect_multiplier

    return hack_infectiousness_adjustments
