from summer import Overwrite, Stratification, Multiply

from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS, Tracing, INFECTION


def get_tracing_strat(quarantine_infect_mult, other_infect_mults) -> Stratification:
    """
    Contact tracing stratification to represent those detected actively through screening of first order contacts of
    symptomatic COVID-19 patients presenting passively.
    """

    tracing_strat = Stratification("tracing", [Tracing.TRACED, Tracing.UNTRACED], DISEASE_COMPARTMENTS)

    # Everyone starts out untraced
    pop_split = {Tracing.TRACED: 0., Tracing.UNTRACED: 1.}
    tracing_strat.set_population_split(pop_split)

    # Everybody starts out untraced when they are infected
    traced_infection_adj = {Tracing.TRACED: Multiply(0.), Tracing.UNTRACED: Multiply(1.)}
    tracing_strat.add_flow_adjustments(INFECTION, traced_infection_adj)

    # Ensure the infectiousness adjustments are the same, which ensures that the stratification order doesn't matter
    multipliers_equal = [quarantine_infect_mult == other_infect_mults[mult] for mult in other_infect_mults]
    msg = "Infectiousness of patients in isolation, quarantine and in hospital must be equal"
    assert all(multipliers_equal), msg

    # Apply infectiousness adjustments
    for compartment in DISEASE_COMPARTMENTS:
        traced_infectious_adj = {Tracing.TRACED: Overwrite(quarantine_infect_mult), Tracing.UNTRACED: None}
        tracing_strat.add_infectiousness_adjustments(compartment, traced_infectious_adj)

    return tracing_strat
