from typing import Dict

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import (
    AGE_CLINICAL_TRANSITIONS, PROGRESS, COMPARTMENTS, DISEASE_COMPARTMENTS, Vaccination, INFECTION
)
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.strat_processing.clinical import (
    add_clinical_adjustments_to_strat, get_all_adjustments, get_blank_adjustments_for_strat,
    update_adjustments_for_strat
)
from autumn.models.covid_19.strat_processing.vaccination import get_stratum_vacc_effect


def get_vaccination_strat(params: Parameters, all_strata: list, vocs: Dict[str, float]) -> Stratification:
    """
    This vaccination stratification ist three strata applied to all compartments of the model.
    First create the stratification object and split the starting population.
    """

    # Create the stratum
    stratification = Stratification("vaccination", all_strata, COMPARTMENTS)

    # Initial conditions, everyone unvaccinated
    pop_split = {stratum: 0. for stratum in all_strata}
    pop_split[Vaccination.UNVACCINATED] = 1.
    stratification.set_population_split(pop_split)

    # Preliminaries
    vacc_params = params.vaccination
    ve_infection, ve_severity, sympt_adjusters, hosp_adjusters, ifr_adjusters, vacc_effects = {}, {}, {}, {}, {}, {}

    # Get vaccination effect parameters in the form needed for the model
    flow_adjs = get_blank_adjustments_for_strat([PROGRESS, *AGE_CLINICAL_TRANSITIONS], vocs)
    vacc_strata = all_strata[1:]  # Affected strata are all but the first

    for stratum in vacc_strata:
        for voc in vocs.keys():
            vacc_effects[stratum], sympt_adjuster, hosp_adjuster, ifr_adjuster = get_stratum_vacc_effect(
                params, stratum, voc_effect=vocs[voc]
            )

            # Get and apply the severity modifications (in the same way as for history stratification)
            severity_adjs = get_all_adjustments(
                params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
                params.sojourn, ifr_adjuster, sympt_adjuster, hosp_adjuster
            )
            update_adjustments_for_strat(stratum, flow_adjs, severity_adjs, voc)
    add_clinical_adjustments_to_strat(stratification, flow_adjs, Vaccination.UNVACCINATED, vocs)

    # Vaccination effect against infection
    infect_adjs = {stratum: Multiply(1. - vacc_effects[stratum]["infection_efficacy"]) for stratum in vacc_strata}
    infect_adjs.update({Vaccination.UNVACCINATED: None})
    stratification.add_flow_adjustments(INFECTION, infect_adjs)

    # Vaccination effect against infectiousness
    infectiousness_adjs = {s: Multiply(1. - getattr(getattr(vacc_params, s), "ve_infectiousness")) for s in vacc_strata}
    infectiousness_adjs.update({Vaccination.UNVACCINATED: None})
    for compartment in DISEASE_COMPARTMENTS:
        stratification.add_infectiousness_adjustments(compartment, infectiousness_adjs)

    # Simplest approach for VoCs is to assign all the VoC infectious seed to the unvaccinated
    # FIXME: This can probably be deleted, once the summer importations split is fixed
    if params.voc_emergence:
        for voc_name, voc_values in params.voc_emergence.items():
            seed_split = {stratum: Multiply(0.) for stratum in vacc_strata}
            seed_split[Vaccination.UNVACCINATED] = Multiply(1.)
            stratification.add_flow_adjustments(f"seed_voc_{voc_name}", seed_split)

    return stratification
