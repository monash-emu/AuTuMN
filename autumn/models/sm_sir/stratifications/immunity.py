from typing import Optional, List, Dict
import pandas as pd
import numpy as np

from summer import Stratification, Multiply
from summer import CompartmentalModel

from autumn.tools.inputs.covid_bgd.queries import get_bgd_vac_coverage
from autumn.tools.inputs.covid_phl.queries import get_phl_vac_coverage
from autumn.tools.inputs.covid_btn.queries import get_btn_vac_coverage
from autumn.tools.inputs.covid_mys.queries import get_mys_vac_coverage
from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName
from autumn.models.sm_sir.parameters import ImmunityStratification, VocComponent, TimeSeries
from autumn.tools.dynamic_proportions.solve_transitions import calculate_transition_rates_from_dynamic_props

ACTIVE_FLOWS = {
    "vaccination": ("none", "low"),
    "boosting": ("low", "high"),
    "waning": ("high", "low"),
    "complete_waning": ("low", "none"),
}


def adjust_susceptible_infection_without_strains(
        low_immune_effect: float,
        high_immune_effect: float,
        immunity_strat: Stratification,
):
    """
    Apply the modification to the immunity stratification to account for immunity to first infection (from the
    susceptible compartment), i.e. vaccine-induced immunity.

    Args:
        low_immune_effect: The protection from low immunity
        high_immune_effect: The protection from high immunity
        immunity_strat: The immunity stratification, to be modified

    """

    low_non_cross_multiplier = 1.0 - low_immune_effect
    high_non_cross_multiplier = 1.0 - high_immune_effect

    infection_adjustments = {
        ImmunityStratum.NONE: None,
        ImmunityStratum.LOW: Multiply(low_non_cross_multiplier),
        ImmunityStratum.HIGH: Multiply(high_non_cross_multiplier),
    }

    immunity_strat.set_flow_adjustments(
        FlowName.INFECTION,
        infection_adjustments,
    )


def adjust_susceptible_infection_with_strains(
        low_immune_effect: float,
        high_immune_effect: float,
        immunity_strat: Stratification,
        voc_params: Optional[Dict[str, VocComponent]],
):
    """
    Apply the modification to the immunity stratification to account for immunity to first infection (from the
    susceptible compartment), accounting for the extent to which each VoC is immune-escape to vaccine-induced immunity.

    Args:
        low_immune_effect: The protection from low immunity
        high_immune_effect: The protection from high immunity
        immunity_strat: The immunity stratification, to be modified
        voc_params: The parameters relating to the VoCs being implemented

    """

    for infecting_strain in voc_params:
        strain_immunity_modifier = 1.0 - voc_params[infecting_strain].immune_escape
        low_non_cross_multiplier = 1.0 - low_immune_effect * strain_immunity_modifier
        high_non_cross_multiplier = 1.0 - high_immune_effect * strain_immunity_modifier

        infection_adjustments = {
            ImmunityStratum.NONE: None,
            ImmunityStratum.LOW: Multiply(low_non_cross_multiplier),
            ImmunityStratum.HIGH: Multiply(high_non_cross_multiplier),
        }

        immunity_strat.set_flow_adjustments(
            FlowName.INFECTION,
            infection_adjustments,
            dest_strata={"strain": infecting_strain},
        )


def adjust_reinfection_without_strains(
        low_immune_effect: float,
        high_immune_effect: float,
        immunity_strat: Stratification,
        reinfection_flows: List[str],
):
    """
    Adjust the rate of reinfection for immunity, in cases in which we don't need to worry about cross-strain immunity,
    because the model has not been stratified by strain.

    Args:
        low_immune_effect: The protection from low immunity
        high_immune_effect: The protection from high immunity
        immunity_strat: The immunity stratification, to be modified
        reinfection_flows: The names of the transition flows representing reinfection

    """

    low_non_cross_multiplier = 1.0 - low_immune_effect
    high_non_cross_multiplier = 1.0 - high_immune_effect

    # The infection processes that we are adapting and for which cross-strain immunity is relevant
    for flow in reinfection_flows:

        # Combine the two mechanisms of protection and format for summer
        reinfection_adjustments = {
            ImmunityStratum.NONE: None,
            ImmunityStratum.LOW: Multiply(low_non_cross_multiplier),
            ImmunityStratum.HIGH: Multiply(high_non_cross_multiplier),
        }

        # Apply to the stratification object
        immunity_strat.set_flow_adjustments(
            flow,
            reinfection_adjustments,
        )


def adjust_reinfection_with_strains(
        low_immune_effect: float,
        high_immune_effect: float,
        immunity_strat: Stratification,
        reinfection_flows: List[str],
        voc_params: Optional[Dict[str, VocComponent]],
):
    """
    Adjust the rate of reinfection for immunity, in cases in which we do need to worry about cross-strain immunity, so
    we have to consider every possible combination of cross-immunity between strains (including the immunity conferred
    by infection with a certain strain and reinfection with that same strain).

    Args:
        low_immune_effect: The protection from low immunity
        high_immune_effect: The protection from high immunity
        immunity_strat: The immunity stratification, to be modified
        reinfection_flows: The names of the transition flows representing reinfection
        voc_params: The parameters relating to the VoCs being implemented

    """

    for infecting_strain in voc_params:

        # The immunity effect for vaccine or non-cross-strain natural immunity escape properties of the strain
        non_cross_effect = 1.0 - voc_params[infecting_strain].immune_escape
        low_non_cross_multiplier = 1.0 - low_immune_effect * non_cross_effect
        high_non_cross_multiplier = 1.0 - high_immune_effect * non_cross_effect

        # Considering people recovered from infection with each modelled strain
        for infected_strain in voc_params:
            for flow in reinfection_flows:

                # Cross protection from previous infection with the "infected" strain against the "infecting" strain
                cross_effect = 1.0 - getattr(voc_params[infected_strain].cross_protection[infecting_strain], flow)

                # Combine the two mechanisms of protection
                reinfection_adjustments = {
                    ImmunityStratum.NONE: Multiply(cross_effect),
                    ImmunityStratum.LOW: Multiply(low_non_cross_multiplier * cross_effect),
                    ImmunityStratum.HIGH: Multiply(high_non_cross_multiplier * cross_effect),
                }

                immunity_strat.set_flow_adjustments(
                    flow,
                    reinfection_adjustments,
                    source_strata={"strain": infected_strain},
                    dest_strata={"strain": infecting_strain},
                )


def get_immunity_strat(
        compartments: List[str],
        immunity_params: ImmunityStratification,
) -> Stratification:
    """
    This stratification is intended to capture all the immunity consideration.
    This relates both to "cross" immunity between the strains being simulated (based on the strain that a person was
    most recently infected with) and immunity induced by processes other than the strains being simulated in the model
    (including vaccination-induced immunity and natural immunity from strains preceding the current simulations).

    Args:
        compartments: Base model compartments
        immunity_params: All the immunity-related model parameters

    Returns:
        The summer Stratification object that captures immunity from anything other than cross-immunity between strains

    """

    # Create the immunity stratification, which applies to all compartments
    immunity_strat = Stratification("immunity", IMMUNITY_STRATA, compartments)

    # Set distribution of starting population
    p_immune = immunity_params.prop_immune
    p_high_among_immune = immunity_params.prop_high_among_immune
    immunity_split_props = {
        ImmunityStratum.NONE: 1.0 - p_immune,
        ImmunityStratum.LOW: p_immune * (1.0 - p_high_among_immune),
        ImmunityStratum.HIGH: p_immune * p_high_among_immune,
    }
    immunity_strat.set_population_split(immunity_split_props)

    return immunity_strat


def apply_reported_vacc_coverage(
        compartment_types: List[str],
        model: CompartmentalModel,
        iso3: str,
        thinning: int,
        model_start_time: int,
        start_immune_prop: float,
        additional_immunity_points: TimeSeries,
):
    """
    Collage up the reported values for vaccination coverage for a country and then call add_dynamic_immunity_to_model to
    apply it to the model as a dynamic stratum.

    Args:
        compartment_types: Unstratified model compartment types being implemented
        model: The model itself
        iso3: The ISO-3 code for the country being implemented
        thinning: Thin out the empiric data to save time with curve fitting and because this must be >=2 (as below)
        model_start_time: Model starting time
        start_immune_prop: Vaccination coverage at the time that the model starts running

    """

    if iso3 == "BGD":
        raw_data = get_bgd_vac_coverage(region="BGD", vaccine="total", dose=2)
    elif iso3 == "PHL":
        raw_data = get_phl_vac_coverage(dose="SECOND_DOSE")
    elif iso3 == "BTN":
        raw_data = get_btn_vac_coverage(region="Bhutan", dose=2)
    elif iso3 == "MYS":
        raw_data = get_mys_vac_coverage(dose="full")

    # Add on the starting effective coverage value
    if iso3 == "BGD" or iso3 == "PHL" or iso3 == "BTN" or iso3 == "MYS":
        vaccine_data = pd.concat(
            (
                pd.Series({model_start_time: start_immune_prop}),
                raw_data
            )
        )
    else:
        vaccine_data = pd.Series({model_start_time: start_immune_prop})

    # Add user-requested additional immunity points
    if additional_immunity_points:
        additional_vacc_series = pd.Series({k: v for k, v in zip(additional_immunity_points.times, additional_immunity_points.values)})
        vacc_data_with_waning = pd.concat((vaccine_data, additional_vacc_series))
    else:
        vacc_data_with_waning = vaccine_data

    # Be explicit about all the difference immunity categories
    vaccine_df = pd.DataFrame(
        {
            "none": 1. - vacc_data_with_waning,
            "low": vacc_data_with_waning,
        },
    )
    vaccine_df["high"] = 0.

    # Apply to model, as below
    thinned_df = vaccine_df[::thinning] if thinning else vaccine_df
    add_dynamic_immunity_to_model(compartment_types, thinned_df, model)


def add_dynamic_immunity_to_model(
        compartments: List[str],
        strata_distributions: pd.DataFrame,
        model: CompartmentalModel,
):
    """
    Use the dynamic flow processes to control the distribution of the population by vaccination status.

    Args:
        strata_distributions: The target proportions at each time point
        model: The model to be adapted
        compartments: The types of compartment being implemented in the model, before stratification

    """

    sc_functions = calculate_transition_rates_from_dynamic_props(strata_distributions, ACTIVE_FLOWS)
    for comp in compartments:
        for transition, strata in ACTIVE_FLOWS.items():
            model.add_transition_flow(
                transition,
                sc_functions[transition],
                comp,
                comp,
                source_strata={"immunity": strata[0]},
                dest_strata={"immunity": strata[1]},
            )
