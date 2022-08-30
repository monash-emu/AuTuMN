from typing import Optional, List, Dict
import pandas as pd
from copy import deepcopy

from summer import Stratification, Multiply
from summer import CompartmentalModel

from autumn.core.utils.pandas import lagged_cumsum
from autumn.core.inputs.covid_bgd.queries import get_bgd_vac_coverage
from autumn.core.inputs.covid_phl.queries import get_phl_vac_coverage
from autumn.core.inputs.covid_btn.queries import get_btn_vac_coverage
from autumn.core.inputs.covid_mys.queries import get_mys_vac_coverage
from autumn.core.inputs.covid_au.queries import get_nt_vac_coverage
from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName
from autumn.models.sm_sir.parameters import ImmunityStratification, VocComponent, TimeSeries
from autumn.model_features.solve_transitions import calculate_transition_rates_from_dynamic_props
from autumn.model_features.outputs import get_strata


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
    susceptible compartment), i.e. vaccine-induced immunity (or for some models this stratification could be taken
    to represent past infection prior to the beginning of the simulation period).

    Args:
        low_immune_effect: The protection from low immunity
        high_immune_effect: The protection from high immunity
        immunity_strat: The immunity stratification, to be modified

    """

    # The infection rate accounting for vaccination-induced immunity (using similar naming as for when we do the same with strains below)
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
    This same function can be applied to the model wherever VoCs are included, regardless of strain structure,
    because the strain stratification (representing history of last infecting strain) does not apply here.

    Args:
        low_immune_effect: The protection from low immunity
        high_immune_effect: The protection from high immunity
        immunity_strat: The immunity stratification, to be modified
        voc_params: The parameters relating to the VoCs being implemented

    """

    for infecting_strain in voc_params:
        
        # The vaccination-specific immunity that will be retained after allowing for the strain's immune escape against vaccination-induced immunity
        non_cross_effect = 1.0 - voc_params[infecting_strain].immune_escape

        # Adjust the rate of infection considering the protection of that immunity status (incorporating the strain's escape properties)
        low_non_cross_multiplier = 1.0 - low_immune_effect * non_cross_effect
        high_non_cross_multiplier = 1.0 - high_immune_effect * non_cross_effect

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
    Works very similarly to adjust_susceptible_infection_without_strains,
    except that we loop over two flow types for early and late reinfection.

    Args:
        low_immune_effect: The protection from low immunity
        high_immune_effect: The protection from high immunity
        immunity_strat: The immunity stratification, to be modified
        reinfection_flows: The names of the transition flows representing reinfection

    """

    # The infection rate accounting for vaccination-induced immunity (using similar naming as for when we do the same with strains below)
    low_non_cross_multiplier = 1.0 - low_immune_effect
    high_non_cross_multiplier = 1.0 - high_immune_effect

    # For both the early and late reinfection transitions
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

        # The vaccination-specific immunity that will be retained after allowing for the strain's immune escape against vaccination-induced immunity
        non_cross_effect = 1.0 - voc_params[infecting_strain].immune_escape

        # Adjust the rate of infection considering the protection of that immunity status (incorporating the strain's escape properties)
        low_non_cross_multiplier = 1.0 - low_immune_effect * non_cross_effect
        high_non_cross_multiplier = 1.0 - high_immune_effect * non_cross_effect

        # Considering people recovered from infection with each modelled strain
        for infected_strain in voc_params:

            # For both the early and late reinfection transitions
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


def get_reported_vacc_coverage(iso3, start_age, end_age, age_specific_vacc):

    # Get the raw data from the loading functions and drop rows with any nans
    if iso3 == "PHL":
        full_series = get_phl_vac_coverage(dose="SECOND_DOSE")
        booster_series = get_phl_vac_coverage(dose="BOOSTER_DOSE")
        assert not age_specific_vacc, "Philippines data not age-specific, so just replicating calculations"
    elif iso3 == "MYS":
        full_series = get_mys_vac_coverage(dose="full")
        booster_series = get_mys_vac_coverage(dose="booster")
        assert not age_specific_vacc, "Malaysia data not age-specific, so just replicating calculations"
    elif iso3 == "AUS":
        full_series = get_nt_vac_coverage(dose=2, start_age=start_age, end_age=end_age)
        booster_series = get_nt_vac_coverage(dose=3, start_age=start_age, end_age=end_age)
    else:
        raise ValueError("Data for country not available (in this function)")
    
    vaccine_data = pd.DataFrame(
        {
            "full": full_series,
            "boost": booster_series,
        }
    ).dropna(axis=0)

    return vaccine_data


def apply_vacc_coverage(
        model: CompartmentalModel,
        iso3: str,
        thinning: int,
        start_immune_prop: float,
        start_prop_high_among_immune: float,
        boosting: bool=True,
        age_specific_vacc: bool=False,
        booster_effect_duration: float=0.,
        extra_coverage: dict={},
):
    """
    Collate up the reported values for vaccination coverage for a country and then call add_dynamic_immunity_to_model to
    apply it to the model as a dynamic stratum.
    Args:
        compartments: Unstratified model compartment types being implemented
        model: The model itself
        iso3: The ISO-3 code for the country being implemented
        thinning: Thin out the empiric data to save time with curve fitting and because this must be >=2 (as below)
        start_immune_prop: Vaccination coverage at the time that the model starts running
    """

    age_vacc_categories = get_strata(model, "agegroup") if age_specific_vacc else ["all_ages"]

    msg = "Age group in requests not present in model or 'all_ages' if vaccination not age-specific"
    assert all([i in age_vacc_categories for i in extra_coverage.keys()]), msg

    for i_age, age_cat in enumerate(age_vacc_categories[3:]):

        start_age = int(age_vacc_categories[i_age]) if age_cat != age_vacc_categories[0] else None
        end_age = int(age_vacc_categories[i_age + 1]) if age_cat != age_vacc_categories[-1] else None

        # Get the data
        vaccine_data = get_reported_vacc_coverage(iso3, start_age, end_age, age_specific_vacc)

        # Get rid of any data that is from before the model starts running
        model_start_time = model.times[0]
        vaccine_data = vaccine_data[model_start_time < vaccine_data.index]

        # Add a column for the proportion of the population recently vaccinated
        if booster_effect_duration and not vaccine_data.empty:
            vaccine_data["recent_boost"] = lagged_cumsum(
                vaccine_data["boost"].diff(), 
                booster_effect_duration,
            )

        # Add on the user requested starting proportions
        vaccine_data.loc[model_start_time] = {
            "full": start_immune_prop, 
            "boost": start_immune_prop * start_prop_high_among_immune,
        }

        # Add on any custom user requests
        age_user_request = extra_coverage.get(age_cat)
        if age_user_request:
            msg = f"Request for {age_cat} does not have standard keys"
            assert list(age_user_request.keys()) == ["full", "boost", "index"], msg
            msg = f"Request for {age_cat} does not have same number of observations for full, boost and index"
            assert len(set([len(vals) for vals in age_user_request.values()])) == 1, msg

            request_index = age_user_request.pop("index")
            request_df = pd.DataFrame.from_dict(age_user_request)
            request_df.index = request_index
            vaccine_data.append(request_df)

        # Sort
        vaccine_data.sort_index(inplace=True)

        # Thin as per user request
        vaccine_data = vaccine_data[::thinning]

        # Format the data to match the model's immunity structure
        vaccine_data["never"] = 1. - vaccine_data["full"]
        if boosting and booster_effect_duration:
            vaccine_data["full_only"] = vaccine_data["full"] - vaccine_data["recent_boost"]
            strata_data = vaccine_data[["never", "full_only", "recent_boost"]]
            strata_data.columns = ["none", "low", "high"]
        elif boosting:
            vaccine_data["full_only"] = vaccine_data["full"] - vaccine_data["boost"]
            strata_data = vaccine_data[["never", "full_only", "boost"]]
            strata_data.columns = ["none", "low", "high"]
        else:
            strata_data = vaccine_data[["never", "full"]]
            strata_data.columns = ["none", "low"]
            strata_data["high"] = 0.

        # Apply to model
        add_dynamic_immunity_to_model(
            strata_data,
            model, 
            age_cat,
        )


def add_dynamic_immunity_to_model(
        strata_distributions: pd.DataFrame,
        model: CompartmentalModel,
        agegroup: str,
):
    """
    Use the dynamic flow processes to control the distribution of the population by vaccination status.

    Args:
        compartments: The types of compartment being implemented in the model, before stratification
        strata_distributions: The target proportions at each time point
        model: The model to be adapted
        agegroup: Relevant agegroup for vaccination flow
    """
    sc_functions = calculate_transition_rates_from_dynamic_props(strata_distributions, ACTIVE_FLOWS)
    age_filter = {} if agegroup == "all_ages" else {"agegroup": agegroup}
    for comp in list(set([comp.name for comp in model.compartments])):
        for transition, strata in ACTIVE_FLOWS.items():
            model.add_transition_flow(
                transition,
                sc_functions[transition],
                comp,
                comp,
                source_strata={"immunity": strata[0], **age_filter},
                dest_strata={"immunity": strata[1], **age_filter},
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
        compartments: Unstratified model compartment types being implemented
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
