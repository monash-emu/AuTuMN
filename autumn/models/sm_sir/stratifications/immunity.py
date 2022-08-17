from typing import Optional, List, Dict
import pandas as pd
from copy import deepcopy

from summer import Stratification, Multiply
from summer import CompartmentalModel

from autumn.core.inputs.covid_bgd.queries import get_bgd_vac_coverage
from autumn.core.inputs.covid_phl.queries import get_phl_vac_coverage
from autumn.core.inputs.covid_btn.queries import get_btn_vac_coverage
from autumn.core.inputs.covid_mys.queries import get_mys_vac_coverage
from autumn.core.inputs.covid_au.queries import get_nt_vac_coverage
from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName
from autumn.models.sm_sir.parameters import ImmunityStratification, VocComponent, TimeSeries
from autumn.model_features.solve_transitions import calculate_transition_rates_from_dynamic_props

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


def apply_reported_vacc_coverage(
        compartments: List[str],
        model: CompartmentalModel,
        iso3: str,
        thinning: int,
        model_start_time: int,
        start_immune_prop: float,
):
    """
    Collate up the reported values for vaccination coverage for a country and then call add_dynamic_immunity_to_model to
    apply it to the model as a dynamic stratum.

    Args:
        compartments: Unstratified model compartment types being implemented
        model: The model itself
        iso3: The ISO-3 code for the country being implemented
        thinning: Thin out the empiric data to save time with curve fitting and because this must be >=2 (as below)
        model_start_time: Model starting time
        start_immune_prop: Vaccination coverage at the time that the model starts running

    """

    # Get the raw data from the loading functions
    if iso3 == "BGD":
        raw_data = get_bgd_vac_coverage(region="BGD", vaccine="total", dose=2)
    elif iso3 == "BTN":
        raw_data = get_btn_vac_coverage(region="Bhutan", dose=2)
    elif iso3 == "MYS":
        raw_data = get_mys_vac_coverage(dose="full")
    elif iso3 == "AUS":
        raw_data = get_nt_vac_coverage(dose=2)

    # Get rid of any data that is from before the model starts running
    vaccine_data = raw_data[model_start_time < raw_data.index]

    # Add on the user requested starting proportion
    vaccine_data = pd.concat(
        (
            pd.Series({model_start_time: start_immune_prop}), 
            vaccine_data,
        )
    )

    # Be explicit about each of the three immunity categories
    vaccine_df = pd.DataFrame(
        {
            "none": 1. - vaccine_data,
            "low": vaccine_data,
            "high": 0.,
        },
    )

    # Apply to model
    add_dynamic_immunity_to_model(
        compartments, 
        vaccine_df[::thinning], 
        model, 
        "all_ages"
    )


def get_recently_vaccinated_prop(coverage_df: pd.DataFrame, recent_timeframe: float) -> pd.DataFrame:
    """
    Calculate the proportion of the population vaccinated in a given recent timeframe over time.

    Args:
        coverage_df: raw coverage data over time
        recent_timeframe: duration in days used to define the recent timeframe

    """
    # Calculate incremental vaccine coverage
    vaccine_increment_df = coverage_df.diff()
    vaccine_increment_df.iloc[0] = coverage_df.iloc[0]

    # Calculate cumulative proportion recently vaccinated
    recent_prop_df = coverage_df.copy()
    for index in coverage_df.index:
        recent_prop_df.loc[index] = vaccine_increment_df.loc[(vaccine_increment_df.index > index - recent_timeframe) & (vaccine_increment_df.index <= index)].sum()

    return recent_prop_df


def apply_reported_vacc_coverage_with_booster(
        compartment_types: List[str],
        model: CompartmentalModel,
        age_groups: List[str],
        iso3: str,
        region: str,
        thinning: int,
        model_start_time: int,
        start_immune_prop: float,
        start_prop_high_among_immune: float,
        booster_effect_duration: float,
        future_monthly_booster_rate: float,
        future_booster_age_allocation,
        age_pops: pd.Series,
        model_end_time: float,
):
    """
    Collage up the reported values for vaccination coverage for a country and then call add_dynamic_immunity_to_model to
    apply it to the model as a dynamic stratum.

    Args:
        compartment_types: Unstratified model compartment types being implemented
        model: The model itself
        age_groups: List of model age groups
        iso3: The ISO-3 code for the country being implemented
        region: The region of the country that is implemented
        thinning: Thin out the empiric data to save time with curve fitting and because this must be >=2 (as below)
        model_start_time: Model starting time
        start_immune_prop: Vaccination coverage at the time that the model starts running
        start_prop_high_among_immune: Starting proportion of highly immune individuals among vaccinated
        booster_effect_duration: Duration of maximal vaccine protection after booster dose (in days)
        future_monthly_booster_rate: Monthly booster rate used to collate additional booster data in the future
        future_booster_age_allocation: Dictionary containing proportion of future booster doses by age group
        age_pops: population by age
        model_end_time: Model end time
    """
    historical_vacc_data = get_historical_vacc_data(iso3, region, model_start_time, start_immune_prop, start_prop_high_among_immune)
   
    if future_monthly_booster_rate is None:
        future_monthly_booster_rate = 0.      

    is_booster_age_specific = future_booster_age_allocation is not None
    if is_booster_age_specific:
        if isinstance(future_booster_age_allocation, dict):
            extra_coverage_by_age = {str(agegroup): p * future_monthly_booster_rate / age_pops.loc[str(agegroup)] for agegroup, p in future_booster_age_allocation.items()}             
            for agegroup in age_groups: 
                # Create dataframe with dynamic distributions including future booster rates and waning
                extra_coverage = extra_coverage_by_age[agegroup] if agegroup in extra_coverage_by_age else 0.
                dynamic_strata_distributions =  get_immune_strata_distributions_from_fixed_increment(historical_vacc_data, extra_coverage, model_end_time, booster_effect_duration, thinning)

                # Add transition flows to the models
                add_dynamic_immunity_to_model(compartment_types, dynamic_strata_distributions, model, agegroup)
        else:  # future_booster_age_allocation is a list of ordered prioritised age groups.
            # First deal with flows relevant to the prioritised age groups
            allocated_doses = {}
            for priority_agegroup in future_booster_age_allocation:
                dynamic_strata_distributions, agegroup_allocated_doses = get_immune_strata_distributions_using_priority(
                    historical_vacc_data, 
                    future_monthly_booster_rate, 
                    model_end_time, 
                    booster_effect_duration, 
                    thinning, 
                    age_pops.loc[str(priority_agegroup)],
                    allocated_doses
                )

                allocated_doses = agegroup_allocated_doses if not allocated_doses else {t: allocated_doses[t] + agegroup_allocated_doses[t] for t in allocated_doses}

                # Add transition flows to the models
                add_dynamic_immunity_to_model(compartment_types, dynamic_strata_distributions, model, str(priority_agegroup))

            # Deal with the flows relevant to the populations not eligible for future booster,
            non_eligible_agegroups = [agegroup for agegroup in age_groups if int(agegroup) not in future_booster_age_allocation]
            for non_eligible_agegroup in non_eligible_agegroups:
                dynamic_strata_distributions =  get_immune_strata_distributions_from_fixed_increment(historical_vacc_data, 0., model_end_time, booster_effect_duration, thinning)
                # Add transition flows to the models
                add_dynamic_immunity_to_model(compartment_types, dynamic_strata_distributions, model, non_eligible_agegroup)

    else:
        # Create dataframe with dynamic distributions including future booster rates and waning
        extra_coverage = future_monthly_booster_rate / age_pops.sum()
        dynamic_strata_distributions =  get_immune_strata_distributions_from_fixed_increment(historical_vacc_data, extra_coverage, model_end_time, booster_effect_duration, thinning)

        # Add transition flows to the models
        add_dynamic_immunity_to_model(compartment_types, dynamic_strata_distributions, model, "all_ages")    


def get_historical_vacc_data(iso3, region, model_start_time, start_immune_prop, start_prop_high_among_immune) -> pd.DataFrame:
    """_summary_

     Args:
        iso3: The ISO-3 code for the country being implemented
        region: The region of the country that is implemented
        model_start_time: Model starting time
        start_immune_prop: Vaccination coverage at the time that the model starts running
        start_prop_high_among_immune: Starting proportion of highly immune individuals among vaccinated       

    Returns:
        pd.DataFrame: dataframe with historucal vaccine coverage
    """
    if iso3 == "BGD":
        raw_data_double = get_bgd_vac_coverage(region="BGD", vaccine="total", dose=2)
        raw_data_booster = get_bgd_vac_coverage(region="BGD", vaccine="total", dose=3)
    elif iso3 == "PHL":
        raw_data_double = get_phl_vac_coverage(dose="SECOND_DOSE")
        raw_data_booster = get_phl_vac_coverage(dose="BOOSTER_DOSE") + get_phl_vac_coverage(dose="ADDITIONAL_DOSE")
    elif iso3 == "BTN":
        raw_data_double = get_btn_vac_coverage(region="Bhutan", dose=2)
        raw_data_booster = get_btn_vac_coverage(region="Bhutan", dose=3)
    elif iso3 == "VNM":
        if region == "Ho Chi Minh City":
            raw_data_double = pd.Series({619: 0.0909, 632: 0.1818, 654: 0.5, 710: 0.6,
                                         732: 0.6364, 746: 0.6591, 763: 0.6652, 791: 0.6671, 912: 0.7})
            raw_data_booster = pd.Series({619: 0.001, 632: 0.001, 654: 0.001, 710: 0.001,
                                          732: 0.1291, 746: 0.3482, 763: 0.4145, 791: 0.4309, 912: 0.5})
        elif region == "Hanoi":
            raw_data_double = pd.Series({822: 0.9, 884: 0.54})
            raw_data_booster = pd.Series({822: 0.045, 884: 0.045})

    # Add on the starting effective coverage value
    historical_vacc_data = {        
        'double': pd.concat(
            (
                pd.Series({model_start_time: start_immune_prop}),
                raw_data_double
            )
        ),
        'booster': pd.concat(
            (
                pd.Series({model_start_time: start_immune_prop * start_prop_high_among_immune}),
                raw_data_booster
            )
        )               
    }

    return historical_vacc_data


def get_immune_strata_distributions_from_fixed_increment(
    historical_vacc_data: pd.DataFrame, 
    extra_coverage: float, 
    model_end_time:int, 
    booster_effect_duration: float, 
    thinning: int
    ) -> pd.DataFrame:
    """
    Create a dataframe with the immunity strata distributions over time, accounting for future booster rates and waning booster immunity.
    This applies to the case where a fixed monthly coverage increment is specified for each age group.

    Args:
        historical_vacc_data: Vaccine coverage over time until latest available time
        extra_coverage: Extra coverage achieved per month in the future
        model_end_time: Model enf time
        booster_effect_duration: Average duration of immunity provided by a booster dose
        thinning: Thin out the empiric data to save time with curve fitting and because this must be >=2 (as below)

    Returns:
        Dataframe with the immunity strata distributions over time
    """
    latest_historical_time = max(historical_vacc_data["booster"].index)
    latest_booster_coverage = historical_vacc_data["booster"].loc[latest_historical_time]
    latest_double_coverage = historical_vacc_data["double"].loc[latest_historical_time]  

    # Create new dataframe    
    vacc_data = deepcopy(historical_vacc_data)

    new_time, new_booster_coverage = latest_historical_time, latest_booster_coverage
    while new_time < model_end_time: 
        new_time += 30
        new_booster_coverage = min(new_booster_coverage + extra_coverage, latest_double_coverage)            
        vacc_data["booster"].loc[new_time] = min(new_booster_coverage, 1.)
        # also extend double_vacc_data to keep the same format as booster_data
        vacc_data["double"].loc[new_time] = latest_double_coverage

    waned_booster_data = get_recently_vaccinated_prop(vacc_data["booster"], booster_effect_duration)

    # Create a single dataframe with the strata distributions over time
    strata_distributions_df = pd.DataFrame(
        {
            "none": 1. - vacc_data["double"],
            "low": vacc_data["double"] - waned_booster_data,
            "high": waned_booster_data
        },
    )

    # Apply thinning
    thinned_strata_distributions_df = strata_distributions_df[::thinning] if thinning else strata_distributions_df

    return thinned_strata_distributions_df


def get_immune_strata_distributions_using_priority(
    historical_vacc_data: pd.DataFrame, 
    future_monthly_booster_rate: float, 
    model_end_time: int, 
    booster_effect_duration: float, 
    thinning: int, 
    agegroup_population: float,
    allocated_doses: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Create a dataframe with the immunity strata distributions over time, accounting for future booster rates and waning booster immunity.
    This applies to the case where doses allocation is specified by order of priority for the different age-groups.

    Args:
        historical_vacc_data: Vaccine coverage over time until latest available time
        future_monthly_booster_rate: Number of doses available per month
        model_end_time: Model enf time
        booster_effect_duration: Average duration of immunity provided by a booster dose
        thinning: Thin out the empiric data to save time with curve fitting and because this must be >=2 (as below)
        agegroup_population: Population size of the relevant agegroup
        allocated_doses: Number of doses already allocated to higher-priority groups over time

    Returns:
        Dataframe with the immunity strata distributions over time
        Dictionary with the newly allocated booster doses over time
    """

    latest_historical_time = max(historical_vacc_data["booster"].index)
    latest_booster_coverage = historical_vacc_data["booster"].loc[latest_historical_time]
    latest_double_coverage = historical_vacc_data["double"].loc[latest_historical_time]  

    # Create new dataframe    
    vacc_data = deepcopy(historical_vacc_data)

    new_time, new_booster_coverage = latest_historical_time, latest_booster_coverage
    agegroup_allocated_doses = {}
    while new_time < model_end_time:
        new_time += 30

        # work out the new booster coverage accounting for already used doses
        non_boosted_pop = (latest_double_coverage - new_booster_coverage) * agegroup_population
        used_doses = allocated_doses[new_time] if new_time in allocated_doses else 0.
        administered_doses = min(future_monthly_booster_rate - used_doses, non_boosted_pop)
        new_booster_coverage = new_booster_coverage + administered_doses / agegroup_population            

        vacc_data["booster"].loc[new_time] = min(new_booster_coverage, 1.)
        # also extend double_vacc_data to keep the same format as booster_data
        vacc_data["double"].loc[new_time] = latest_double_coverage
        agegroup_allocated_doses[new_time] = administered_doses

    waned_booster_data = get_recently_vaccinated_prop(vacc_data["booster"], booster_effect_duration)

    # Create a single dataframe with the strata distributions over time
    strata_distributions_df = pd.DataFrame(
        {
            "none": 1. - vacc_data["double"],
            "low": vacc_data["double"] - waned_booster_data,
            "high": waned_booster_data
        },
    )

    # Apply thinning
    thinned_strata_distributions_df = strata_distributions_df[::thinning] if thinning else strata_distributions_df

    return thinned_strata_distributions_df, agegroup_allocated_doses


def add_dynamic_immunity_to_model(
        compartments: List[str],
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
        agegroup: Relevant agegroup for vaccination flow. 

    """

    sc_functions = calculate_transition_rates_from_dynamic_props(strata_distributions, ACTIVE_FLOWS)
    age_filter = {} if agegroup == "all_ages" else {"agegroup": agegroup}
    for comp in compartments:
        for transition, strata in ACTIVE_FLOWS.items():
            model.add_transition_flow(
                transition,
                sc_functions[transition],
                comp,
                comp,
                source_strata={"immunity": strata[0], **age_filter},
                dest_strata={"immunity": strata[1], **age_filter},
            )
