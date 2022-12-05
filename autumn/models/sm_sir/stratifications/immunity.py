from typing import Optional, List, Dict
import pandas as pd

from summer import Stratification, Multiply
from summer import CompartmentalModel

from autumn.core.utils.pandas import increment_last_period
from autumn.core.inputs.covid_phl.queries import get_phl_vac_coverage
from autumn.core.inputs.covid_au.queries import get_nt_vac_coverage
from autumn.core.inputs.covid_mys.queries import get_mys_vac_coverage
from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName, IMMUNITY_STRATA_WPRO,\
    ImmunityStratumWPRO
from autumn.models.sm_sir.parameters import ImmunityStratification, VocComponent, TimeSeries
from autumn.model_features.solve_transitions import calculate_transition_rates_from_dynamic_props
from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.core.inputs.database import get_input_db
from numpy import log

SATURATION_COVERAGE = .80

from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName
from autumn.models.sm_sir.parameters import ImmunityStratification, VocComponent, Vaccination
from autumn.model_features.solve_transitions import calculate_transition_rates_from_dynamic_props
from autumn.model_features.outputs import get_strata

ACTIVE_FLOWS = {
    "vaccination": ("none", "low"),
    "boosting": ("low", "high"),
    "waning": ("high", "low"),
    "complete_waning": ("low", "none"),
}


def set_dynamic_vaccination_flows(
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


def adjust_susceptible_infection_without_strains(
        vaccine_model: str,
        immunity_strat: Stratification,
        immune_effect=0.0,
        low_immune_effect=0.0,
        high_immune_effect=0.0

):
    """
    Apply the modification to the immunity stratification to account for immunity to first infection (from the
    susceptible compartment), i.e. vaccine-induced immunity (or for some models this stratification could be taken
    to represent past infection prior to the beginning of the simulation period).
    Args:
        low_immune_effect: The protection from low immunity
        high_immune_effect: The protection from high immunity
        immunity_strat: The immunity stratification, to be modified
        immune_effect: The protection from vaccination
        vaccine_model: to differentiate of it is sm_sir default vaccination or for WPRO model unvaccinated and vaccinated classification
    """
    if vaccine_model == "WPRO":
        infection_adjustments = {
            ImmunityStratumWPRO.UNVACCINATED: None,
            ImmunityStratumWPRO.VACCINATED: Multiply(1.0 - immune_effect),
        }
    else:

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
        immunity_strat: Stratification,
        voc_params: Optional[Dict[str, VocComponent]],
        vaccine_model: str,
        immune_effect=0.0,
        low_immune_effect=0.0,
        high_immune_effect=0.0,
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
        immune_effect: The protection from immunity (associated with WPRO model)
        vaccine_model: to differentiate of it is sm_sir default vaccination or for WPRO model unvaccinated and vaccinated classification

    """

    for infecting_strain in voc_params:
        
        # The vaccination-specific immunity that will be retained after allowing for the strain's immune escape against vaccination-induced immunity
        non_cross_effect = 1.0 - voc_params[infecting_strain].immune_escape

        if vaccine_model == "WPRO":
            # Adjust the rate of infection considering the protection of that immunity status (incorporating the strain's escape properties)
            non_cross_multiplier = 1.0 - immune_effect * non_cross_effect

            infection_adjustments = {
                ImmunityStratumWPRO.UNVACCINATED: None,
                ImmunityStratumWPRO.VACCINATED: Multiply(non_cross_multiplier),
            }

        else:
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
        immunity_strat: Stratification,
        reinfection_flows: List[str],
        voc_params: Optional[Dict[str, VocComponent]],
        vaccine_model: str,
        immune_effect=0.0,
        low_immune_effect=0.0,
        high_immune_effect=0.0,

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
        immune_effect: The infection protection from vaccine immunity associated witht he WPRO model
        vaccine_model: to differentiate of it is sm_sir default vaccination or for WPRO model unvaccinated and vaccinated classification

    """

    for infecting_strain in voc_params:

        # The vaccination-specific immunity that will be retained after allowing for the strain's immune escape against vaccination-induced immunity
        non_cross_effect = 1.0 - voc_params[infecting_strain].immune_escape

        # Adjust the rate of infection considering the protection of that immunity status (incorporating the strain's escape properties)
        low_non_cross_multiplier = 1.0 - low_immune_effect * non_cross_effect
        high_non_cross_multiplier = 1.0 - high_immune_effect * non_cross_effect

        non_cross_multiplier = 1.0 - immune_effect * non_cross_effect # for the WPRO model

        # Considering people recovered from infection with each modelled strain
        for infected_strain in voc_params:

            # For both the early and late reinfection transitions
            for flow in reinfection_flows:

                # Cross protection from previous infection with the "infected" strain against the "infecting" strain
                cross_effect = 1.0 - getattr(voc_params[infected_strain].cross_protection[infecting_strain], flow)

                if vaccine_model == "WPRO":
                    # Combine the two mechanisms of protection
                    reinfection_adjustments = {
                        ImmunityStratumWPRO.UNVACCINATED: Multiply(cross_effect),
                        ImmunityStratumWPRO.VACCINATED: Multiply(non_cross_multiplier * cross_effect),
                    }
                else:
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


def get_immunity_strat_wpro(
        compartments: List[str],
) -> Stratification:
    """
    Args:
        compartments: Unstratified model compartment types being implemented

    Returns:
        The summer Stratification object that captures vaccine-related immunity

    """

    # Create the immunity stratification, which applies to all compartments
    immunity_strat = Stratification("immunity", IMMUNITY_STRATA_WPRO, compartments)

    # Set distribution of starting population (all unvaccinated)
    immunity_split_props = {
        ImmunityStratumWPRO.UNVACCINATED: 1.0,
        ImmunityStratumWPRO.VACCINATED: 0.
    }
    immunity_strat.set_population_split(immunity_split_props)

    return immunity_strat


def get_reported_vacc_coverage(iso3, start_age, end_age, age_specific_vacc):

    # Get the raw data from the loading functions and drop rows with any nans
    if iso3 == "PHL":
        full_series = get_phl_vac_coverage(dose="SECOND_DOSE")
        booster_series = get_phl_vac_coverage(dose="BOOSTER_DOSE")
        assert not age_specific_vacc, "Philippines data not age-specific, so just replicating calculations"
    elif iso3 == "AUS":
        full_series = get_nt_vac_coverage(dose=2)
        booster_series = get_nt_vac_coverage(dose=3)
    elif iso3 == "MYS":
        full_series = get_mys_vac_coverage(dose="full")
        booster_series = get_mys_vac_coverage(dose="booster")
    else:
        raise ValueError("Data for country not available (in this function)")
    
    vaccine_data = pd.DataFrame(
        {
            "full": full_series,
            "boost": booster_series,
        }
    ).dropna(axis=0)

    return vaccine_data


def add_user_request_to_vacc(
    extra_coverage: Dict[str, dict], 
    age_cat: str,
    vaccine_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Add on any custom user requests to the vaccination data dataframe.

    Args:
        extra_coverage: The user request for extra coverage values
        age_cat: The age group being considered (or 'all_ages')
        vaccine_data: The partially processed empiric vaccination data
    Returns:
        The dataframe with the user requests included
    """
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
    
    return vaccine_data


def get_strata_values_from_vacc_data(
    boosting: bool,
    booster_waning: bool,
    vaccine_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Format the data to match the model's immunity structure.

    Args:
        boosting: Whether there is a boosted stratum modelled
        booster_waning: Whether the the boosted stratum only considers recently boosted
        vaccine_data: The fully processed vaccine data dataframe
    """

    vaccine_data["never"] = 1. - vaccine_data["full"]
    if boosting and booster_waning:
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
    
    return strata_data


def apply_vacc_coverage(
        model: CompartmentalModel,
        iso3: str,
        start_immune_prop: float,
        start_prop_high_among_immune: float,
        vacc_params: Vaccination,
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

    age_specific_vacc = vacc_params.age_specific_vacc
    booster_effect_duration = vacc_params.booster_effect_duration
    extra_coverage = vacc_params.extra_vacc_coverage
    boosting = vacc_params.boosting

    age_vacc_categories = get_strata(model, "agegroup") if age_specific_vacc else ["all_ages"]

    msg = "Age group in requests not present in model (or 'all_ages' if vaccination not age-specific)"
    assert all([i in age_vacc_categories for i in extra_coverage.keys()]), msg

    for i_age, age_cat in enumerate(age_vacc_categories):

        start_age = int(age_vacc_categories[i_age]) if age_cat != age_vacc_categories[0] else None
        end_age = int(age_vacc_categories[i_age + 1]) if age_cat != age_vacc_categories[-1] else None

        # Get the data
        vaccine_data = get_reported_vacc_coverage(
            iso3, 
            start_age, 
            end_age, 
            age_specific_vacc,
        )

        # Thin as per user request
        vaccine_data = vaccine_data[::vacc_params.data_thinning]

        # Get rid of any data that is from before the model starts running
        model_start_time = model.times[0]
        vaccine_data = vaccine_data[model_start_time < vaccine_data.index]

        # Add on the user requested starting proportions
        starting_values = {
            "full": start_immune_prop, 
            "boost": start_immune_prop * start_prop_high_among_immune,
        }
        vaccine_data.loc[model_start_time] = starting_values
        vaccine_data.loc[-1000] = starting_values  # To prevent issues with interpolation later

        # Sort
        vaccine_data.sort_index(inplace=True)
        
        # Add on any custom user requests
        vaccine_data = add_user_request_to_vacc(
            extra_coverage, 
            age_cat, 
            vaccine_data
        )
        
        # Add a column for the proportion of the population recently vaccinated
        if booster_effect_duration and not vaccine_data.empty:
            vaccine_data["recent_boost"] = increment_last_period(
                booster_effect_duration,
                vaccine_data["boost"]
            )

        # Get the actual values for the model strata
        strata_data = get_strata_values_from_vacc_data(
            boosting, 
            bool(booster_effect_duration),
            vaccine_data,
        )

        # Apply to model
        add_dynamic_immunity_to_model(
            strata_data,
            model, 
            age_cat,
        )


def get_time_variant_vaccination_rates(iso3: str, age_pops: pd.Series):
    """
    Create a time-variant function returning the vaccination rates matching coverage data over time

    Args:
        iso3: Country ISO3 code
        age_pops: Population size of the modelled age groups

    Returns:
        A function of time returning the instantaneous vaccnation rate
    """
    total_pop = age_pops.sum()

    # Load vaccine coverage data and prepare pandas dataframe for calculations
    input_db = get_input_db()
    vacc_data = input_db.query(table_name='owid', conditions= {"iso_code": iso3}, columns=["date", "people_fully_vaccinated_per_hundred"])
    vacc_data.dropna(inplace=True)
    vacc_data["date_int"] = (pd.to_datetime(vacc_data.date)- COVID_BASE_DATETIME).dt.days
    vacc_data["n_doses"] = total_pop * vacc_data["people_fully_vaccinated_per_hundred"] / 100.
    vacc_data.drop(["people_fully_vaccinated_per_hundred", "date"], axis=1)
    t_min, t_max = vacc_data["date_int"].iloc[0], vacc_data["date_int"].iloc[-1]

    """
     Determine age-specific time-variant transition rates 
    """
    # Work out maximum vaccination coverage by age
    max_data_coverage = vacc_data["n_doses"].max() / total_pop
    saturation_coverage = max(SATURATION_COVERAGE, max_data_coverage)
    saturation_doses = {agegroup: popsize * saturation_coverage  for agegroup, popsize in age_pops.items()}

    # Work out the minimum number of doses required before each agegroup becomes eligible for vaccination
    trigger_doses = {agegroup: sum([saturation_doses[a] for a in age_pops.index if int(a) > int(agegroup)]) for agegroup in age_pops.index}

    # Create time-variant vaccination rate functions for each age group
    tv_vacc_rate_funcs = {}
    for agegroup, this_sat_doses in saturation_doses.items():
        # Work out age-specific coverage over time
        vacc_data[f"coverage_agegroup_{agegroup}"] = (vacc_data['n_doses'] - trigger_doses[agegroup]).clip(lower=0.).clip(upper=this_sat_doses) / age_pops.loc[agegroup]

        # For each time interval [t0, t1], the vaccination rate alpha verifies: (1 - V0) * exp(-alpha * delta_t) = 1 - V1
        # Then alpha = (log(1 - V0) - log(1 - V1)) / delta_t
        vacc_data[f"log_unvacc_p_agegroup_{agegroup}"] = log(1. - vacc_data[f"coverage_agegroup_{agegroup}"])
        vacc_data[f"vacc_rate_agegroup_{agegroup}"] = - vacc_data[f"log_unvacc_p_agegroup_{agegroup}"].diff(periods=-1) / vacc_data["date_int"].diff(periods=-1)

        # Create a continuous scale-up function
        tv_vacc_rate_funcs[agegroup] = make_tv_vacc_rate_func(agegroup, t_min, t_max, vacc_data)

    return tv_vacc_rate_funcs


def make_tv_vacc_rate_func(agegroup: str, t_min: int, t_max: int, vacc_data: pd.DataFrame):
    """
    Simple factory function
    """
    def tv_vacc_rate(t, computed_values=None):
        if t < t_min or t >= t_max:
            return 0.
        else:
            return vacc_data[vacc_data["date_int"] <= t].iloc[-1][f"vacc_rate_agegroup_{agegroup}"]

    return tv_vacc_rate


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


def set_dynamic_vaccination_flows_wpro(
    compartments: List[str],
    model: CompartmentalModel,
    iso3: str,
    age_pops: pd.Series,
):
    """
    Add vaccination flows based on coverage data from Our World in Data

    Args:
        compartments: The list of model compartment types
        model: The model object
        iso3: Country ISO3 code
        age_pops: Population size of modelled age groups
    """
    # Get vaccination coverage data and determine time-variant transition rate
    tv_vacc_rate_funcs = get_time_variant_vaccination_rates(iso3, age_pops)
    # Request transition flows
    for agegroup, tv_vacc_rate in tv_vacc_rate_funcs.items():

        for compartment in compartments:
            model.add_transition_flow(
                name=FlowName.VACCINATION,
                fractional_rate=tv_vacc_rate,
                source=compartment,
                dest=compartment,
                source_strata={"immunity": ImmunityStratumWPRO.UNVACCINATED, "agegroup": agegroup},
                dest_strata={"immunity": ImmunityStratumWPRO.VACCINATED, "agegroup": agegroup},
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
