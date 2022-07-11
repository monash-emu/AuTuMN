from typing import List
from summer import Stratification, Multiply, CompartmentalModel
import pandas as pd
from numpy import log

from autumn.core.inputs.database import get_input_db
from autumn.models.sm_covid.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName
from autumn.settings.constants import COVID_BASE_DATETIME

def adjust_susceptible_infection_without_strains(
        immune_effect: float,        
        immunity_strat: Stratification,
):
    """
    Apply the modification to the immunity stratification to account for immunity to first infection (from the
    susceptible compartment), i.e. vaccine-induced immunity (or for some models this stratification could be taken
    to represent past infection prior to the beginning of the simulation period).

    Args:
        immune_effect: The protection from vaccination
        immunity_strat: The immunity stratification, to be modified

    """

    infection_adjustments = {
        ImmunityStratum.UNVACCINATED: None,
        ImmunityStratum.VACCINATED: Multiply(1.0 - immune_effect),
    }

    immunity_strat.set_flow_adjustments(
        FlowName.INFECTION,
        infection_adjustments,
    )


def get_immunity_strat(
        compartments: List[str],
) -> Stratification:
    """
    Args:
        compartments: Unstratified model compartment types being implemented

    Returns:
        The summer Stratification object that captures vaccine-related immunity

    """

    # Create the immunity stratification, which applies to all compartments
    immunity_strat = Stratification("immunity", IMMUNITY_STRATA, compartments)

    # Set distribution of starting population (all unvaccinated)
    immunity_split_props = {
        ImmunityStratum.UNVACCINATED: 1.0,
        ImmunityStratum.VACCINATED: 0.
    }
    immunity_strat.set_population_split(immunity_split_props)

    return immunity_strat


def get_time_variant_vaccination_rate(iso3: str):
    """
    Create a time-variant function returning the vaccination rates matching coverage data over time

    Args:
        iso3: Country ISO3 code

    Returns:
        A function of time returning the instantaneous vaccnation rate
    """
    # Load vaccine coverage data
    input_db = get_input_db()
    vacc_data = input_db.query(table_name='owid', conditions= {"iso_code": iso3}, columns=["date", "people_fully_vaccinated_per_hundred"])
    vacc_data.dropna(inplace=True)
    vacc_data["date_int"] = (pd.to_datetime(vacc_data.date)- COVID_BASE_DATETIME).dt.days

    """
     Determine time-variant transition rate 
    """
    # For each time interval [t0, t1], the vaccination rate alpha verifies: (1 - V0) * exp(-alpha * delta_t) = 1 - V1
    # Then alpha = (log(1 - V0) - log(1 - V1)) / delta_t
    vacc_data["log_unvacc_p"] = log(1. - vacc_data["people_fully_vaccinated_per_hundred"] / 100.) 
    vacc_data["vacc_rate"] = - vacc_data["log_unvacc_p"].diff(periods=-1) / vacc_data["date_int"].diff(periods=-1)

    # Create a continuous scale-up function
    t_min, t_max = vacc_data["date_int"].iloc[0], vacc_data["date_int"].iloc[-1] 
    def tv_vacc_rate(t, computed_values=None):
        if t < t_min or t >= t_max:
            return 0.
        else:
            return vacc_data[vacc_data["date_int"] <= t].iloc[-1]["vacc_rate"]

    return tv_vacc_rate


def set_dynamic_vaccination_flows(
    compartments: List[str], 
    model: CompartmentalModel, 
    iso3: str, 
    age_groups: List[str],
):
    """
    Add vaccination flows based on coverage data from Our World in Data 

    Args:
        compartments: The list of model compartment types
        model: The model object
        iso3: Country ISO3 code
        age_groups: List of model age groups
    """
    # Get vaccination coverage data and determine time-variant transition rate 
    tv_vacc_rate = get_time_variant_vaccination_rate(iso3)

    # Request transition flows
    for compartment in compartments:
        model.add_transition_flow(
            name=FlowName.VACCINATION, 
            fractional_rate=tv_vacc_rate, 
            source=compartment, 
            dest=compartment, 
            source_strata={"immunity": ImmunityStratum.UNVACCINATED}, 
            dest_strata={"immunity": ImmunityStratum.VACCINATED}, 
            expected_flow_count=len(age_groups)
        )
