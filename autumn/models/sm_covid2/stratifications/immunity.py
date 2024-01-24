from typing import List
from summer2 import Stratification, Multiply, CompartmentalModel
import pandas as pd
from numpy import log

from jax import numpy as jnp
from summer2.parameters import Function, Data, Time

from summer2.functions.util import piecewise_constant

from autumn.models.sm_covid2.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName
from autumn.settings.constants import COVID_BASE_DATETIME

from autumn.projects.sm_covid2.common_school.utils import get_owid_data

SATURATION_COVERAGE = 0.80


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
        The summer2 Stratification object that captures vaccine-related immunity

    """

    # Create the immunity stratification, which applies to all compartments
    immunity_strat = Stratification("immunity", IMMUNITY_STRATA, compartments)

    # Set distribution of starting population (all unvaccinated)
    immunity_split_props = {ImmunityStratum.UNVACCINATED: 1.0, ImmunityStratum.VACCINATED: 0.0}
    immunity_strat.set_population_split(immunity_split_props)

    return immunity_strat

def get_vacc_data(iso3: str):
    vacc_data = get_owid_data(columns=["date", "iso_code", "people_fully_vaccinated_per_hundred"], iso_code=iso3)

    return vacc_data.dropna()

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
    vacc_data = get_vacc_data(iso3)

    effect_delay = 14
    vacc_data["date_int"] = (pd.to_datetime(vacc_data.date) - COVID_BASE_DATETIME).dt.days + effect_delay
    vacc_data["n_doses"] = total_pop * vacc_data["people_fully_vaccinated_per_hundred"] / 100.0
    vacc_data.drop(["people_fully_vaccinated_per_hundred", "date"], axis=1)
    t_min, t_max = vacc_data["date_int"].iloc[0], vacc_data["date_int"].iloc[-1]

    """
     Determine age-specific time-variant transition rates 
    """
    # Work out maximum vaccination coverage by age
    max_data_coverage = vacc_data["n_doses"].max() / total_pop
    saturation_coverage = max(SATURATION_COVERAGE, max_data_coverage)
    saturation_doses = {
        agegroup: popsize * saturation_coverage for agegroup, popsize in age_pops.items()
    }

    # Work out the minimum number of doses required before each agegroup becomes eligible for vaccination
    trigger_doses = {
        agegroup: sum([saturation_doses[a] for a in age_pops.index if int(a) > int(agegroup)])
        for agegroup in age_pops.index
    }

    # Create time-variant vaccination rate functions for each age group
    tv_vacc_rate_funcs = {}
    for agegroup, this_sat_doses in saturation_doses.items():
        # Work out age-specific coverage over time
        vacc_data[f"coverage_agegroup_{agegroup}"] = (
            vacc_data["n_doses"] - trigger_doses[agegroup]
        ).clip(lower=0.0).clip(upper=this_sat_doses) / age_pops.loc[agegroup]

        # For each time interval [t0, t1], the vaccination rate alpha verifies: (1 - V0) * exp(-alpha * delta_t) = 1 - V1
        # Then alpha = (log(1 - V0) - log(1 - V1)) / delta_t
        vacc_data[f"log_unvacc_p_agegroup_{agegroup}"] = log(
            1.0 - vacc_data[f"coverage_agegroup_{agegroup}"]
        )
        vacc_data[f"vacc_rate_agegroup_{agegroup}"] = -vacc_data[
            f"log_unvacc_p_agegroup_{agegroup}"
        ].diff(periods=-1) / vacc_data["date_int"].diff(periods=-1)

        # Create a continuous scale-up function
        tv_vacc_rate_funcs[agegroup] = make_tv_vacc_rate_func(agegroup, t_min, t_max, vacc_data)

    return tv_vacc_rate_funcs


def make_tv_vacc_rate_func(agegroup: str, t_min: int, t_max: int, vacc_data: pd.DataFrame):
    """
    Simple factory function
    """

    def tv_vacc_rate(t, computed_values=None):
        if t < t_min or t >= t_max:
            return 0.0
        else:
            return vacc_data[vacc_data["date_int"] <= t].iloc[-1][f"vacc_rate_agegroup_{agegroup}"]

    agegroup_data = vacc_data[f"vacc_rate_agegroup_{agegroup}"]

    bp = Data(jnp.array([*vacc_data["date_int"]]))
    vals = Data(jnp.array((0.0, *agegroup_data.iloc[:-1], 0.0)))

    return Function(piecewise_constant, [Time, bp, vals])


def set_dynamic_vaccination_flows(
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
                source_strata={"immunity": ImmunityStratum.UNVACCINATED, "agegroup": agegroup},
                dest_strata={"immunity": ImmunityStratum.VACCINATED, "agegroup": agegroup},
            )
