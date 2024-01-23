import pandas as pd
from datetime import datetime
from summer2 import CompartmentalModel, Stratification, Multiply, StrainStratification, Overwrite
from summer2.parameters import Function, Time
from typing import Dict, Union, List
from jax import numpy as jnp

from autumn.core.inputs.database import get_input_db
from autumn.models.sm_covid2.parameters import VocComponent
from autumn.models.sm_covid2.constants import FlowName, Compartment, ImmunityStratum
from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.model_features.strains import broadcast_infection_flows_over_source

from autumn.core.inputs.demography.queries import get_gisaid_country_name_from_iso3

from autumn.settings import INPUT_DATA_PATH
from pathlib import Path
prepared_gisaid_csv_path = Path(INPUT_DATA_PATH) / "gisaid-voc/gisaid_variants_statistics_prepared.csv"


def voc_seed_func(time, entry_rate, start_time, seed_duration):
    offset = time - start_time
    return jnp.where(offset > 0, jnp.where(offset < seed_duration, entry_rate, 0.0), 0.0)


def make_voc_seed_func(entry_rate: float, start_time: float, seed_duration: float):
    """
    Create a simple step function to allow seeding of the VoC strain for the period from the requested time point.
    The function starts from zero, steps up to entry_rate from the start_time and then steps back down again to zero after seed_duration has elapsed.

    Args:
        entry_rate: The entry rate
        start_time: The requested time at which seeding should start
        seed_duration: The number of days that the seeding should go for

    Returns:
        The simple step function

    """
    return Function(voc_seed_func, [Time, entry_rate, start_time, seed_duration])


def get_strain_strat(
    voc_params: Dict[str, VocComponent],
    compartments: List[str],
    latent_progression_flows: List[str],
):
    """
    Stratify the model by strain, with at least two strata, being wild or "ancestral" virus type and the variants of
    concern ("VoC").

    We are now stratifying all the compartments, including the recovered ones. The recovered compartment stratified by
    strain represents people whose last infection was with that strain.

    Args:
        voc_params: All the VoC parameters (one VocComponent parameters object for each VoC)
        compartments: All the model's unstratified compartment types
        latent_progression_flows: List of flows related to latency progression

    Returns:
        The strain stratification summer object

    """

    # Process the requests
    strains = list(voc_params.keys())
    affected_compartments = [comp for comp in compartments if comp != Compartment.SUSCEPTIBLE]

    # Create the stratification object
    strain_strat = StrainStratification("strain", strains, affected_compartments)

    # Assign the starting population
    population_split = {strain: voc_params[strain].seed_prop for strain in strains}
    strain_strat.set_population_split(population_split)

    # Adjust the contact rate
    transmissibility_adjustment = {
        strain: Multiply(voc_params[strain].contact_rate_multiplier) for strain in strains
    }
    strain_strat.set_flow_adjustments(FlowName.INFECTION, transmissibility_adjustment)

    # Adjust latency transition
    adjustments_latent = {strain: None for strain in strains}
    n_latent_transitions = len(latent_progression_flows)
    for strain in strains:
        if voc_params[strain].incubation_overwrite_value:
            progression_rate = (
                n_latent_transitions * 1.0 / voc_params[strain].incubation_overwrite_value
            )
            adjustments_latent.update({strain: Overwrite(progression_rate)})

    for flowname in latent_progression_flows:
        strain_strat.set_flow_adjustments(flowname, adjustments_latent)

    return strain_strat


def get_first_variant_report_date(variant: str, iso3: str, perc_threshold: float = 1.):
   
    colname = {
        "delta": "Delta Count",
        "omicron": "Omicron Count"
    }[variant]

    variants_global_emergence_date = {
        "delta": datetime(2020, 10, 1),  # October 2020 according to WHO
        "omicron": datetime(2021, 11, 1),  # November 2021 according to WHO
    }

    gisaid_data = pd.read_csv(prepared_gisaid_csv_path)
    gisaid_data = gisaid_data[(gisaid_data[colname].isnull() == False) & (gisaid_data['Country'] == get_gisaid_country_name_from_iso3(iso3))]

    gisaid_data['Date'] = pd.to_datetime(gisaid_data['Date'])
    gisaid_data['Percentage'] = 100. * gisaid_data[colname] / gisaid_data['Total']

    report_dates = gisaid_data[(gisaid_data['Percentage'] >= perc_threshold) & (gisaid_data['Total'] > 10) & (gisaid_data[colname] > 1)]['Date']
    report_dates = report_dates[report_dates >= variants_global_emergence_date[variant]]

    return report_dates.min()


def seed_vocs_using_gisaid(
    model: CompartmentalModel,
    all_voc_params: Dict[str, VocComponent],
    seed_compartment: str,
    iso3: str,
    infectious_seed: float,
):
    """
    Use importation flows to seed VoC cases.

    Generally seeding to the infectious compartment, because unlike Covid model, this compartment always present.

    Note that the entry rate will get repeated for each compartment as the requested compartments for entry are
    progressively stratified after this process is applied (but are split over the previous stratifications of the
    compartment to which this is applied, because the split_imports argument is True).

    Args:
        model: The summer model object
        all_voc_params: The VoC-related parameters
        seed_compartment: The compartment that VoCs should be seeded to
        iso3: The modelled country's iso3
        infectious_seed: The total size of the infectious seed
    """

    for voc_name, this_voc_params in all_voc_params.items():
        voc_seed_params = this_voc_params.new_voc_seed
        if voc_seed_params:
            # work out seed time using gisaid data
            first_report_date = get_first_variant_report_date(voc_name, iso3)
            first_report_date_as_int = (first_report_date - COVID_BASE_DATETIME).days
            seed_time = first_report_date_as_int + voc_seed_params.time_from_gisaid_report

            entry_rate = infectious_seed / voc_seed_params.seed_duration
            voc_seed_func = make_voc_seed_func(entry_rate, seed_time, voc_seed_params.seed_duration)
            model.add_importation_flow(
                f"seed_voc_{voc_name}",
                voc_seed_func,
                dest=seed_compartment,
                dest_strata={"strain": voc_name},
                split_imports=True,
            )


def apply_reinfection_flows_with_strains(
    model: CompartmentalModel,
    base_compartments: List[str],
    infection_dest: str,
    age_groups: List[str],
    voc_params: Dict[str, VocComponent],
    strain_strata: List[str],
    contact_rate: Union[float, callable],
    suscept_adjs: pd.Series,
):
    """
    Apply the reinfection flows, making sure that it is possible to be infected with any strain after infection with any strain.
    We'll work out whether this occurs at a reduced rate because of immunity later, in the various functions of the immunity.py file.

    Args:
        model: The SM-SIR model being adapted
        base_compartments: The unstratified model compartments
        infection_dest: Where people end up first after having been infected
        age_groups: The modelled age groups
        voc_params: The VoC-related parameters
        strain_strata: The strains being implemented or a list of an empty string if no strains in the model
        contact_rate: The model's contact rate
        suscept_adjs: Adjustments to the rate of infection of susceptibles based on modelled age groups

    """

    # Loop over all infecting strains
    for dest_strain in strain_strata:
        dest_filter = {"strain": dest_strain}

        # Adjust for infectiousness of infecting strain
        strain_adjuster = voc_params[dest_strain].contact_rate_multiplier

        # Loop over all age groups
        for age_group in age_groups:
            age_filter = {"agegroup": age_group}
            dest_filter.update(age_filter)
            source_filter = age_filter

            # Get an adjuster that considers both the relative infectiousness of the strain and the relative susceptibility of the age group
            contact_rate_adjuster = strain_adjuster * suscept_adjs[age_group]
            strain_age_contact_rate = contact_rate * contact_rate_adjuster

            # Need to broadcast the flows over the recovered status for the strains
            broadcast_infection_flows_over_source(
                model,
                FlowName.REINFECTION,
                Compartment.RECOVERED,
                infection_dest,
                source_filter,
                dest_filter,
                strain_age_contact_rate,
                exp_flows=1,
            )


def adjust_susceptible_infection_with_strains(
    immune_effect: float,
    immunity_strat: Stratification,
    voc_params: Dict[str, VocComponent],
):
    """
    Apply the modification to the immunity stratification to account for immunity to first infection (from the
    susceptible compartment), accounting for the extent to which each VoC is immune-escape to vaccine-induced immunity.
    This same function can be applied to the model wherever VoCs are included, regardless of strain structure,
    because the strain stratification (representing history of last infecting strain) does not apply here.

    Args:
        immune_effect: The protection from immunity
        immunity_strat: The immunity stratification, to be modified
        voc_params: The parameters relating to the VoCs being implemented

    """

    for infecting_strain in voc_params:

        # The vaccination-specific immunity that will be retained after allowing for the strain's immune escape against vaccination-induced immunity
        non_cross_effect = 1.0 - voc_params[infecting_strain].vacc_immune_escape

        # Adjust the rate of infection considering the protection of that immunity status (incorporating the strain's escape properties)
        non_cross_multiplier = 1.0 - immune_effect * non_cross_effect

        infection_adjustments = {
            ImmunityStratum.UNVACCINATED: None,
            ImmunityStratum.VACCINATED: Multiply(non_cross_multiplier),
        }

        immunity_strat.set_flow_adjustments(
            FlowName.INFECTION,
            infection_adjustments,
            dest_strata={"strain": infecting_strain},
        )


def adjust_reinfection_with_strains(
    immune_effect: float,
    immunity_strat: Stratification,
    voc_params: Dict[str, VocComponent],
):
    """
    Adjust the rate of reinfection for immunity, in cases in which we do need to worry about cross-strain immunity, so
    we have to consider every possible combination of cross-immunity between strains (including the immunity conferred
    by infection with a certain strain and reinfection with that same strain).

    Args:
        immune_effect: The infection protection from vaccine immunity
        immunity_strat: The immunity stratification, to be modified
        voc_params: The parameters relating to the VoCs being implemented

    """

    for infecting_strain in voc_params:

        # The vaccination-specific immunity that will be retained after allowing for the strain's immune escape against vaccination-induced immunity
        non_cross_effect = 1.0 - voc_params[infecting_strain].vacc_immune_escape

        # Adjust the rate of infection considering the protection of that immunity status (incorporating the strain's escape properties)
        non_cross_multiplier = 1.0 - immune_effect * non_cross_effect

        # Considering people recovered from infection with each modelled strain
        for infected_strain in voc_params:

            # Cross protection from previous infection with the "infected" strain against the "infecting" strain
            cross_effect_multiplier = (
                1.0 - voc_params[infected_strain].cross_protection[infecting_strain]
            )

            # Combine the two mechanisms of protection
            reinfection_adjustments = {
                ImmunityStratum.UNVACCINATED: Multiply(cross_effect_multiplier),
                ImmunityStratum.VACCINATED: Multiply(
                    non_cross_multiplier * cross_effect_multiplier
                ),
            }

            immunity_strat.set_flow_adjustments(
                FlowName.REINFECTION,
                reinfection_adjustments,
                source_strata={"strain": infected_strain},
                dest_strata={"strain": infecting_strain},
            )
