from types import MethodType
from typing import List

import numpy as np

from summer2 import CompartmentalModel, Stratification, Multiply

from autumn import inputs
from autumn.mixing.mixing import create_assortative_matrix
from autumn.region import Region

from apps.covid_19.constants import Compartment, COMPARTMENTS
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.preprocess.victorian_mixing import build_victorian_mixing_matrix_func
from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA

CLUSTER_STRATA = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]


def get_cluster_strat(params: Parameters) -> Stratification:
    cluster_strat = Stratification("cluster", CLUSTER_STRATA, COMPARTMENTS)
    country = params.country
    vic = params.victorian_clusters

    # Determine how to split up population by cluster
    # There is -0.5% to +4% difference per age group between sum of region population in 2018 and
    # total VIC population in 2020
    region_pops = {
        region: sum(
            inputs.get_population_by_agegroup(
                AGEGROUP_STRATA, country.iso3, region.upper(), year=2018
            )
        )
        for region in CLUSTER_STRATA
    }
    sum_region_props = sum(region_pops.values())
    cluster_split_props = {region: pop / sum_region_props for region, pop in region_pops.items()}
    cluster_strat.set_population_split(cluster_split_props)

    # Adjust contact rate multipliers
    contact_rate_adjustments = {}
    for cluster in Region.VICTORIA_SUBREGIONS:
        cluster_name = cluster.replace("-", "_")
        contact_rate_multiplier = getattr(vic, f"contact_rate_multiplier_{cluster_name}")
        contact_rate_adjustments[cluster_name] = Multiply(contact_rate_multiplier)

    # Add in flow adjustments per-region so we can calibrate the contact rate for each region.
    cluster_strat.add_flow_adjustments("infection", contact_rate_adjustments)

    # Use an identity mixing matrix to temporarily declare no inter-cluster mixing, which will then be over-written
    cluster_mixing_matrix = np.eye(len(CLUSTER_STRATA))
    cluster_strat.set_mixing_matrix(cluster_mixing_matrix)

    return cluster_strat


def apply_post_cluster_strat_hacks(params: Parameters, model: CompartmentalModel):
    regional_clusters = [region.replace("-", "_") for region in Region.VICTORIA_RURAL]
    vic = params.victorian_clusters
    country = params.country

    # A bit of a hack - to get rid of all the infectious populations from the regional clusters
    for i_comp, comp in enumerate(model.compartments):
        if any(
            [comp.has_stratum("cluster", cluster) for cluster in regional_clusters]
        ) and not comp.has_name(Compartment.SUSCEPTIBLE):
            model.initial_population[i_comp] = 0.0

    """
    Hack in a custom (144x144) mixing matrix where each region is adjusted individually
    based on its time variant mobility data.
    """

    # Get the inter-cluster mixing matrix
    intercluster_mixing_matrix = create_assortative_matrix(vic.intercluster_mixing, CLUSTER_STRATA)

    # Replace regional Victoria maximum effect calibration parameters with the metro values for consistency
    for microdist_process in ["face_coverings", "behaviour"]:
        vic.regional.mobility.microdistancing[f"{microdist_process}_adjuster"].parameters.effect = \
            vic.metro.mobility.microdistancing[f"{microdist_process}_adjuster"].parameters.effect

    # Get new mixing matrix
    static_mixing_matrix = inputs.get_country_mixing_matrix("all_locations", country.iso3)
    get_mixing_matrix = build_victorian_mixing_matrix_func(
        static_mixing_matrix,
        vic.metro.mobility,
        vic.regional.mobility,
        country,
        intercluster_mixing_matrix,
    )
    setattr(model, "_get_mixing_matrix", MethodType(get_mixing_matrix, model))
