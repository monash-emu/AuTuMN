import numpy as np
from summer import CompartmentalModel, Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, Compartment
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.preprocess.victorian_mixing import (
    build_victorian_mixing_matrix_func,
)
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.tools import inputs
from autumn.settings import Region

CLUSTER_STRATA = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]

cluster_contact_groups = {
    "south_and_east": [Region.SOUTH_EAST_METRO, Region.SOUTH_METRO],
    "regional": [i_region for i_region in Region.VICTORIA_RURAL if i_region != Region.BARWON_SOUTH_WEST]
}


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
        if cluster in cluster_contact_groups["south_and_east"]:
            multiplier_name = f"contact_rate_multiplier_{Region.SOUTH_METRO.replace('-', '_')}"
        elif cluster in cluster_contact_groups["regional"]:
            multiplier_name = "contact_rate_multiplier_regional"
        else:
            multiplier_name = f"contact_rate_multiplier_{cluster_name}"
        contact_rate_multiplier = getattr(vic, multiplier_name)
        contact_rate_adjustments[cluster_name] = Multiply(contact_rate_multiplier)

    # for cluster in Region.VICTORIA_METRO + [Region.BARWON_SOUTH_WEST]:
    #     cluster_name = cluster.replace("-", "_")
    #     contact_rate_multiplier = getattr(vic, f"contact_rate_multiplier_{cluster_name}")
    #     contact_rate_adjustments[cluster_name] = Multiply(contact_rate_multiplier)
    # for cluster in Region.VICTORIA_RURAL:
    #     if cluster != Region.BARWON_SOUTH_WEST:
    #         cluster_name = cluster.replace("-", "_")
    #         contact_rate_multiplier = getattr(vic, "contact_rate_multiplier_regional")
    #         contact_rate_adjustments[cluster_name] = Multiply(contact_rate_multiplier)

    # Add in flow adjustments per-region so we can calibrate the contact rate for each region.
    cluster_strat.add_flow_adjustments("infection", contact_rate_adjustments)

    # Use an identity mixing matrix to temporarily declare no inter-cluster mixing, which will then be over-written
    cluster_mixing_matrix = np.eye(len(CLUSTER_STRATA))
    cluster_strat.set_mixing_matrix(cluster_mixing_matrix)

    return cluster_strat


def apply_post_cluster_strat_hacks(params: Parameters, model: CompartmentalModel):
    metro_clusters = [region.replace("-", "_") for region in Region.VICTORIA_METRO]
    regional_clusters = [region.replace("-", "_") for region in Region.VICTORIA_RURAL]
    vic = params.victorian_clusters
    country = params.country

    # A bit of a hack - to get rid of all the infectious populations from the regional clusters
    for i_comp, comp in enumerate(model.compartments):
        if any(
            [comp.has_stratum("cluster", cluster) for cluster in regional_clusters]
        ) and not comp.has_name(Compartment.SUSCEPTIBLE):
            model.initial_population[i_comp] = 0.0
        elif any(
            [comp.has_stratum("cluster", cluster) for cluster in metro_clusters]
        ) and not comp.has_name(Compartment.SUSCEPTIBLE):
            model.initial_population[i_comp] *= 9.0 / 4.0

    """
    Hack in a custom (144x144) mixing matrix where each region is adjusted individually
    based on its time variant mobility data.
    """

    # Get the inter-cluster mixing matrix
    # intercluster_mixing_matrix = create_assortative_matrix(vic.intercluster_mixing, CLUSTER_STRATA)
    intercluster_mixing_matrix = get_vic_cluster_adjacency_matrix(vic.intercluster_mixing)

    # Replace regional Victoria maximum effect calibration parameters with the metro values for consistency
    for microdist_process in ["face_coverings", "behaviour"]:
        vic.regional.mobility.microdistancing[
            f"{microdist_process}_adjuster"
        ].parameters.effect = vic.metro.mobility.microdistancing[
            f"{microdist_process}_adjuster"
        ].parameters.effect

    # Get new mixing matrix
    static_mixing_matrix = inputs.get_country_mixing_matrix("all_locations", country.iso3)
    get_mixing_matrix = build_victorian_mixing_matrix_func(
        static_mixing_matrix,
        vic.metro.mobility,
        vic.regional.mobility,
        country,
        intercluster_mixing_matrix,
    )
    return get_mixing_matrix


def create_assortative_matrix(off_diagonal_values, strata):
    """
    Create a matrix with all values the same except for the diagonal elements, which are greater, according to the
    requested value to go in the off-diagonal elements. To be used for creating a standard assortative mixing matrix
    according to any number of interacting groups.
    """

    matrix_dimensions = len(strata)
    assert 0. <= off_diagonal_values <= 1. / matrix_dimensions
    off_diagonal_values = (np.ones([matrix_dimensions, matrix_dimensions]) * off_diagonal_values)
    diagonal_top_ups = np.eye(matrix_dimensions) * (1. - matrix_dimensions * off_diagonal_values)
    assortative_matrix = off_diagonal_values + diagonal_top_ups

    # Check matrix symmetric and rows and columns sum to one
    tolerance = 1e-10
    assert np.all(np.abs(assortative_matrix - assortative_matrix.T) == 0.)
    assert np.all(assortative_matrix.sum(axis=0) - 1 < tolerance)
    assert np.all(assortative_matrix.sum(axis=1) - 1 < tolerance)

    return assortative_matrix


def get_vic_cluster_adjacency_matrix(off_diagonal_value):
    """
    Create adjacency matrix for Victoria inter-cluster mixing
    """

    # Data should really be in the parameters file, but this is likely to be temporary
    adjacency_dict = {
        Region.BARWON_SOUTH_WEST: [Region.WEST_METRO, Region.NORTH_METRO, Region.GRAMPIANS],
        Region.GRAMPIANS: [Region.LODDON_MALLEE, Region.WEST_METRO, Region.NORTH_METRO],
        Region.LODDON_MALLEE: [Region.HUME, Region.NORTH_METRO, Region.WEST_METRO],
        Region.HUME: [Region.GIPPSLAND, Region.NORTH_METRO, Region.SOUTH_EAST_METRO],
        Region.GIPPSLAND: [Region.SOUTH_EAST_METRO, Region.SOUTH_METRO],
        Region.WEST_METRO: [Region.NORTH_METRO],
        Region.NORTH_METRO: [Region.SOUTH_EAST_METRO],
        Region.SOUTH_EAST_METRO: [Region.SOUTH_METRO],
    }

    # Start from matrix of zeros
    adj_matrix = np.zeros((9, 9))

    # Populate adjacent elements
    for index_region in adjacency_dict:
        index_region_position = Region.VICTORIA_SUBREGIONS.index(index_region)
        for other_region in adjacency_dict[index_region]:
            other_region_position = Region.VICTORIA_SUBREGIONS.index(other_region)
            adj_matrix[index_region_position, other_region_position] = off_diagonal_value
            adj_matrix[other_region_position, index_region_position] = off_diagonal_value

    # Fill remaining elements to sum rows and columns to one for assortative geographic mixing
    np.fill_diagonal(
        adj_matrix,
        1. - adj_matrix.sum(axis=0)
    )

    # Check matrix symmetric and rows and columns sum to one
    tolerance = 1e-10
    assert np.all(np.abs(adj_matrix - adj_matrix.T) == 0.)
    assert np.all(adj_matrix.sum(axis=0) - 1 < tolerance)
    assert np.all(adj_matrix.sum(axis=1) - 1 < tolerance)

    # To convert to a dataframe to output as CSV to check:
    # adj_dataframe = pd.DataFrame(adj_matrix, index=Region.VICTORIA_SUBREGIONS, columns=Region.VICTORIA_SUBREGIONS)

    return adj_matrix
