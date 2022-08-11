from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd
import itertools

from summer import Stratification, Multiply

from autumn.models.sm_jax.mixing_matrix import build_dynamic_mixing_matrix
from autumn.models.sm_jax.parameters import Parameters
from autumn.models.sm_jax.constants import FlowName
from autumn.core.inputs import get_population_by_agegroup
from autumn.core.utils.utils import weighted_average


def get_relevant_indices(
        standard_breaks: List[int],
        model_groups: List[str],
) -> Dict[str, List[int]]:
    """
    Find the standard source age brackets relevant to each modelled age bracket.

    Args:
        standard_breaks: The standard source age brackets (5-year brackets to 75+)
        model_groups: The age brackets being applied in the model

    Returns:
        Keys are the modelled age groups, values are lists containing the standard agebreaks that apply to each one

    """

    # Collate up the dictionary by modelled age groups
    relevant_indices = {}
    model_groups = [int(group) for group in model_groups]
    for i_age, model_agegroup in enumerate(model_groups):
        age_index_low = standard_breaks.index(model_agegroup)
        age_index_up = standard_breaks[-1] if model_agegroup == model_groups[-1] else standard_breaks.index(model_groups[i_age + 1])
        relevant_indices[str(model_agegroup)] = standard_breaks[age_index_low: age_index_up]

    # Should be impossible for this check to fail
    msg = "Not all source age groups being mapped to modelled age groups"
    assert list(itertools.chain.from_iterable(relevant_indices.values())) == standard_breaks, msg

    return relevant_indices


def convert_param_agegroups(
        iso3: str,
        region: Union[None, str],
        source_dict: Dict[int, float],
        modelled_age_groups: List[str],
) -> pd.Series:
    """
    Converts the source parameters to match the model age groups.

    Args:
        iso3: Parameter for get_population_by_agegroup
        region: Parameter for get_population_by_agegroup
        source_dict: A list of parameter values provided according to 5-year band, starting from 0-4
        modelled_age_groups: Parameter for get_population_by_agegroup

    Returns:
        The dictionary of the processed parameters in the format needed by the model

    """

    # Get default age brackets and the population structured with these default categories
    source_agebreaks = list(source_dict.keys())
    total_pops_5year_bands = get_population_by_agegroup(source_agebreaks, iso3, region=region, year=2020)
    total_pops_5year_dict = {age: pop for age, pop in zip(source_agebreaks, total_pops_5year_bands)}

    msg = "Modelled age group(s) incorrectly specified, not in standard age breaks"
    assert all([int(age_group) in source_agebreaks for age_group in modelled_age_groups]), msg

    # Find out which of the standard source categories (values) apply to each modelled age group (keys)
    relevant_source_indices = get_relevant_indices(source_agebreaks, modelled_age_groups)

    # For each age bracket
    param_values = {}
    for model_agegroup in modelled_age_groups:
        relevant_indices = relevant_source_indices[model_agegroup]
        values = {k: source_dict[k] for k in relevant_indices}
        weights = {k: total_pops_5year_dict[k] for k in relevant_indices}
        param_values[model_agegroup] = weighted_average(values, weights)

    return pd.Series(param_values)


def get_agegroup_strat(
        params: Parameters,
        age_groups: List[str],
        age_pops: pd.Series,
        matrices: np.array,
        compartments: List[str],
        is_dynamic_matrix: bool,
        age_suscept: Optional[pd.Series],
) -> Stratification:
    """
    Function to create the age group stratification object.

    We use "Stratification" instead of "AgeStratification" for this model, to avoid triggering
    automatic demography features (which work on the assumption that the time is in years, so would be totally wrong)
    This will be revised in future versions of summer, in which model times will be datetime objects rather than AuTuMN
    bespoke data structures.

    Args:
        params: All model parameters
        age_groups: List of age groups as string
        age_pops: The population distribution by age
        matrices: The static age-specific mixing matrix
        compartments: All the model compartments
        is_dynamic_matrix: Whether to use the dynamically scaling matrix or the static (all locations) mixing matrix
        age_suscept: Adjustments to infection rate based on the susceptibility of modelled age groups

    Returns:
        The age stratification summer object

    """

    age_strat = Stratification("agegroup", age_groups, compartments)

    # Heterogeneous mixing by age
    dynamic_args = matrices, params.mobility, params.country
    final_matrix = build_dynamic_mixing_matrix(*dynamic_args) if is_dynamic_matrix else matrices["all_locations"]
    age_strat.set_mixing_matrix(final_matrix)

    # Set distribution of starting population
    age_split_props = age_pops / age_pops.sum()
    age_strat.set_population_split(age_split_props.to_dict())

    # Adjust infection flows based on the susceptibility of the age group
    if isinstance(age_suscept, pd.Series):
        age_strat.set_flow_adjustments(
            FlowName.INFECTION,
            {k: Multiply(v) for k, v in age_suscept.items()}
        )

    return age_strat
