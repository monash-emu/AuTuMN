from .age_stratification import add_zero_to_age_breakpoints, split_age_parameter
from .data_structures import (
    convert_boolean_list_to_indices,
    create_cumulative_dict,
    element_list_division,
    element_list_multiplication,
    increment_list_by_index,
    normalise_dict,
    order_dict_by_keys,
)
from .flowchart import create_flowchart
from .stratification_funcs import (
    create_additive_function,
    create_function_of_function,
    create_multiplicative_function,
    create_sloping_step_function,
    create_time_variant_multiplicative_function,
)
from .string import (
    create_stratified_name,
    create_stratum_name,
    extract_reversed_x_positions,
    extract_x_positions,
    find_name_components,
    find_stem,
    find_stratum_index_from_string,
)


def get_all_proportions(names, proportions):
    """
    Determine what % of population get assigned to the different groups.
    """
    proportion_allocated = sum(proportions.values())
    remaining_names = [n for n in names if n not in proportions]
    count_remaining = len(remaining_names)
    assert set(proportions.keys()).issubset(names), "Invalid proprotion keys"
    assert proportion_allocated <= 1, "Sum of proportions must not exceed 1.0"
    if not remaining_names:
        assert proportion_allocated == 1, "Sum of proportions must be 1.0"
        return proportions
    else:
        # Divide the remaining proprotions equally
        starting_proportion = (1 - proportion_allocated) / count_remaining
        remaining_proportions = {name: starting_proportion for name in remaining_names}
        return {**proportions, **remaining_proportions}


from typing import List, Tuple, Dict


def get_stratified_compartments(
    stratification_name: str,
    strata_names: List[str],
    stratified_compartments: List[str],
    split_proportions: Dict[str, float],
    current_names: List[str],
    current_values: List[float],
) -> Tuple[Dict[str, float], List[str]]:
    """
    Stratify the model compartments into sub-compartments, based on the strata names provided,
    Split the population according to the provided proprotions.
    Stratification will be applied  to compartment_names and compartment_values.
    Only compartments specified in `stratified_compartments` will be stratified.
    """
    to_add = {}
    to_remove = []
    # Find the existing compartments that need stratification
    compartments_to_stratify = [c for c in current_names if find_stem(c) in stratified_compartments]
    for compartment in compartments_to_stratify:
        # Add new stratified compartment.
        for stratum in strata_names:
            name = create_stratified_name(compartment, stratification_name, stratum)
            idx = current_names.index(compartment)
            value = current_values[idx] * split_proportions[stratum]
            to_add[name] = value

        # Remove the original compartment, since it has now been stratified.
        to_remove.append(compartment)

    return to_add, to_remove
