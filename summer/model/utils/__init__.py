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
