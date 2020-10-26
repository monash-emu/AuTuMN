from typing import List, Tuple, Dict, Callable

import numpy as np

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
from .stratification_funcs import (
    create_additive_function,
    create_function_of_function,
    create_multiplicative_function,
    create_sloping_step_function,
    create_time_variant_multiplicative_function,
)
from .validation import validate_stratify, validate_model
