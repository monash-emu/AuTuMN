

def stratify_by_age(model_to_stratify, age_strata, mixing_matrix, starting_props):
    """
    Stratify model by age
    Note that because the string passed is 'agegroup' rather than 'age', the standard SUMMER demography is not triggered
    """
    age_breakpoints = [int(i_break) for i_break in age_strata]
    model_to_stratify.stratify(
        "agegroup",
        age_breakpoints,
        [],
        starting_props,
        mixing_matrix=mixing_matrix,
        adjustment_requests={'to_infectious': {stratum: 1. for stratum in age_strata}},
        verbose=False
    )
    return model_to_stratify


def stratify_by_location(model_to_stratify, location_mixing, location_strata):
    props_location = {'a': 0.333, 'b': 0.333, 'c': 0.334}
    model_to_stratify.stratify(
        "location",
        location_strata,
        [],
        requested_proportions=props_location,
        verbose=False,
        entry_proportions=props_location,
        mixing_matrix=location_mixing,
    )
    return model_to_stratify


