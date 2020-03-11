

def stratify_by_age(model_to_stratify, age_strata):
    """
    Stratify model by age
    Note that because the string passed is 'agegroup' rather than 'age', the standard SUMMER demography is not triggered
    """
    age_breakpoints = [int(i_break) for i_break in age_strata]
    age_params = {}
    model_to_stratify.stratify(
        "agegroup",
        age_breakpoints,
        [],
        {},
        adjustment_requests=age_params,
        verbose=False,
    )
    return model_to_stratify