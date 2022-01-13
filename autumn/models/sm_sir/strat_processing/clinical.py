from autumn.models.covid_19.detection import find_cdr_function_from_test_data


def get_cdr_func(detect_prop: float, params):
    """

    Args:
        detect_prop: Currently just a single value representing the case detection rate over time
        params:

    Returns:
        The case detection rate function of time

    """

    testing_params = params.testing_to_detection
    if testing_params:
        pop_params = params.population
        cdr_func = find_cdr_function_from_test_data(
            testing_params, params.country.iso3, pop_params.region, pop_params.year
        )

    else:

        def cdr_func(time, processes):
            return detect_prop

    return cdr_func
