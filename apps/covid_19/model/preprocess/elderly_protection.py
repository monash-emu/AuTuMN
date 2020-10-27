from apps.covid_19.model.parameters import TimeSeries


def apply_elderly_protection(params):
    """
    Decrease the rate of contacts involving the elderly
    """
    params.mobility.age_mixing = {}
    for age_group in params.elderly_mixing_reduction['age_categories']:
        params.mobility.age_mixing[age_group] = TimeSeries(
                times=params.elderly_mixing_reduction['drop_time_range'],
                values=[
                    1,
                    1 - params.elderly_mixing_reduction['relative_reduction'],
                ]
        )

    return params
