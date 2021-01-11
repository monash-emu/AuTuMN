from apps.covid_19.model.parameters import TimeSeries


def get_elderly_protection_mixing(elderly_mixing_reduction: dict):
    """
    Decrease the rate of contacts involving the elderly.
    If age-specific adjustments are already requested, the elderly protection will be ignored
    """
    age_mixing = {}
    for age_group in elderly_mixing_reduction["age_categories"]:
        age_mixing[age_group] = TimeSeries(
            times=elderly_mixing_reduction["drop_time_range"],
            values=[
                1,
                1 - elderly_mixing_reduction["relative_reduction"],
            ],
        )

    return age_mixing
