from autumn.models.covid_19.parameters import TimeSeries


def get_adjusted_age_specific_mixing(age_categories, adjustment_start_time, adjustment_end_time, contact_rate_multiplier):
    """
    Adjust the rate of contacts by age base on the parameter age_specific_risk_multiplier.
    This will be ignored if age-specific adjustments are already requested through the mobility parameter.
    """
    assert adjustment_end_time > adjustment_start_time + 2, "the age-specific adjustment has to last more than two days"

    age_mixing = {}
    for age_group in age_categories:
        age_mixing[age_group] = TimeSeries(
            times=[
                adjustment_start_time - 1,
                adjustment_start_time + 1,
                adjustment_end_time - 1,
                adjustment_end_time + 1
            ],
            values=[
                1.,
                contact_rate_multiplier,
                contact_rate_multiplier,
                1.
            ],
        )

    return age_mixing
