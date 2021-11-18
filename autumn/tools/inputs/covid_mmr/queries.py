from autumn.tools.inputs.database import get_input_db


def get_mmr_testing_numbers():
    """
    Returns daily PCR test numbers for Myanmar
    """

    input_db = get_input_db()
    df = input_db.query(
        "covid_mmr",
        columns=["date_index", "tests"],
    )

    df.dropna(how="any", inplace=True)

    test_dates = df.date_index.to_numpy()
    test_values = df.tests.to_numpy()

    return test_dates, test_values


def get_mmr_vac_coverage(age_group, age_pops):

    times = [
        366,  # 1st Jan 2021
        393,  # 27th Jan
        499,  # 13th May
        522,  # 5th June
        599,  # 21st Aug
        606,  # 28th Aug
        613,  # 4th Sept
        620,  # 11th Sept
        627,  # 18th Sept
        634,  # 25th Sept
        641,  # 2nd Oct
        648,  # 9th Oct
        655,  # 16th Oct
        662,  # 23rd Oct
        665,  # 26th Oct
        670,  # 31st Oct
        678,  # 8th Nov
        731,  # 31st December 2021
        1096,  # 31st December 2022
    ]

    # For the adult population
    if int(age_group) >= 15:
        adult_denominator = sum(age_pops[3:])

        # Slide 5 of Mya Yee Mon's PowerPoint sent on 12th November - applied to the 15+ population only
        at_least_one_dose = [
            0,
            104865,
            1772177,
            1840758,
            4456857,
            4683410,
            4860264,
            4944654,
            5530365,
            7205913,
            8390746,
            9900823,
            11223285,
            12387573,
            12798322,
            13244996,
            13905795,
        ]

        # Convert doses to coverage
        coverage_values = [i_doses / adult_denominator for i_doses in at_least_one_dose]

        # Add future targets
        target_inflation_for_age = sum(age_pops) / adult_denominator
        target_all_age_coverage = [0.4, 0.7]
        target_adult_coverage = [
            target_inflation_for_age * i_cov for i_cov in target_all_age_coverage
        ]
        assert all([0.0 <= i_cov <= 1.0 for i_cov in target_adult_coverage])
        coverage_values += target_adult_coverage

    # For the children, no vaccination
    else:
        coverage_values = [0.0] * len(times)

    return times, coverage_values
