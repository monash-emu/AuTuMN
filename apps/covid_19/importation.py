from autumn.curve import scale_up_function

# dummy proportions for now:
importation_props_by_age = {
    "0": 0.04,
    "5": 0.04,
    "10": 0.04,
    "15": 0.04,
    "20": 0.08,
    "25": 0.09,
    "30": 0.09,
    "35": 0.09,
    "40": 0.09,
    "45": 0.08,
    "50": 0.08,
    "55": 0.08,
    "60": 0.04,
    "65": 0.04,
    "70": 0.04,
    "75": 0.04,
}


def set_tv_importation_rate(model, importation_times, importation_n_cases):
    """
    When imported cases need to be accounted for to inflate the force of infection but they are not explicitly included
    in the modelled population.
    """
    # scale-up curve for importation numbers
    importation_numbers_scale_up = scale_up_function(importation_times, importation_n_cases)

    # time-variant infectiousness of imported cases
    tv_imported_infectiousness = scale_up_function(
        [75.0, 77.0, 88.0, 90.0],
        [
            1.0,
            1.0 - model.parameters["self_isolation_effect"],
            1.0 - model.parameters["self_isolation_effect"],
            1.0 - model.parameters["enforced_isolation_effect"],
        ],
        method=4,
    )

    def tv_recruitment_rate(t):
        return (
            importation_numbers_scale_up(t)
            * tv_imported_infectiousness(t)
            * model.parameters["contact_rate"]
            / model.starting_population
        )

    model.parameters["import_secondary_rate"] = "import_secondary_rate"
    model.adaptation_functions["import_secondary_rate"] = tv_recruitment_rate

    return model


def set_tv_importation_as_birth_rates(model, importation_times, importation_n_cases):
    """
    When imported cases are explicitly simulated as part of the modelled population. They enter the late_infectious
    compartment through a birth process
    """
    # inflate importation numbers to account for undetected cases (assumed to be asymptomatic or sympt non hospital)
    prop_detected = (
        model.parameters["symptomatic_props_imported"]
        * model.parameters["prop_detected_among_symptomatic_imported"]
    )
    importation_n_cases = [n / prop_detected for n in importation_n_cases]

    # scale-up curve for importation numbers
    importation_numbers_scale_up = scale_up_function(
        importation_times, importation_n_cases, method=5, smoothness=5.0
    )

    def tv_recruitment_rate(t):
        return importation_numbers_scale_up(t) / model.starting_population

    model.parameters["crude_birth_rate"] = "crude_birth_rate"
    model.time_variants["crude_birth_rate"] = tv_recruitment_rate

    return model
