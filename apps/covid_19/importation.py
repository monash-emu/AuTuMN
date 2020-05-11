from autumn.curve import scale_up_function


def set_tv_importation_rate(model, importation_times, importation_n_cases):
    """
    When imported cases need to be accounted for to inflate the force of infection but they are not explicitly included
    in the modelled population.
    """
    # scale-up curve for importation numbers
    importation_numbers_scale_up = scale_up_function(importation_times, importation_n_cases)

    # time-variant infectiousness of imported cases
    tv_imported_infectiousness = scale_up_function(
        [75., 77., 88., 90.],
        [1., 1. - model.parameters['self_isolation_effect'], 1. - model.parameters['self_isolation_effect'],
         1. - model.parameters['enforced_isolation_effect']],
        method=4
    )

    def tv_recruitment_rate(t):
        return importation_numbers_scale_up(t) * tv_imported_infectiousness(t) * model.parameters['contact_rate'] / \
               model.starting_population

    model.parameters["import_secondary_rate"] = "import_secondary_rate"
    model.adaptation_functions["import_secondary_rate"] = tv_recruitment_rate

    return model


def set_tv_importation_as_birth_rates(model, importation_times, importation_n_cases):
    """
    When imported cases are explicitly simulated as part of the modelled population. They enter the late_infectious
    compartment through a birth process
    """
    # scale-up curve for importation numbers
    importation_numbers_scale_up = scale_up_function(importation_times, importation_n_cases)

    def tv_recruitment_rate(t):
        return importation_numbers_scale_up(t) / model.starting_population

    model.parameters["crude_birth_rate"] = "crude_birth_rate"
    model.time_variants["crude_birth_rate"] = tv_recruitment_rate

    return model
