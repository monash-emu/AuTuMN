from autumn.curve import scale_up_function


def set_tv_importation_rate(model, importation_times, importation_n_cases):

    # scale importation rates using population size
    importation_rates = [n / model.starting_population for n in importation_n_cases]

    # adjust rates for proportion of asymptomatic cases (assuming 40% symptomatic)
    importation_rates = [r / 0.4 for r in importation_rates]

    # shift importation times to account for incubation + pre-symptomatic period + time to diagnostic
    shift = (
        model.parameters["compartment_periods"]["exposed"]
        + model.parameters["compartment_periods"]["presympt"]
        + 2.0
    )
    shifted_importation_times = [t - shift for t in importation_times]

    tv_param = scale_up_function(shifted_importation_times, importation_rates)

    model.parameters["importation_rate"] = "importation_rate"
    model.adaptation_functions["importation_rate"] = tv_param

    return model
