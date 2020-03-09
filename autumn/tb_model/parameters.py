def add_time_variant_parameter_to_model(model, parameter_name, parameter_function, n_stratifications):
    """
    Add a time variant parameter to a model, regardless of whether it is stratified or not

    :param model: EpiModel or StratifiedModel
        The model to be updated
    :param parameter_name: str
        Name of the parameter string
    :param parameter_function: function
        Parameter function with time the input variable, giving the parameter value as the output
    :param n_stratifications: int
        Number of stratifications that have been implemented
    :return:
    """
    if n_stratifications == 0:
        model.time_variants[parameter_name] = parameter_function
    else:
        model.parameters[parameter_name] = parameter_name
        model.adaptation_functions[parameter_name] = parameter_function