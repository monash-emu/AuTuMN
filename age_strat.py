
def adapt_params_to_stratification(data_breakpoints, model_breakpoints, data_param_vals, assumed_max_params=100.):

    """
    Create a new set of parameters associated to the model stratification given parameter values that are known for
    another stratification.

    Args:
        data_breakpoints: tuple defining the breakpoints used in data.
        model_breakpoints: tuple defining the breakpoints used in the model.
        data_param_vals: dictionary containing the parameter values associated with each category defined by data_breakpoints
                         format example: {'_age0to5': 0.0, '_age5to15': 0.5, '_age15up': 1.0}
        assumed_max_params: the assumed maximal value for the parameter (exemple, age: 100 yo).

    Returns:
        dictionary containing the parameter values associated with each category defined by model_breakpoints
    """

    # Analogous method to that in model.define_age_structure
    def get_strat_from_breakpoints(breakpoints):
        strat = {}
        name = '_age0to' + str(int(breakpoints[0]))
        strat[name] = [0., breakpoints[0]]
        for i in range(len(breakpoints) - 1):
            name = '_age' + str(int(breakpoints[i])) + 'to' + str(int(breakpoints[i + 1]))
            strat[name] = [breakpoints[i], breakpoints[i + 1]]
        name = '_age' + str(int(breakpoints[-1])) + 'up'
        strat[name] = [breakpoints[-1], float('inf')]
        return strat

    data_strat = get_strat_from_breakpoints(data_breakpoints)
    model_strat = get_strat_from_breakpoints(model_breakpoints)

    assert data_param_vals.viewkeys() == data_strat.viewkeys()


    model_param_vals = {}
    for new_name, new_range in model_strat.iteritems():
        new_low, new_up = new_range[0], new_range[1]
        considered_old_cats = []
        for old_name, old_range in data_strat.iteritems():
            if (old_range[0] <= new_low <= old_range[1]) or (old_range[0] <= new_up <= old_range[1]):
                considered_old_cats.append(old_name)
        beta = 0.  # store the new value for the parameter
        for old_name in considered_old_cats:
            alpha = data_param_vals[old_name]
            # calculate the weight to be affected to alpha (w = w_right - w_left)
            w_left = max(new_low, data_strat[old_name][0])
            if (data_strat[old_name][1] == float('inf')) and (new_up == float('inf')):
                w_right = assumed_max_params
                new_up = assumed_max_params
            else:
                w_right = min(new_up, data_strat[old_name][1])

            beta += alpha * (w_right - w_left)
        beta = beta / (new_up - new_low)
        model_param_vals[new_name] = beta

    return(model_param_vals)


# * * * * * * * * * * * * * * * * * * * * * *
#                   Test

if __name__ == "__main__":

    data_breaks = [5., 15.]
    model_breaks = [2., 7., 20.]

    data_param_vals = {'_age0to5': 0.5,
                       '_age5to15': 0.,
                       '_age15up': 1.0}

    model_param = adapt_params_to_stratification(data_breaks, model_breaks, data_param_vals)
    print(model_param)

