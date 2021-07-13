# from math import exp, log
from numpy import mean, exp, log


def tune_jumping_sd(eval_points, eval_loglikelihoods, relative_likelihood_reduction):
    """
    Identify how much parameter variation is required in order to modify the likelihood by a given quantity
    (target_likelihood_ratio), when starting from the maximum likelihood value.
    :param eval_points: list of evaluated parameter values
    :param eval_loglikelihoods: list of loglikelihood values associated with eval_points
    :param relative_likelihood_reduction: relative likelihood reduction associated with a typical jump starting from MLE.
    :return: the tuned jumping sd
    """
    assert 0. < relative_likelihood_reduction < 1., "relative_likelihood_reduction must be in (0, 1)"

    # identify max value and optimal point
    s = set(zip(eval_points, eval_loglikelihoods))
    best_point, max_loglikelihood = max(s, key=lambda x: x[1])
    index_max = eval_points.index(best_point)

    # identify lower_threshold
    if max_loglikelihood < -100:
        print("more parameter samples may be required to identify higher-likelihood regions")
        return None

    max_likelihood = exp(max_loglikelihood)
    lower_threshold_likelihood = max_likelihood * (1 - relative_likelihood_reduction)
    lower_threshold_loglikelihood = log(lower_threshold_likelihood)

    # identify cut_off points moving both to the left and to the right of best_point
    moving_directions = []
    if index_max > 0:
        moving_directions.append(-1)  # move to the left
    if index_max < len(eval_points) - 1:
        moving_directions.append(1)  # move to the right

    cutoff_solutions = []
    for moving_direction in moving_directions:
        latest_over_index, latest_over_point, latest_over_val = index_max, best_point, max_loglikelihood
        while True:
            eval_index = latest_over_index + moving_direction
            eval_point = eval_points[eval_index]
            eval_value = eval_loglikelihoods[eval_index]

            if eval_value <= lower_threshold_loglikelihood:
                delta_y_ratio = (lower_threshold_loglikelihood - eval_value) / (latest_over_val - lower_threshold_loglikelihood)
                sol = (eval_point + latest_over_point * delta_y_ratio) / (1. + delta_y_ratio)
                cutoff_solutions.append(sol)
                break
            elif eval_index == 0 or eval_index == len(eval_points) - 1:
                break
            else:
                latest_over_index, latest_over_point, latest_over_val = eval_index, eval_point, eval_value

    if len(cutoff_solutions) > 0:
        gaps = [abs(best_point - c) for c in cutoff_solutions]
        jumping_sd = mean(gaps)
    else:
        jumping_sd = eval_points[-1] - eval_points[0]

    return jumping_sd


def perform_all_params_proposal_tuning(project, calibration, n_points=100, relative_likelihood_reduction=0.5):

    params_to_tune = [p.name for p in calibration.priors if p.sampling is None]

    for p in params_to_tune:  # FIXME: these tasks should be run in parallel
        tuned_proposal_sd = calibration.tune_proposal(p.name, project, n_points, relative_likelihood_reduction)

    # FIXME: we need to store all the tuned values in a single yaml file
    # Required format for the stored dictionary: {"param_name_1": tuned_proposal_sd_1, "param_name_2": tuned_proposal_sd_2, ...}
