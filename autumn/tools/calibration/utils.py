import math
import os
from typing import Any, Dict, List

import numpy as np
import yaml
from pyDOE import lhs
from scipy import special, stats
from scipy.optimize import minimize

from autumn.tools.db import Database
from autumn.tools import db


def add_dispersion_param_prior_for_gaussian(par_priors, target_outputs):
    for t in target_outputs:
        if t["loglikelihood_distri"] in ["normal", "trunc_normal"] and "sd" not in t:
            max_val = max(t["values"])
            # sd_ that would make the 95% gaussian CI cover half of the max value (4*sd = 95% width)
            sd_ = 0.25 * max_val / 4.0
            lower_sd = sd_ / 2.0
            upper_sd = 2.0 * sd_

            par_priors.append(
                {
                    "param_name": t["output_key"] + "_dispersion_param",
                    "distribution": "uniform",
                    "distri_params": [lower_sd, upper_sd],
                },
            )
    return par_priors


def sample_prior(prior_dict, quantile):
    """
    Sample a particular prior, specified with prior_dict using the quantile specified with prop.

    """
    if prior_dict["distribution"] == "uniform":
        sample = prior_dict["distri_params"][0] + quantile * (
            prior_dict["distri_params"][1] - prior_dict["distri_params"][0]
        )
    elif prior_dict["distribution"] == "lognormal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        sample = math.exp(mu + math.sqrt(2) * sd * special.erfinv(2 * quantile - 1))
    elif prior_dict["distribution"] == "trunc_normal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        bounds = prior_dict["trunc_range"]
        sample = stats.truncnorm.ppf(
            quantile, (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
        )
    elif prior_dict["distribution"] == "normal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        sample = stats.norm.ppf(
            quantile, loc=mu, scale=sd
        )
    elif prior_dict["distribution"] == "beta":
        sample = stats.beta.ppf(
            quantile,
            prior_dict["distri_params"][0],
            prior_dict["distri_params"][1],
        )
    elif prior_dict["distribution"] == "gamma":
        sample = stats.gamma.ppf(
            quantile,
            prior_dict["distri_params"][0],
            0.0,
            prior_dict["distri_params"][1],
        )
    else:
        raise_error_unsupported_prior(prior_dict["distribution"])

    return float(sample)


def draw_independent_samples(independent_priors):
    """
    :param priors: list of prior dictionaries
    :return: a dictionary of sampled values
    """
    independent_samples = {}
    for prior_dict in independent_priors:
        independent_samples[prior_dict["param_name"]] = sample_prior(prior_dict, np.random.uniform())

    return independent_samples


def sample_starting_params_from_lhs(par_priors: List[Dict[str, Any]], n_samples: int):
    """
    Use Latin Hypercube Sampling to define MCMC starting points
    :param par_priors: a list of dictionaries defining the prior distributions
    :param n_samples: integer
    :return: a list of dictionaries
    """
    list_of_starting_params = [{} for _ in range(n_samples)]

    # Draw a Latin hypercube (all values in [0-1])
    hypercube = lhs(n=len(par_priors), samples=n_samples, criterion="center")
    for j, prior_dict in enumerate(par_priors):
        for i in range(n_samples):
            prop = hypercube[i, j]
            quantile = sample_prior(prior_dict, prop)
            list_of_starting_params[i][prior_dict["param_name"]] = quantile

    return list_of_starting_params


def find_decent_starting_point(prior_dict):
    """
    Find an MCMC parameter initial value, using the mean of the specified prior
    :param prior_dict:
    :return: float (starting point)
    """
    if prior_dict["distribution"] == "uniform":
        x = np.mean(prior_dict["distri_params"])
    elif prior_dict["distribution"] == "beta":
        a = prior_dict["distri_params"][0]
        b = prior_dict["distri_params"][1]
        x = a / (a + b)
    elif prior_dict["distribution"] == "gamma":
        shape = prior_dict["distri_params"][0]
        scale = prior_dict["distri_params"][1]
        x = shape * scale
    elif prior_dict["distribution"] == "normal":
        x = prior_dict["distri_params"][0]
    else:
        raise_error_unsupported_prior(prior_dict["distribution"])

    return x


def calculate_prior(prior_dict, x, log=True):
    """
    Calculate the log-prior value given the distribution details and the evaluation point
    :param prior_dict: distribution details
    :param x: evaluation point
    :param log: boolean
        Whether to return the log-PDF of the PDF
    :return: log-PDF(x) or PDF(x)
    """
    if prior_dict["distribution"] == "uniform":
        if log:
            y = math.log(1.0 / (prior_dict["distri_params"][1] - prior_dict["distri_params"][0]))
        else:
            y = 1.0 / (prior_dict["distri_params"][1] - prior_dict["distri_params"][0])
    elif prior_dict["distribution"] == "lognormal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        if log:
            y = stats.lognorm.logpdf(
                x=x, s=sd, scale=math.exp(mu)
            )  # see documentation of stats.lognorm for scale
        else:
            y = stats.lognorm.pdf(x=x, s=sd, scale=math.exp(mu))
    elif prior_dict["distribution"] == "trunc_normal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        bounds = prior_dict["trunc_range"]
        if log:
            y = stats.truncnorm.logpdf(
                x, (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
            )
        else:
            y = stats.truncnorm.pdf(
                x, (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
            )
    elif prior_dict["distribution"] == "normal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        if log:
            y = stats.norm.logpdf(
                x, loc=mu, scale=sd
            )
        else:
            y = stats.norm.pdf(
                x, loc=mu, scale=sd
            )
    elif prior_dict["distribution"] == "beta":
        a = prior_dict["distri_params"][0]
        b = prior_dict["distri_params"][1]
        if log:
            y = stats.beta.logpdf(x, a, b)
        else:
            y = stats.beta.pdf(x, a, b)
    elif prior_dict["distribution"] == "gamma":
        shape = prior_dict["distri_params"][0]
        scale = prior_dict["distri_params"][1]
        if log:
            y = stats.gamma.logpdf(x, shape, 0.0, scale)
        else:
            y = stats.gamma.pdf(x, shape, 0.0, scale)
    else:
        raise_error_unsupported_prior()
    return float(y)


def raise_error_unsupported_prior(distribution):
    raise ValueError(distribution + "distribution not supported in autumn_mcmc at the moment")


def collect_map_estimate(calib_dirpath: str):
    """
    Read all MCMC outputs found in mcmc_db_folder and print the map parameter values.
    :return: dict of parameters
    """
    mcmc_tables = []
    db_paths = [
        os.path.join(calib_dirpath, f)
        for f in os.listdir(calib_dirpath)
        if f.endswith(".db") and not f.startswith("mcmc_percentiles")
    ]
    for db_path in db_paths:
        db = Database(db_path)
        mcmc_tables.append(db.query("mcmc_run").sort_values(by="loglikelihood", ascending=False))

    print("Maximum loglikelihood for each chain:")
    print([mcmc_tables[i]["loglikelihood"].iloc[0] for i in range(len(mcmc_tables))])
    print()

    print("Chains' lengths:")
    print([len(mcmc_tables[i]["loglikelihood"]) for i in range(len(mcmc_tables))])
    print()

    best_chain_index = np.argmax(
        [mcmc_tables[i]["loglikelihood"].iloc[0] for i in range(len(mcmc_tables))]
    )
    non_param_cols = ["idx", "Scenario", "loglikelihood", "accept"]
    param_list = [c for c in mcmc_tables[0].columns if c not in non_param_cols]
    map_estimates = {}
    for param in param_list:
        map_estimates[param] = mcmc_tables[best_chain_index][param].iloc[0]
    return map_estimates, best_chain_index


def print_reformated_map_parameters(map_estimates):
    param_as_list_indices = {}
    param_stem_already_printed = []
    for key, value in map_estimates.items():
        if "dispersion_param" in key:
            continue
        if "(" in key:
            stem = key.split("(")[0]
            index = key.split("(")[1].split(")")[0]
            if stem not in param_as_list_indices:
                param_as_list_indices[stem] = [index]
            else:
                param_as_list_indices[stem].append(index)
        elif "." in key:
            components = key.split(".")
            i = 0
            for comp in components:
                print_this = False
                str_to_print = ""
                for j in range(i):
                    str_to_print += "  "
                if comp not in param_stem_already_printed:
                    str_to_print += comp + ": "
                    print_this = True
                    if i == len(components) - 1:
                        str_to_print += str(value)
                    else:
                        param_stem_already_printed.append(comp)
                if print_this:
                    print(str_to_print)
                i += 1
        else:
            print(key + ": " + str(value))
    for param_stem, index_list in param_as_list_indices.items():
        str_to_print = param_stem + ": ["
        index_list.sort()
        i = 0
        for index in index_list:
            if i > 0:
                str_to_print += ", "
            str_to_print += str(map_estimates[param_stem + "(" + str(index) + ")"])
            i += 1
        str_to_print += "]"
        print(str_to_print)


def specify_missing_prior_params(priors: dict):
    """
    Work out the prior distribution parameters if they were not specified
    """
    for i, p_dict in enumerate(priors):
        if "distri_params" not in p_dict:
            assert (
                "distri_mean" in p_dict and "distri_ci" in p_dict
            ), "Please specify distri_mean and distri_ci."
            if "distri_ci_width" in p_dict:
                distri_params = find_distribution_params_from_mean_and_ci(
                    p_dict["distribution"],
                    p_dict["distri_mean"],
                    p_dict["distri_ci"],
                    p_dict["distri_ci_width"],
                )
            else:
                distri_params = find_distribution_params_from_mean_and_ci(
                    p_dict["distribution"],
                    p_dict["distri_mean"],
                    p_dict["distri_ci"],
                )
            if p_dict["distribution"] == "beta":
                priors[i]["distri_params"] = [
                    distri_params["a"],
                    distri_params["b"],
                ]
            elif p_dict["distribution"] == "gamma":
                priors[i]["distri_params"] = [
                    distri_params["shape"],
                    distri_params["scale"],
                ]
            else:
                raise_error_unsupported_prior(p_dict["distribution"])


def find_distribution_params_from_mean_and_ci(distribution, mean, ci, ci_width=0.95):
    """
    Work out the parameters of a given distribution based on a desired mean and CI
    :param distribution: string
        either 'gamma' or 'beta' for now
    :param mean: float
        the desired mean value
    :param ci: list or tuple of length 2
        the lower and upper bounds of the CI
    :param ci_width:
        the width of the desired CI
    :return:
        a dictionary with the parameters
    """
    assert len(ci) == 2 and ci[1] > ci[0] and 0.0 < ci_width < 1.0
    percentile_low = (1.0 - ci_width) / 2.0
    percentile_up = 1.0 - percentile_low

    if distribution == "beta":
        assert 0.0 < ci[0] < 1.0 and 0.0 < ci[1] < 1.0 and 0.0 < mean < 1.0

        def distance_to_minimise(a):
            b = a * (1.0 - mean) / mean
            vals = stats.beta.ppf([percentile_low, percentile_up], a, b)
            dist = sum([(ci[i] - vals[i]) ** 2 for i in range(2)])
            return dist

        sol = minimize(distance_to_minimise, [1.0], bounds=[(0.0, None)], tol=1.0e-32)
        best_a = sol.x
        best_b = best_a * (1.0 - mean) / mean
        params = {"a": best_a, "b": best_b}

    elif distribution == "gamma":
        assert ci[0] > 0 and ci[1] > 0 and mean > 0.0

        def distance_to_minimise(scale):
            shape = mean / scale
            vals = stats.gamma.ppf([percentile_low, percentile_up], shape, 0, scale)
            dist = sum([(ci[i] - vals[i]) ** 2 for i in range(2)])
            return dist

        sol = minimize(distance_to_minimise, [1.0], bounds=[(1.0e-11, None)])
        best_scale = sol.x
        best_shape = mean / best_scale

        params = {"shape": best_shape, "scale": best_scale}
    else:
        raise ValueError(distribution + " distribution is not supported for the moment")

    return params


def ignore_calibration_target_before_date(target, date):
    indices = [i_time for i_time, time in enumerate(target["times"]) if time > date]
    return truncate_target(target, indices)


def ignore_calibration_target_before_index(target, index):

    # Truncate the target based on the index requested, including that index itself (which is why one is subtracted)
    indices = [i_index for i_index in range(len(target["times"])) if i_index > index - 1]
    return truncate_target(target, indices)


def truncate_target(target, indices):
    return {
        "times": [target["times"][i_time] for i_time in indices],
        "values": [target["values"][i_time] for i_time in indices],
    }


def get_uncertainty_df(calib_dir_path, mcmc_tables, targets):

    try:  # if PBI processing has been performed already
        uncertainty_df = db.load.load_uncertainty_table(calib_dir_path)
    except:  # calculates percentiles
        derived_output_tables = db.load.load_derived_output_tables(calib_dir_path)
        mcmc_all_df = db.load.append_tables(mcmc_tables)
        do_all_df = db.load.append_tables(derived_output_tables)

        # Determine max chain length, throw away first half of that
        max_run = mcmc_all_df["run"].max()
        half_max = max_run // 2
        mcmc_all_df = mcmc_all_df[mcmc_all_df["run"] >= half_max]
        uncertainty_df = db.uncertainty.calculate_mcmc_uncertainty(mcmc_all_df, do_all_df, targets, True)
    return uncertainty_df

