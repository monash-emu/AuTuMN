import numpy as np
import math
from scipy import stats
import os
from autumn.db import Database


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

    best_chain_index = np.argmax(
        [mcmc_tables[i]["loglikelihood"].iloc[0] for i in range(len(mcmc_tables))]
    )
    non_param_cols = ["idx", "Scenario", "loglikelihood", "accept"]
    param_list = [c for c in mcmc_tables[0].columns if c not in non_param_cols]
    map_estimates = {}
    for param in param_list:
        map_estimates[param] = mcmc_tables[best_chain_index][param].iloc[0]
    return map_estimates, best_chain_index


if __name__ == "__main__":
    calib_dir = os.path.join(
        "../../data", "outputs", "calibrate", "covid_19", "sweden", "762c076f-2020-07-06"
    )
    map_estimates, best_chain_index = collect_map_estimate(calib_dir)
    for key, value in map_estimates.items():
        print(key + ": " + str(value))

    print()
    print("Obtained from chain " + str(best_chain_index))
