import yaml
import os
import logging
from time import time
from itertools import chain
from datetime import datetime
from typing import List, Callable
import shutil

import math
import pandas as pd
import numpy as np
import copy
from scipy.optimize import Bounds, minimize
from scipy import stats, special

from summer.model import StratifiedModel
from autumn import constants
from autumn import db, plots
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit import Timer
from autumn.tool_kit.params import update_params, read_param_value_from_string
from autumn.tool_kit.utils import (
    get_git_branch,
    get_git_hash,
)
from .utils import (
    calculate_prior,
    specify_missing_prior_params,
    raise_error_unsupported_prior,
    sample_starting_params_from_lhs,
)

from .transformations import (
    make_transform_func_with_lower_bound,
    make_transform_func_with_upper_bound,
    make_transform_func_with_two_bounds,
)

from .constants import ADAPTIVE_METROPOLIS

BEST_LL = "best_ll"
BEST_START = "best_start_time"

logger = logging.getLogger(__name__)


class CalibrationMode:
    """Different ways to run the calibration."""

    AUTUMN_MCMC = "autumn_mcmc"
    LEAST_SQUARES = "lsm"
    MODES = [AUTUMN_MCMC, LEAST_SQUARES]


class InitialisationTypes:
    """Different ways to set the intial point for the MCMC."""

    LHS = "lhs"
    CURRENT_PARAMS = "current_params"


class Calibration:
    """
    This class handles model calibration using a Bayesian algorithm if sampling from the posterior distribution is
    required, or using maximum likelihood estimation if only one calibrated parameter set is required.
    A Metropolis Hastings algorithm is used with or without adaptive proposal function. The adaptive approach employed
    was published by Haario et al. ('An  adaptive  Metropolis  algorithm', Bernoulli 7(2), 2001, 223-242)
    """

    def __init__(
        self,
        app_name: str,
        model_builder: Callable[[dict], StratifiedModel],
        model_parameters: dict,
        priors: List[dict],
        targeted_outputs: List[dict],
        chain_index: int,
        total_nb_chains: int,
        region_name: str = "main",
        adaptive_proposal: bool = True,
        initialisation_type: str = InitialisationTypes.LHS,
    ):
        self.app_name = app_name
        self.model_builder = model_builder  # a function that builds a new model without running it
        self.model_parameters = model_parameters
        self.best_start_time = None
        self.priors = priors  # a list of dictionaries. Each dictionary describes the prior distribution for a parameter
        self.adaptive_proposal = adaptive_proposal

        self.param_list = [self.priors[i]["param_name"] for i in range(len(self.priors))]
        self.targeted_outputs = (
            targeted_outputs  # a list of dictionaries. Each dictionary describes a target
        )

        # Validate target output start time.
        model_start = model_parameters["default"]["time"]["start"]
        max_prior_start = None
        for p in priors:
            if p["param_name"] == "time.start":
                max_prior_start = max(p["distri_params"])

        for t in targeted_outputs:
            t_name = t["output_key"]
            min_year = min(t["years"])
            msg = f"Target {t_name} has time {min_year} before model start {model_start}."
            assert min_year >= model_start, msg
            if max_prior_start:
                msg = f"Target {t_name} has time {min_year} before prior start {max_prior_start}."
                assert min_year >= max_prior_start, msg

        # Set a custom end time for all model runs - there is no point running
        # the models after the last calibration targets.
        self.end_time = 2 + max([max(t["years"]) for t in targeted_outputs])

        self.chain_index = chain_index

        # work out missing distribution params for priors
        specify_missing_prior_params(self.priors)

        # Select starting params
        self.starting_point = set_initial_point(
            self.priors, model_parameters, chain_index, total_nb_chains, initialisation_type
        )

        # Save metadata output dir.
        self.output = CalibrationOutputs(chain_index, app_name, region_name)
        metadata = {
            "app_name": app_name,
            "region_name": region_name,
            "start_time": datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
            "git_branch": get_git_branch(),
            "git_commit": get_git_hash(),
        }
        self.output.write_metadata(f"meta-{chain_index}.yml", metadata)
        self.output.write_metadata(f"params-{chain_index}.yml", model_parameters)
        self.output.write_metadata(f"priors-{chain_index}.yml", priors)
        self.output.write_metadata(f"targets-{chain_index}.yml", targeted_outputs)

        self.data_as_array = None  # will contain all targeted data points in a single array
        self.format_data_as_array()
        self.workout_unspecified_target_sds()  # for likelihood definition
        self.workout_unspecified_time_weights()  # for likelihood weighting
        self.workout_unspecified_jumping_sds()  # for proposal function definition

        self.param_bounds = self.get_parameter_bounds()

        self.transform = {}
        self.build_transformations()

        self.iter_num = 0

        self.latest_scenario = None
        self.run_mode = None
        self.main_table = {}
        self.mcmc_trace_matrix = None  # will store the results of the MCMC model calibration
        self.mle_estimates = {}  # will store the results of the maximum-likelihood calibration

        self.evaluated_params_ll = []  # list of tuples:  [(theta_0, ll_0), (theta_1, ll_1), ...]

        if self.chain_index == 0:
            plots.calibration.plot_pre_calibration(self.priors, self.output.output_dir)

    def run_model_with_params(self, proposed_params: dict):
        """
        Run the model with a set of params.
        """
        logger.info(f"Running iteration {self.iter_num}...")

        # Update default parameters to use calibration params.
        param_updates = {"time.end": self.end_time}
        for i, param_name in enumerate(self.param_list):
            param_updates[param_name] = proposed_params[i]

        params = copy.deepcopy(self.model_parameters)
        update_func = lambda ps: update_params(ps, param_updates)
        scenario = Scenario(self.model_builder, 0, params)
        scenario.run(update_func=update_func)
        self.latest_scenario = scenario
        return scenario

    def loglikelihood(self, params, to_return=BEST_LL):
        """
        Calculate the loglikelihood for a set of parameters
        """
        scenario = self.run_model_with_params(params)
        model = scenario.model
        model_start_time = model.times[0]
        considered_start_times = [model_start_time]
        best_start_time = None
        if self.run_mode == CalibrationMode.LEAST_SQUARES:
            # Initial best loglikelihood is a very large +ve number.
            best_ll = 1.0e60
        else:
            # Initial best loglikelihood is a very large -ve number.
            best_ll = -1.0e60

        for considered_start_time in considered_start_times:
            time_shift = considered_start_time - model_start_time
            ll = 0  # loglikelihood if using bayesian approach. Sum of squares if using lsm mode
            for target in self.targeted_outputs:
                key = target["output_key"]
                data = np.array(target["values"])
                time_weigths = target["time_weights"]
                indices = []
                for year in target["years"]:
                    shifted_time = year - time_shift
                    time_idxs = np.where(scenario.model.times == shifted_time)[0]
                    time_idx = time_idxs[0]
                    indices.append(time_idx)

                model_output = model.derived_outputs[key][indices]
                if self.run_mode == CalibrationMode.LEAST_SQUARES:
                    squared_distance = (data - model_output) ** 2
                    ll += np.sum([w * d for (w, d) in zip(time_weigths, squared_distance)])
                else:
                    if "loglikelihood_distri" not in target:  # default distribution
                        target["loglikelihood_distri"] = "normal"
                    if target["loglikelihood_distri"] == "normal":
                        if key + "_dispersion_param" in self.param_list:
                            normal_sd = params[self.param_list.index(key + "_dispersion_param")]
                        else:
                            normal_sd = target["sd"]
                        squared_distance = (data - model_output) ** 2
                        ll += -(0.5 / normal_sd ** 2) * np.sum(
                            [w * d for (w, d) in zip(time_weigths, squared_distance)]
                        )
                    elif target["loglikelihood_distri"] == "poisson":
                        for i in range(len(data)):
                            ll += (
                                round(data[i]) * math.log(abs(model_output[i]))
                                - model_output[i]
                                - math.log(math.factorial(round(data[i])))
                            ) * time_weigths[i]
                    elif target["loglikelihood_distri"] == "negative_binomial":
                        assert key + "_dispersion_param" in self.param_list
                        # the dispersion parameter varies during the MCMC. We need to retrieve its value
                        n = [
                            params[i]
                            for i in range(len(params))
                            if self.param_list[i] == key + "_dispersion_param"
                        ][0]
                        for i in range(len(data)):
                            # We use the parameterisation based on mean and variance and assume define var=mean**delta
                            mu = model_output[i]
                            # work out parameter p to match the distribution mean with the model output
                            p = mu / (mu + n)
                            ll += stats.nbinom.logpmf(round(data[i]), n, 1.0 - p) * time_weigths[i]
                    else:
                        raise ValueError("Distribution not supported in loglikelihood_distri")

            if self.run_mode == CalibrationMode.LEAST_SQUARES:
                is_new_best_ll = ll < best_ll
            else:
                is_new_best_ll = ll > best_ll

            if is_new_best_ll:
                best_ll, best_start_time = (ll, considered_start_time)

        if self.run_mode == CalibrationMode.LEAST_SQUARES:
            # FIXME: Store a record of the MCMC run.
            logger.error("No data stored for least squares")
            pass

        self.evaluated_params_ll.append((copy.copy(params), copy.copy(best_ll)))

        if to_return == BEST_LL:
            return best_ll
        elif to_return == BEST_START:
            return best_start_time
        else:
            raise ValueError("to_return not recognised")

    def format_data_as_array(self):
        """
        create a list of data values based on the target outputs
        """
        data = []
        for target in self.targeted_outputs:
            data.append(target["values"])

        data = list(chain(*data))  # create a simple list from a list of lists
        self.data_as_array = np.asarray(data)

    def workout_unspecified_target_sds(self):
        """
        If the sd parameter of the targeted output is not specified, it will automatically be calculated such that the
        95% CI of the associated normal distribution covers 50% of the mean value of the target.
        :return:
        """
        for i, target in enumerate(self.targeted_outputs):
            if "sd" not in target.keys():
                if (
                    "cis" in target.keys()
                ):  # match normal likelihood 95% width with data 95% CI with
                    self.targeted_outputs[i]["sd"] = (
                        target["cis"][0][1] - target["cis"][0][0]
                    ) / 4.0
                else:
                    self.targeted_outputs[i]["sd"] = 0.25 / 4.0 * max(target["values"])

    def workout_unspecified_time_weights(self):
        """
        Will assign a weight to each time point of each calibration target. If no weights were requested, we will use
        1/n for each time point, where n is the number of time points.
        If a list of weights was specified, it will be rescaled so the weights sum to 1.
        """
        for i, target in enumerate(self.targeted_outputs):
            if "time_weights" not in target.keys():
                target["time_weights"] = [1.0 / len(target["years"])] * len(target["years"])
            else:
                assert isinstance(target["time_weights"], list) and len(
                    target["time_weights"]
                ) == len(target["years"])
                s = sum(target["time_weights"])
                target["time_weights"] = [w / s for w in target["time_weights"]]

    def workout_unspecified_jumping_sds(self):
        for i, prior_dict in enumerate(self.priors):
            if "jumping_sd" not in prior_dict.keys():
                if prior_dict["distribution"] == "uniform":
                    prior_width = prior_dict["distri_params"][1] - prior_dict["distri_params"][0]
                elif prior_dict["distribution"] == "lognormal":
                    mu = prior_dict["distri_params"][0]
                    sd = prior_dict["distri_params"][1]
                    quantile_2_5 = math.exp(mu + math.sqrt(2) * sd * special.erfinv(2 * 0.025 - 1))
                    quantile_97_5 = math.exp(mu + math.sqrt(2) * sd * special.erfinv(2 * 0.975 - 1))
                    prior_width = quantile_97_5 - quantile_2_5
                elif prior_dict["distribution"] == "trunc_normal":
                    mu = prior_dict["distri_params"][0]
                    sd = prior_dict["distri_params"][1]
                    bounds = prior_dict["trunc_range"]
                    quantile_2_5 = stats.truncnorm.ppf(
                        0.025, (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
                    )
                    quantile_97_5 = stats.truncnorm.ppf(
                        0.975, (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
                    )
                    prior_width = quantile_97_5 - quantile_2_5
                elif prior_dict["distribution"] == "beta":
                    quantile_2_5 = stats.beta.ppf(
                        0.025,
                        prior_dict["distri_params"][0],
                        prior_dict["distri_params"][1],
                    )
                    quantile_97_5 = stats.beta.ppf(
                        0.975,
                        prior_dict["distri_params"][0],
                        prior_dict["distri_params"][1],
                    )
                    prior_width = quantile_97_5 - quantile_2_5
                elif prior_dict["distribution"] == "gamma":
                    quantile_2_5 = stats.gamma.ppf(
                        0.025,
                        prior_dict["distri_params"][0],
                        0.0,
                        prior_dict["distri_params"][1],
                    )
                    quantile_97_5 = stats.gamma.ppf(
                        0.975,
                        prior_dict["distri_params"][0],
                        0.0,
                        prior_dict["distri_params"][1],
                    )
                    prior_width = quantile_97_5 - quantile_2_5
                else:
                    raise_error_unsupported_prior(prior_dict["distribution"])

                #  95% of the sampled values within [mu - 2*sd, mu + 2*sd], i.e. interval of witdth 4*sd
                relative_prior_width = (
                    0.25  # fraction of prior_width in which 95% of samples should fall
                )
                self.priors[i]["jumping_sd"] = relative_prior_width * prior_width / 4.0

    def run_fitting_algorithm(
        self,
        run_mode=CalibrationMode.AUTUMN_MCMC,
        n_iterations=100,
        n_burned=10,
        n_chains=1,
        available_time=None,
        haario_scaling_factor=2.4,
    ):
        """
        master method to run model calibration.

        :param run_mode: string
            either 'autumn_mcmc' or 'lsm' (for least square minimisation using scipy.minimize function)
        :param n_iterations: number of iterations requested for sampling (excluding burn-in phase)
        :param n_burned: number of burned iterations before effective sampling
        :param n_chains: number of chains to be run
        :param available_time: maximal simulation time allowed (in seconds)
        """
        self.run_mode = run_mode
        if run_mode not in CalibrationMode.MODES:
            msg = f"Requested run mode is not supported. Must be one of {CalibrationMode.MODES}"
            raise ValueError(msg)

        # Initialise random seed differently for different chains
        np.random.seed(get_random_seed(self.chain_index))

        try:
            # Run the selected fitting algorithm.
            if run_mode == CalibrationMode.AUTUMN_MCMC:
                self.run_autumn_mcmc(
                    n_iterations, n_burned, n_chains, available_time, haario_scaling_factor
                )
            elif run_mode == CalibrationMode.LEAST_SQUARES:
                self.run_least_squares()
        finally:
            self.output.write_data_to_disk()

    def run_least_squares(self):
        """
        Run least squares minimization algorithm to calibrate model parameters.
        """
        lower_bounds = []
        upper_bounds = []
        x0 = []
        for prior in self.priors:
            lower_bound, upper_bound = get_parameter_bounds_from_priors(prior)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            if not any([math.isinf(lower_bound), math.isinf(upper_bound)]):
                x0.append(0.5 * (lower_bound + upper_bound))
            elif all([math.isinf(lower_bound), math.isinf(upper_bound)]):
                x0.append(0.0)
            elif math.isinf(lower_bound):
                x0.append(upper_bound)
            else:
                x0.append(lower_bound)
        bounds = Bounds(lower_bounds, upper_bounds)

        sol = minimize(self.loglikelihood, x0, bounds=bounds)
        self.mle_estimates = sol.x

        logger.info("Best solution: %s", self.mle_estimates)

    def test_in_prior_support(self, params):
        in_support = True
        for i, prior_dict in enumerate(self.priors):
            # Work out bounds for acceptable values, using the support of the prior distribution
            lower_bound = self.param_bounds[i, 0]
            upper_bound = self.param_bounds[i, 1]
            if params[i] < lower_bound or params[i] > upper_bound:
                in_support = False
                break

        return in_support

    def run_autumn_mcmc(
        self, n_iterations: int, n_burned: int, n_chains: int, available_time, haario_scaling_factor
    ):
        """
        Run our hand-rolled MCMC algoruthm to calibrate model parameters.
        """
        start_time = time()
        if n_chains > 1:
            msg = "Autumn MCMC method does not support multiple-chain runs at the moment."
            raise ValueError(msg)

        self.mcmc_trace_matrix = None  # will store param trace and loglikelihood evolution

        last_accepted_params = None
        last_accepted_params_trans = None
        last_acceptance_quantity = None  # acceptance quantity is defined as loglike + logprior
        for i_run in range(int(n_iterations + n_burned)):
            # Propose new paramameter set.
            proposed_params_trans = self.propose_new_params_trans(
                last_accepted_params_trans, haario_scaling_factor
            )
            proposed_params = self.get_original_params(proposed_params_trans)

            is_within_prior_support = self.test_in_prior_support(
                proposed_params
            )  # should always be true with transformed params

            if is_within_prior_support:
                # Evaluate log-likelihood.
                proposed_loglike = self.loglikelihood(proposed_params)

                # Evaluate log-prior.
                proposed_logprior = self.logprior(proposed_params)

                # posterior distribution
                proposed_log_posterior = proposed_loglike + proposed_logprior

                # transform the density
                proposed_acceptance_quantity = proposed_log_posterior
                for i, prior_dict in enumerate(
                    self.priors
                ):  # multiply the density with the determinant of the Jacobian
                    proposed_acceptance_quantity += math.log(
                        self.transform[prior_dict["param_name"]]["inverse_derivative"](
                            proposed_params_trans[i]
                        )
                    )

                is_auto_accept = (
                    last_acceptance_quantity is None
                    or proposed_acceptance_quantity >= last_acceptance_quantity
                )
                if is_auto_accept:
                    accept = True
                else:
                    accept_prob = np.exp(proposed_acceptance_quantity - last_acceptance_quantity)
                    accept = (np.random.binomial(n=1, p=accept_prob, size=1) > 0)[0]
            else:
                accept = False
                proposed_loglike = None
                proposed_acceptance_quantity = None

            # Update stored quantities.
            if accept:
                last_accepted_params_trans = proposed_params_trans
                last_accepted_params = proposed_params
                last_acceptance_quantity = proposed_acceptance_quantity

            self.update_mcmc_trace(last_accepted_params_trans)

            # Store model outputs
            self.output.store_mcmc_iteration(
                self.param_list,
                proposed_params,
                proposed_loglike,
                proposed_log_posterior,
                accept,
                i_run,
            )
            if accept:
                self.output.store_model_outputs(self.latest_scenario, self.iter_num)

            self.iter_num += 1
            iters_completed = i_run + 1
            logger.info(f"{iters_completed} MCMC iterations completed.")

            if available_time:
                # Stop iterating if we have run out of time.
                elapsed_time = time() - start_time
                if elapsed_time > available_time:
                    msg = f"Stopping MCMC simulation after {iters_completed} iterations because of {available_time}s time limit"
                    logger.info(msg)
                    break

    def build_adaptive_covariance_matrix(self, haario_scaling_factor):
        scaling_factor = haario_scaling_factor ** 2 / len(self.priors)  # from Haario et al. 2001
        cov_matrix = np.cov(self.mcmc_trace_matrix, rowvar=False)
        adaptive_cov_matrix = scaling_factor * cov_matrix + scaling_factor * ADAPTIVE_METROPOLIS[
            "EPSILON"
        ] * np.eye(len(self.priors))
        return adaptive_cov_matrix

    def get_parameter_bounds(self):
        param_bounds = None
        for i, prior_dict in enumerate(self.priors):
            # Work out bounds for acceptable values, using the support of the prior distribution
            lower_bound, upper_bound = get_parameter_bounds_from_priors(prior_dict)
            if i == 0:
                param_bounds = np.array([[lower_bound, upper_bound]])
            else:
                param_bounds = np.concatenate(
                    (param_bounds, np.array([[lower_bound, upper_bound]]))
                )
        return param_bounds

    def build_transformations(self):
        """
        Build transformation functions between the parameter space and R^n.
        """
        for i, prior_dict in enumerate(self.priors):
            self.transform[prior_dict["param_name"]] = {
                "direct": None,  # param support to R
                "inverse": None,  # R to param space
                "inverse_derivative": None,  # R to R
            }
            lower_bound = self.param_bounds[i, 0]
            upper_bound = self.param_bounds[i, 1]

            original_sd = self.priors[i]["jumping_sd"]  # we will need to transform the jumping step

            # trivial case of an unbounded parameter
            if lower_bound == -float("inf") and upper_bound == float("inf"):
                self.transform[prior_dict["param_name"]]["direct"] = lambda x: x
                self.transform[prior_dict["param_name"]]["inverse"] = lambda x: x
                self.transform[prior_dict["param_name"]]["inverse_derivative"] = lambda x: 1.0

                representative_point = None
            # case of a lower-bounded parameter with infinite support
            elif upper_bound == float("inf"):
                for func_type in ["direct", "inverse", "inverse_derivative"]:
                    self.transform[prior_dict["param_name"]][
                        func_type
                    ] = make_transform_func_with_lower_bound(lower_bound, func_type)
                representative_point = lower_bound + 10 * original_sd
                if self.starting_point[prior_dict["param_name"]] <= lower_bound:
                    self.starting_point[prior_dict["param_name"]] = lower_bound + original_sd / 10

            # case of an upper-bounded parameter with infinite support
            elif lower_bound == -float("inf"):
                for func_type in ["direct", "inverse", "inverse_derivative"]:
                    self.transform[prior_dict["param_name"]][
                        func_type
                    ] = make_transform_func_with_upper_bound(upper_bound, func_type)

                representative_point = upper_bound - 10 * original_sd
                if self.starting_point[prior_dict["param_name"]] >= upper_bound:
                    self.starting_point[prior_dict["param_name"]] = upper_bound - original_sd / 10
            # case of a lower- and upper-bounded parameter
            else:
                for func_type in ["direct", "inverse", "inverse_derivative"]:
                    self.transform[prior_dict["param_name"]][
                        func_type
                    ] = make_transform_func_with_two_bounds(lower_bound, upper_bound, func_type)

                representative_point = 0.5 * (lower_bound + upper_bound)
                if self.starting_point[prior_dict["param_name"]] <= lower_bound:
                    self.starting_point[prior_dict["param_name"]] = lower_bound + original_sd / 10
                elif self.starting_point[prior_dict["param_name"]] >= upper_bound:
                    self.starting_point[prior_dict["param_name"]] = upper_bound - original_sd / 10

            if representative_point is not None:
                transformed_low = self.transform[prior_dict["param_name"]]["direct"](
                    representative_point - original_sd / 2
                )
                transformed_up = self.transform[prior_dict["param_name"]]["direct"](
                    representative_point + original_sd / 2
                )
                self.priors[i]["jumping_sd"] = abs(transformed_up - transformed_low)

    def get_original_params(self, transformed_params):
        original_params = []
        for i, prior_dict in enumerate(self.priors):
            original_params.append(
                self.transform[prior_dict["param_name"]]["inverse"](transformed_params[i])
            )
        return original_params

    def propose_new_params_trans(self, prev_params_trans, haario_scaling_factor=2.4):
        """
        calculated the joint log prior
        :param prev_params: last accepted parameter values as a list ordered using the order of self.priors
        :return: a new list of parameter values
        """
        # prev_params assumed to be the manually calibrated parameters for first step
        if prev_params_trans is None:
            prev_params_trans = []
            for prior_dict in self.priors:
                start_point = self.starting_point[prior_dict["param_name"]]
                prev_params_trans.append(
                    self.transform[prior_dict["param_name"]]["direct"](start_point)
                )
            return prev_params_trans

        new_params_trans = []
        use_adaptive_proposal = (
            self.adaptive_proposal and self.iter_num > ADAPTIVE_METROPOLIS["N_STEPS_FIXED_PROPOSAL"]
        )

        if use_adaptive_proposal:
            adaptive_cov_matrix = self.build_adaptive_covariance_matrix(haario_scaling_factor)
            if np.all((adaptive_cov_matrix == 0)):
                use_adaptive_proposal = (
                    False  # we can't use the adaptive method for this step as the covariance is 0.
                )
            else:
                new_params_trans = sample_from_adaptive_gaussian(
                    prev_params_trans, adaptive_cov_matrix
                )

        if not use_adaptive_proposal:
            for i, prior_dict in enumerate(self.priors):
                sample = np.random.normal(
                    loc=prev_params_trans[i], scale=prior_dict["jumping_sd"], size=1
                )[0]
                new_params_trans.append(sample)

        return new_params_trans

    def logprior(self, params):
        """
        calculated the joint log prior
        :param params: model parameters as a list of values ordered using the order of self.priors
        :return: the natural log of the joint prior
        """
        logp = 0.0
        for i, param_name in enumerate(self.param_list):
            prior_dict = [d for d in self.priors if d["param_name"] == param_name][0]
            logp += calculate_prior(prior_dict, params[i], log=True)

        return logp

    def update_mcmc_trace(self, params_to_store):
        """
        store mcmc iteration into param_trace
        :param params_to_store: model parameters as a list of values ordered using the order of self.priors
        :param loglike_to_store: current loglikelihood value
        """
        if self.mcmc_trace_matrix is None:
            self.mcmc_trace_matrix = np.array([params_to_store])
        else:
            self.mcmc_trace_matrix = np.concatenate(
                (self.mcmc_trace_matrix, np.array([params_to_store]))
            )


class CalibrationOutputs:
    """
    Handles writing outputs for the calibration process
    TODO: Add custom output dir so that we don't need to do silly things in Luigi tasks.
    """

    def __init__(self, chain_id: int, app_name: str, region_name: str):
        self.chain_id = chain_id
        # Track last accepted run for weight calculations.
        self.last_accepted_run = 0
        # List of dicts for tracking MCMC progress.
        self.mcmc_runs = []
        self.mcmc_params = []
        # Model output data - list of DataFrames.
        self.outputs = []
        self.derived_outputs = []

        # Setup output directory
        project_dir = os.path.join(constants.OUTPUT_DATA_PATH, "calibrate", app_name, region_name)
        timestamp = datetime.now().strftime("%Y-%m-%d")
        self.output_dir = os.path.join(project_dir, timestamp)

        db_name = f"calibration-{chain_id}"
        self.output_db_path = os.path.join(self.output_dir, db_name)
        if os.path.exists(self.output_db_path):
            # Delete existing data.
            logger.info("File found at %s, recreating %s", self.output_db_path, self.output_dir)
            shutil.rmtree(self.output_dir)

        logger.info("Created data directory at %s", self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def write_metadata(self, filename, data):
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, "w") as f:
            yaml.dump(data, f)

    def store_model_outputs(self, scenario: Scenario, iter_num: int):
        """
        Record the model outputs for this iteration
        """
        assert scenario, "No model has been run"
        model = scenario.model
        outputs_df = db.store.build_outputs_table([model], run_id=iter_num, chain_id=self.chain_id)
        derived_outputs_df = db.store.build_derived_outputs_table(
            [model], run_id=iter_num, chain_id=self.chain_id
        )
        self.outputs.append(outputs_df)
        self.derived_outputs.append(derived_outputs_df)

    def store_mcmc_iteration(
        self,
        param_list: list,
        proposed_params: list,
        proposed_loglike: float,
        proposed_acceptance_quantity: float,
        accept: bool,
        i_run: int,
    ):
        """
        Records the MCMC iteration details
        :param proposed_params: the current parameter values
        :param proposed_loglike: the current loglikelihood
        :param accept: whether the iteration was accepted or not
        :param i_run: the iteration number
        """
        mcmc_run = {
            "chain": self.chain_id,
            "run": i_run,
            "loglikelihood": proposed_loglike,
            "ap_loglikelihood": proposed_acceptance_quantity,
            "accept": 1 if accept else 0,
            "weight": 0,  # Default to zero, re-calculate this later.
        }
        self.mcmc_runs.append(mcmc_run)
        if accept:
            # Write run parameters.
            for name, value in zip(param_list, proposed_params):
                mcmc_params = {
                    "chain": self.chain_id,
                    "run": i_run,
                    "name": name,
                    "value": value,
                }
                self.mcmc_params.append(mcmc_params)

    def write_data_to_disk(self):
        """
        Write in-memory calibration data to disk
        """
        if not self.mcmc_runs:
            logger.info("No data to write to disk")
            return

        with Timer("Writing calibration data to disk."):
            # try:
            outputs_db = db.FeatherDatabase(self.output_db_path)
            # Consolidate outputs and write to disk
            outputs_df = pd.concat(self.outputs, copy=False, ignore_index=True)
            derived_outputs_df = pd.concat(self.derived_outputs, copy=False, ignore_index=True)
            outputs_db.dump_df(db.store.Table.OUTPUTS, outputs_df)
            outputs_db.dump_df(db.store.Table.DERIVED, derived_outputs_df)
            # Write parameters
            mcmc_params_df = pd.DataFrame.from_dict(self.mcmc_params)
            outputs_db.dump_df(db.store.Table.PARAMS, mcmc_params_df)
            # Calculate iterations weights, then write to disk
            weight = 0
            for mcmc_run in reversed(self.mcmc_runs):
                weight += 1
                if mcmc_run["accept"]:
                    mcmc_run["weight"] = weight
                    weight = 0

            mcmc_runs_df = pd.DataFrame.from_dict(self.mcmc_runs)
            outputs_db.dump_df(db.store.Table.MCMC, mcmc_runs_df)


def get_random_seed(chain_index: int):
    """
    Get a random seed for the calibration.
    Mocked out by unit tests.
    """
    return chain_index * 1000 + int(time())


def get_parameter_bounds_from_priors(prior_dict):
    """
    Determine lower and upper bounds of a parameter by analysing its assigned prior distribution
    :param prior_dict: dictionary defining a parameter's prior distribution
    :return: lower_bound, upper_bound
    """
    if prior_dict["distribution"] == "uniform":
        lower_bound = prior_dict["distri_params"][0]
        upper_bound = prior_dict["distri_params"][1]
    elif prior_dict["distribution"] in ["lognormal", "gamma", "weibull", "exponential"]:
        lower_bound = 0.0
        upper_bound = float("inf")
    elif prior_dict["distribution"] == "trunc_normal":
        lower_bound = prior_dict["trunc_range"][0]
        upper_bound = prior_dict["trunc_range"][1]
    elif prior_dict["distribution"] == "beta":
        lower_bound = 0.0
        upper_bound = 1.0
    else:
        raise ValueError("prior distribution bounds detection currently not handled.")

    return lower_bound, upper_bound


def set_initial_point(priors, model_parameters, chain_index, total_nb_chains, initialisation_type):
    """
    Determine the starting point of the MCMC.
    """
    if initialisation_type == InitialisationTypes.LHS:
        # draw samples using LHS based on the prior distributions
        np.random.seed(0)  # Set deterministic random seed for Latin Hypercube Sampling
        starting_points = sample_starting_params_from_lhs(priors, total_nb_chains)

        return starting_points[chain_index - 1]
    elif initialisation_type == InitialisationTypes.CURRENT_PARAMS:
        # use the current parameter values from the yaml files
        starting_points = {}
        for param_dict in priors:
            if param_dict["param_name"].endswith("dispersion_param"):
                assert param_dict["distribution"] == "uniform"
                starting_points[param_dict["param_name"]] = np.mean(param_dict["distri_params"])
            else:
                starting_points[param_dict["param_name"]] = read_param_value_from_string(
                    model_parameters["default"], param_dict["param_name"]
                )

        return starting_points
    else:
        raise ValueError(f"{initialisation_type} is not a supported Initialisation Type")


def sample_from_adaptive_gaussian(prev_params, adaptive_cov_matrix):
    return np.random.multivariate_normal(prev_params, adaptive_cov_matrix)
