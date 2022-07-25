import logging, math, os, shutil
from datetime import datetime
from itertools import chain
from time import time
from typing import List, Callable
import pickle
from copy import copy

import yaml
import numpy as np
import pandas as pd
from scipy import special, stats
from summer import CompartmentalModel

from autumn import settings
from autumn.core import db, plots
from autumn.core.utils.git import get_git_branch, get_git_hash
from autumn.core.utils.timer import Timer
from autumn.calibration.priors import BasePrior
from autumn.calibration.targets import BaseTarget
from autumn.calibration.proposal_tuning import tune_jumping_stdev
from autumn.core.project.params import read_param_value_from_string
from autumn.core.project import Project, get_project, Params

from .constants import ADAPTIVE_METROPOLIS
from .transformations import (
    make_transform_func_with_lower_bound,
    make_transform_func_with_two_bounds,
    make_transform_func_with_upper_bound,
)
from .utils import (
    calculate_prior,
    raise_error_unsupported_prior,
    sample_starting_params_from_lhs,
    specify_missing_prior_params,
    draw_independent_samples,
)
from .targets import truncnormal_logpdf

ModelBuilder = Callable[[dict,dict], CompartmentalModel]

logger = logging.getLogger(__name__)


class CalibrationMode:
    """Different ways to run the calibration."""

    AUTUMN_MCMC = "autumn_mcmc"
    MODES = [AUTUMN_MCMC]


class MetroInit:
    """Different ways to set the intial point for the MCMC."""

    LHS = "lhs"
    CURRENT_PARAMS = "current_params"


# Multiplier scaling the covariance matrix in the Haario Metropolis. 2.4 is the value recommended by Haario.
# Greater values increase jumping step size and reduce the acceptance ratio
DEFAULT_HAARIO_SCALING_FACTOR = 2.4

DEFAULT_METRO_INIT = MetroInit.CURRENT_PARAMS
DEFAULT_METRO_STEP = 0.1

DEFAULT_STEPS = 50


class Calibration:
    """
    Handles model calibration.

    If sampling from the posterior distribution is required, uses a Bayesian algorithm.
    If only one calibrated parameter set is required, uses maximum likelihood estimation.

    A Metropolis Hastings algorithm is used with or without adaptive proposal function.
    The adaptive approach employed was published by Haario et al.

        'An  adaptive  Metropolis  algorithm', Bernoulli 7(2), 2001, 223-242

    """

    def __init__(
        self,
        priors: List[BasePrior],
        targets: List[BaseTarget],
        haario_scaling_factor: float = DEFAULT_HAARIO_SCALING_FACTOR,
        adaptive_proposal: bool = True,
        metropolis_init: str = DEFAULT_METRO_INIT,
        metropolis_init_rel_step_size: float = DEFAULT_METRO_STEP,
        fixed_proposal_steps: int = DEFAULT_STEPS,
        seed: int = None,
        initial_jumping_stdev_ratio: float = 0.25,
        jumping_stdev_adjustment: float = 0.5,
        random_process=None,
        hierarchical_priors: list = []
    ):
        """
        Defines a new calibration.
        """
        check_hierarchical_priors(hierarchical_priors, priors)
        self.hierarchical_priors = hierarchical_priors
        self.all_priors = [p.to_dict() for p in priors] + [h_p.to_dict() for h_p in hierarchical_priors]

        self.includes_random_process = False
        if random_process is not None:
            self.random_process = random_process
            self.set_up_random_process()
        
        #self.targets = [t.to_dict() for t in targets]
        self.targets = remove_early_points_to_prevent_crash(targets, self.all_priors)

        self.haario_scaling_factor = haario_scaling_factor
        self.adaptive_proposal = adaptive_proposal
        self.initialisation_type = metropolis_init
        self.metropolis_init_rel_step_size = metropolis_init_rel_step_size
        self.n_steps_fixed_proposal = fixed_proposal_steps
        self.initial_jumping_stdev_ratio = initial_jumping_stdev_ratio
        self.jumping_stdev_adjustment = jumping_stdev_adjustment

        self.split_priors_by_type()

        if seed is None:
            seed = int(time())
        self.seed = seed

        # Set this to True for mock tests that have trouble with pickling
        self._no_pickle = False

    @staticmethod
    def from_existing(pkl_file, output_dir):
        obj = pickle.load(open(pkl_file, 'rb'))
        obj.output = CalibrationOutputs.from_existing(obj.chain_idx, output_dir)
        return obj

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['transform']
        del state['project']
        del state['output']

        # Probably can't pickle models...
        state['latest_model'] = None

        # These are items that are not members of the class/object dictionary,
        # but are still required for restoring state
        state['_extra'] = {}
        state['_extra']['project'] = {'model_name': self.project.model_name, 'project_name': self.project.region_name}
        state['_extra']['rng'] = np.random.get_state()

        return state

    def __setstate__(self, state):

        # These are items that are not members of the class/object dictionary,
        # but are still required for restoring state
        _extra = state.pop('_extra')

        self.__dict__.update(state)
        self.project = get_project(**_extra['project'])
        self.build_transformations(update_jumping_stdev=False)
        np.random.set_state(_extra['rng'])

        #self.output = CalibrationOutputs.open_existing(self.chain_idx, state[])

    

    def set_up_random_process(self):
        self.includes_random_process = True

        # add priors for coefficients, using 80% weight for the first order, splitting remaining 20% between remaining orders
        # only relevant if order > 1
        order = self.random_process.order
        if order > 1:
            coeff_means = [.8] + [.2 / (order - 1)] * (order - 1)
            for i, coeff_mean in enumerate(coeff_means):
                self.all_priors.append({
                    "param_name":  f"random_process.coefficients({i})",
                    "distribution": "trunc_normal",
                    "distri_params": [coeff_mean, 0.05],
                    "trunc_range": [0., 1.],
                })

        # add prior for noise sd
        self.all_priors.append({
            "param_name": "random_process.noise_sd",
            "distribution": "uniform",
            "distri_params": [0.01, 1.], 
        })

        # add priors for rp values
        n_values = len(self.random_process.values)
        self.all_priors += [
            {
                "param_name": f"random_process.values({i_val})",
                "distribution": "uniform",
                "distri_params": [-2., 2.],
                "skip_evaluation": True
            } for i_val in range(1, n_values)  # the very first value will be fixed to 0.
        ]

    def split_priors_by_type(self):
        # Distinguish independent sampling parameters from standard (iteratively sampled) calibration parameters
        independent_sample_idxs = [
            idx for idx in range(len(self.all_priors)) if self.all_priors[idx].get("sampling") == "lhs"
        ]

        self.iterative_sampling_priors = [
            param_dict
            for i_param, param_dict in enumerate(self.all_priors)
            if i_param not in independent_sample_idxs
        ]
        self.independent_sampling_priors = [
            param_dict
            for i_param, param_dict in enumerate(self.all_priors)
            if i_param in independent_sample_idxs
        ]
        self.iterative_sampling_param_names = [
            self.iterative_sampling_priors[i]["param_name"]
            for i in range(len(self.iterative_sampling_priors))
        ]
        self.independent_sampling_param_names = [
            self.independent_sampling_priors[i]["param_name"]
            for i in range(len(self.independent_sampling_priors))
        ]

    def tune_proposal(self, param_name, project: Project, n_points=100, relative_likelihood_reduction=0.5):
        assert param_name in self.iterative_sampling_param_names, f"{param_name} is not an iteratively sampled parameter"
        assert n_points > 1, "A minimum of two points is required to perform proposal tuning"

        self._is_first_run = True

        # We must perform a few initialisation tasks (needs refactoring)
        # work out missing distribution params for priors
        specify_missing_prior_params(self.iterative_sampling_priors)
        specify_missing_prior_params(self.independent_sampling_priors)

        # rebuild self.all_priors, following changes to the two sets of priors
        self.all_priors = self.iterative_sampling_priors + self.independent_sampling_priors

        self.project = project
        self.model_parameters = project.param_set.baseline
        self.end_time = 2 + max([max(t.data.index) for t in self.targets])
        target_names = [t.data.name for t in self.targets]
        self.derived_outputs_whitelist = list(set(target_names))
        self.run_mode = CalibrationMode.AUTUMN_MCMC
        self.workout_unspecified_target_sds()  # for likelihood definition
        self.workout_unspecified_time_weights()  # for likelihood weighting

        prior_dict = [p_dict for p_dict in self.all_priors if p_dict["param_name"] == param_name][0]
        lower_bound, upper_bound = get_parameter_finite_range_from_prior(prior_dict)

        starting_point = read_current_parameter_values(self.all_priors, self.model_parameters.to_dict())

        eval_points = list(np.linspace(start=lower_bound, stop=upper_bound, num=n_points, endpoint=True))
        eval_log_postertiors = []
        for i_run, eval_point in enumerate(eval_points):
            self.run_num = i_run
            update = {param_name: eval_point}
            updated_params = {**starting_point, **update}
            log_likelihood = self.loglikelihood(updated_params)
            log_prior = self.logprior(updated_params)
            eval_log_postertiors.append(log_likelihood + log_prior)
        return tune_jumping_stdev(eval_points, eval_log_postertiors, relative_likelihood_reduction)

    def run(
        self,
        project: Project,
        max_seconds: float,
        chain_idx: int,
        num_chains: int,
        derived_outputs_to_plot: List[str] = None,
    ):
        self.project = project
        self.model_parameters = project.param_set.baseline
        self.chain_idx = chain_idx
        model_parameters_data = self.model_parameters.to_dict()

        # 

        # Figure out which derived outputs we have to calculate.
        derived_outputs_to_plot = derived_outputs_to_plot or []
        target_names = [t.data.name for t in self.targets]
        self.derived_outputs_whitelist = list(set(target_names + derived_outputs_to_plot))

        # Validate target output start time.
        self.validate_target_start_time(model_parameters_data)

        # Set a custom end time for all model runs - there is no point running
        # the models after the last calibration targets.
        self.end_time = 2 + max([max(t.data.index) for t in self.targets])

        # work out missing distribution params for priors
        specify_missing_prior_params(self.iterative_sampling_priors)
        specify_missing_prior_params(self.independent_sampling_priors)

        # rebuild self.all_priors, following changes to the two sets of priors
        self.all_priors = self.iterative_sampling_priors + self.independent_sampling_priors

        # initialise hierarchical priors' parameters
        self.update_hierarchical_prior_params(self.model_parameters)

        # Select starting params
        # Random seed is reset in here; make sure any other seeding happens after this
        self.starting_point = set_initial_point(
            self.all_priors, model_parameters_data, chain_idx, num_chains, self.initialisation_type
        )

        # Set chain specific seed
        # Chain 0 will have seed equal to that set in the calibration initialisation
        self.seed_chain = chain_idx * 1000 + self.seed

        # initialise output and save metadata
        self.output = CalibrationOutputs(chain_idx, project.model_name, project.region_name)
        self.save_metadata(chain_idx, project, model_parameters_data)

        self.workout_unspecified_target_sds()  # for likelihood definition
        self.workout_unspecified_time_weights()  # for likelihood weighting
        self.workout_unspecified_jumping_stdevs()  # for proposal function definition
        self.param_bounds = self.get_parameter_bounds()

        self.build_transformations()

        self.latest_model = None
        self.mcmc_trace_matrix = None  # will store the results of the MCMC model calibration

        if self.chain_idx == 0:
            plots.calibration.plot_pre_calibration(self.all_priors, self.output.output_dir)

        self.is_vic_super_model = False
        if "victorian_clusters" in model_parameters_data:
            if model_parameters_data["victorian_clusters"]:
                self.is_vic_super_model = True

        # Set up a flag so that we run a full model validation the first iteration,
        # but disable for subsequent iterations
        self._is_first_run = True

        # Actually run the calibration
        self.run_fitting_algorithm(
            run_mode=CalibrationMode.AUTUMN_MCMC,
            n_chains=num_chains,
            available_time=max_seconds,
        )

    def update_hierarchical_prior_params(self, current_params=None):
        for h_p in self.hierarchical_priors:
            # work out hyper-parameter values
            distri_params = copy(h_p.hyper_parameters)
            for i, p in enumerate(distri_params):
                if isinstance(p, str):
                    if isinstance(current_params, Params):
                        distri_params[i] = current_params[p]
                    else:
                        param_index = [par['param_name'] for par in self.all_priors].index(p)
                        distri_params[i] = current_params[param_index]
            
            # update prior lists
            for prior in self.all_priors:
                if prior["param_name"] == h_p.name:
                    prior["distri_params"] = distri_params
                    break    

            for prior in self.iterative_sampling_priors:
                if prior["param_name"] == h_p.name:
                    prior["distri_params"] = distri_params
                    break   

    def validate_target_start_time(self, model_parameters_data):
        model_start = model_parameters_data["time"]["start"]
        max_prior_start = None
        for p in self.all_priors:
            if p["param_name"] == "time.start":
                max_prior_start = max(p["distri_params"])

        for t in self.targets:
            t_name = t.data.name
            min_t = min(t.data.index)
            msg = f"Target {t_name} has time {min_t} before model start {model_start}."
            assert min_t >= model_start, msg
            if max_prior_start:
                msg = f"Target {t_name} has time {min_t} before prior start {max_prior_start}."
                assert min_t >= max_prior_start, msg

    def save_metadata(self, chain_idx, project, model_parameters_data):
        metadata = {
            "app_name": project.model_name,
            "region_name": project.region_name,
            "start_time": datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
            "git_branch": get_git_branch(),
            "git_commit": get_git_hash(),
            "seed_chain": self.seed_chain,
            "seed": self.seed
        }
        self.output.write_metadata(f"meta-{chain_idx}.yml", metadata)
        self.output.write_metadata(f"params-{chain_idx}.yml", model_parameters_data)
        self.output.write_metadata(f"priors-{chain_idx}.yml", self.all_priors)
        self.output.write_metadata(f"targets-{chain_idx}.yml", self.targets)

    def run_model_with_params(self, proposed_params: dict):
        """
        Run the model with a set of params.
        """
        logger.info(f"Running iteration {self.run_num}...")
        # Update default parameters to use calibration params.
        param_updates = {"time.end": self.end_time}
        for param_name, value in proposed_params.items():
            param_updates[param_name] = value
        iter_params = self.model_parameters.update(param_updates, calibration_format=True)

        # Update the random_process attribute with the current rp config for later likelihood evaluation
        if self.includes_random_process:
            if self.random_process.order > 1:
                self.random_process.coefficients = [proposed_params[f"random_process.coefficients({i})"] for i in range(self.random_process.order)]
            self.random_process.noise_sd = proposed_params["random_process.noise_sd"]
            self.random_process.values = [0.] + [proposed_params[f"random_process.values({k})"] for k in range(1, len(self.random_process.values))]

        if self._is_first_run:
            self.build_options = dict(enable_validation = True)

        self.latest_model = self.project.run_baseline_model(
            iter_params, derived_outputs_whitelist=self.derived_outputs_whitelist,
            build_options = self.build_options
        )
        
        if self._is_first_run:
            self._is_first_run = False
            self.build_options['enable_validation'] = False
            self.build_options['derived_outputs_idx_cache'] = self.latest_model._derived_outputs_idx_cache

        return self.latest_model

    def loglikelihood(self, all_params_dict):
        """
        Calculate the loglikelihood for a set of parameters
        """
        model = self.run_model_with_params(all_params_dict)

        ll = 0  # loglikelihood if using bayesian approach.
        for target in self.targets:
            key = target.data.name
            data = target.data.to_numpy()
            time_weights = target.time_weights
            indices = []
            for t in target.data.index:
                time_idxs = np.where(model.times == t)[0]
                time_idx = time_idxs[0]
                indices.append(time_idx)

            model_output = model.derived_outputs[key][indices]
            if self.run_mode == CalibrationMode.AUTUMN_MCMC:
                if target.loglikelihood_distri in ["normal", "trunc_normal"]:
                    # Retrieve the value of the standard deviation
                    if key + "_dispersion_param" in all_params_dict:
                        normal_sd = all_params_dict[key + "_dispersion_param"]
                    elif "target_output_ratio" in all_params_dict:
                        normal_sd = all_params_dict["target_output_ratio"] * max(target.data)
                    else:
                        normal_sd = target.stdev

                    if target.loglikelihood_distri == "normal":
                        squared_distance = (data - model_output) ** 2
                        ll += -(0.5 / normal_sd ** 2) * np.sum(
                            [w * d for (w, d) in zip(time_weights, squared_distance)]
                        )
                    else:  # this is a truncated normal likelihood
                        logpdf_arr =  truncnormal_logpdf(data, model_output, target.trunc_range, normal_sd)
                        ll += (logpdf_arr * time_weights).sum()
                elif target.loglikelihood_distri == "poisson":
                    for i in range(len(data)):
                        ll += (
                            round(data[i]) * math.log(abs(model_output[i]))
                            - model_output[i]
                            - math.log(math.factorial(round(data[i])))
                        ) * time_weights[i]
                elif target.loglikelihood_distri == "negative_binomial":
                    if key + "_dispersion_param" in all_params_dict:
                        # the dispersion parameter varies during the MCMC. We need to retrieve its value
                        n = all_params_dict[key + "_dispersion_param"]
                    elif target.dispersion_param is not None:
                        n = target.dispersion_param
                    else:
                        raise ValueError(f"A dispersion_param is required for target {key}")

                    for i in range(len(data)):
                        # We use the parameterisation based on mean and variance and assume define var=mean**delta
                        mu = model_output[i]
                        # work out parameter p to match the distribution mean with the model output
                        p = mu / (mu + n)
                        ll += stats.nbinom.logpmf(round(data[i]), n, 1.0 - p) * time_weights[i]
                else:
                    raise ValueError("Distribution not supported in loglikelihood_distri")

        return ll

    def workout_unspecified_target_sds(self):
        """
        If the sd parameter of the targeted output is not specified, it will be calculated automatically such that the
        95% CI of the associated normal distribution covers a width equivalent to 25% of the maximum value of the target.
        :return:
        """
        for i, target in enumerate(self.targets):
            if target.stdev is None:
                if (
                    # Do we ever use this?  Doesn't show up anywhere in the codebase..
                    target.cis is not None
                ):  # match normal likelihood 95% width with data 95% CI with
                # +++ This will crash, but we should rewrite it when it does (Romain to explain), since this is very opaque right now...
                    target.stdev = (
                        target["cis"][0][1] - target["cis"][0][0]
                    ) / 4.0
                else:
                    target.stdev = 0.25 / 4.0 * max(target.data)

    def workout_unspecified_time_weights(self):
        """
        Will assign a weight to each time point of each calibration target. If no weights were requested, we will use
        1/n for each time point, where n is the number of time points.
        If a list of weights was specified, it will be rescaled so the weights sum to 1.
        """
        for i, target in enumerate(self.targets):
            if target.time_weights is None:
                target.time_weights = np.ones(len(target.data)) / len(target.data)
            else:
                assert len(target.time_weights) == len(target.data)
                s = sum(target.time_weights)
                target.time_weights = target.time_weights / s

    def workout_unspecified_jumping_stdevs(self):
        for i, prior_dict in enumerate(self.iterative_sampling_priors):
            if "jumping_stdev" not in prior_dict.keys():
                prior_low, prior_high = get_parameter_finite_range_from_prior(prior_dict)
                prior_width = prior_high - prior_low

                #  95% of the sampled values within [mu - 2*sd, mu + 2*sd], i.e. interval of witdth 4*sd
                relative_prior_width = (
                    self.metropolis_init_rel_step_size  # fraction of prior_width in which 95% of samples should fall
                )
                self.iterative_sampling_priors[i]["jumping_stdev"] = (
                    relative_prior_width * prior_width * self.initial_jumping_stdev_ratio
                )

    def run_fitting_algorithm(
        self,
        run_mode: str,
        n_chains=1,
        available_time=None,
    ):
        """
        master method to run model calibration.

        :param run_mode: string
            only 'autumn_mcmc' is currently supported
        :param n_chains: number of chains to be run
        :param available_time: maximal simulation time allowed (in seconds)
        """
        self.run_mode = run_mode
        if run_mode not in CalibrationMode.MODES:
            msg = f"Requested run mode is not supported. Must be one of {CalibrationMode.MODES}"
            raise ValueError(msg)

        # Initialise random seed differently for different chains
        np.random.seed(self.seed_chain)

        try:
            # Run the selected fitting algorithm.
            if run_mode == CalibrationMode.AUTUMN_MCMC:
                self.run_autumn_mcmc(available_time)

        finally:
            self.write_outputs()

    def write_outputs(self):
        """Ensure output data from run is written to disk, including model state for resume
        """
        self.output.write_data_to_disk()
        if not self._no_pickle:
            state_pkl_filename = os.path.join(self.output.output_dir, f"calstate-{self.chain_idx}.pkl")
            pickle.dump(self, open(state_pkl_filename, 'wb'))

    def test_in_prior_support(self, iterative_params):
        in_support = True
        for i, prior_dict in enumerate(self.iterative_sampling_priors):
            param_name = prior_dict["param_name"]
            # Work out bounds for acceptable values, using the support of the prior distribution
            lower_bound = self.param_bounds[param_name][0]
            upper_bound = self.param_bounds[param_name][1]
            if iterative_params[i] < lower_bound or iterative_params[i] > upper_bound:
                in_support = False
                break

        return in_support

    def run_autumn_mcmc(
        self,
        available_time
    ):
        """
        Run our hand-rolled MCMC algorithm to calibrate model parameters.
        """

        self.mcmc_trace_matrix = None  # will store param trace and loglikelihood evolution

        self.last_accepted_iterative_params_trans = None
        self.last_acceptance_quantity = None  # acceptance quantity is defined as loglike + logprior
        self.n_accepted = 0
        self.n_iters_real = 0  # Actual number of iterations completed, as opposed to run_num.
        self.run_num = 0  # Canonical id of the MCMC run, will be the same as iters until reset by adaptive algo.
    
        self.enter_mcmc_loop(available_time)

    def resume_autumn_mcmc(self, available_time: int = None, max_iters: int = None, finalise=True):
        try:
            self.enter_mcmc_loop(available_time, max_iters)
        finally:
            if finalise:
                self.write_outputs()
            
    def enter_mcmc_loop(self, available_time: int = None, max_iters: int = None):
        start_time = time()

        if max_iters:
            if self.n_iters_real >= max_iters:
                msg = f"Not resuming run. Existing run already has {self.n_iters_real} iterations; max_iters = {max_iters}"
                logger.info(msg)
                return

        while True:
            logging.info("Running MCMC iteration %s, run %s", self.n_iters_real, self.run_num)

            # Not actually LHS sampling - just sampling directly from prior.
            independent_samples = draw_independent_samples(self.independent_sampling_priors)

            # Propose new parameter set.
            proposed_iterative_params_trans = self.propose_new_iterative_params_trans(
                self.last_accepted_iterative_params_trans, self.haario_scaling_factor
            )
            proposed_iterative_params = self.get_original_params(proposed_iterative_params_trans)

            self.update_hierarchical_prior_params(proposed_iterative_params)

            is_within_prior_support = self.test_in_prior_support(
                proposed_iterative_params
            )  # should always be true but this is a good safety check

            # combine all sampled params into a single dictionary
            iterative_samples_dict = {
                self.iterative_sampling_param_names[i]: proposed_iterative_params[i]
                for i in range(len(proposed_iterative_params))
            }
            all_params_dict = {**iterative_samples_dict, **independent_samples}

            if is_within_prior_support:
                # Evaluate log-likelihood.
                proposed_loglike = self.loglikelihood(all_params_dict)

                # Evaluate log-prior.
                proposed_logprior = self.logprior(all_params_dict)

                # Evaluate the log-likelihood of the random process if applicable
                if self.includes_random_process:
                    proposed_logprior += self.random_process.evaluate_rp_loglikelihood()

                # posterior distribution
                proposed_log_posterior = proposed_loglike + proposed_logprior

                # transform the density
                proposed_acceptance_quantity = proposed_log_posterior
                
                # for i, prior_dict in enumerate(
                #     self.iterative_sampling_priors
                # ):  # multiply the density with the determinant of the Jacobian
                #     inv_derivative = self.transform[prior_dict["param_name"]]["inverse_derivative"](
                #         proposed_iterative_params_trans[i]
                #     )
                #     if inv_derivative > 0:
                #         proposed_acceptance_quantity += math.log(inv_derivative)
                #     else:
                #         proposed_acceptance_quantity += math.log(1.0e-100)

                is_auto_accept = (
                    self.last_acceptance_quantity is None
                    or proposed_acceptance_quantity >= self.last_acceptance_quantity
                )
                if is_auto_accept:
                    accept = True
                else:
                    accept_prob = np.exp(proposed_acceptance_quantity - self.last_acceptance_quantity)
                    accept = (np.random.binomial(n=1, p=accept_prob, size=1) > 0)[0]
            else:
                accept = False
                proposed_loglike = None
                proposed_acceptance_quantity = None

            # Update stored quantities.
            if accept:
                self.last_accepted_iterative_params_trans = proposed_iterative_params_trans
                self.last_acceptance_quantity = proposed_acceptance_quantity
                self.n_accepted += 1

            self.update_mcmc_trace(self.last_accepted_iterative_params_trans)

            # Store model outputs
            self.output.store_mcmc_iteration(
                all_params_dict,
                proposed_loglike,
                proposed_log_posterior,
                accept,
                self.run_num,
            )
            if accept:
                self.output.store_model_outputs(self.latest_model, self.run_num)

            logging.info("Finished MCMC iteration %s, run %s", self.n_iters_real, self.run_num)
            self.run_num += 1
            self.n_iters_real += 1
            if available_time:
                # Stop iterating if we have run out of time.
                elapsed_time = time() - start_time
                if elapsed_time > available_time:
                    msg = f"Stopping MCMC simulation after {self.n_iters_real} iterations because of {available_time}s time limit"
                    logger.info(msg)
                    break
            if max_iters:
                # Stop running if we have performed enough iterations
                if self.n_iters_real >= max_iters:
                    msg = f"Stopping MCMC simulation after {self.n_iters_real} iterations, maximum iterations hit"
                    logger.info(msg)
                    break

            # Check that the pre-adaptive phase ended with a decent acceptance ratio
            if self.adaptive_proposal and self.run_num == self.n_steps_fixed_proposal:
                acceptance_ratio = self.n_accepted / self.run_num
                logger.info(
                    "Pre-adaptive phase completed at %s iterations after %s runs with an acceptance ratio of %s.",
                    self.n_iters_real,
                    self.run_num,
                    acceptance_ratio,
                )
                if acceptance_ratio < ADAPTIVE_METROPOLIS["MIN_ACCEPTANCE_RATIO"]:
                    logger.info("Acceptance ratio too low, restart sampling from scratch.")
                    (
                        self.run_num,
                        self.n_accepted,
                        self.last_accepted_params_trans,
                        self.last_acceptance_quantity,
                    ) = (0, 0, None, None)
                    self.reduce_proposal_step_size()
                    self.output.delete_stored_iterations()
                else:
                    logger.info("Acceptance ratio acceptable, continue sampling.")

    def reduce_proposal_step_size(self):
        """
        Reduce the "jumping_stdev" associated with each parameter during the pre-adaptive phase
        """
        for i in range(len(self.iterative_sampling_priors)):
            self.iterative_sampling_priors[i]["jumping_stdev"] *= self.jumping_stdev_adjustment

    def build_adaptive_covariance_matrix(self, haario_scaling_factor):
        scaling_factor = haario_scaling_factor ** 2 / len(
            self.iterative_sampling_priors
        )  # from Haario et al. 2001
        cov_matrix = np.cov(self.mcmc_trace_matrix, rowvar=False)
        adaptive_cov_matrix = scaling_factor * cov_matrix + scaling_factor * ADAPTIVE_METROPOLIS[
            "EPSILON"
        ] * np.eye(len(self.iterative_sampling_priors))
        return adaptive_cov_matrix

    def get_parameter_bounds(self):
        param_bounds = {}
        for i, prior_dict in enumerate(
            self.iterative_sampling_priors + self.independent_sampling_priors
        ):
            # Work out bounds for acceptable values, using the support of the prior distribution
            lower_bound, upper_bound = get_parameter_bounds_from_priors(prior_dict)
            param_bounds[prior_dict["param_name"]] = [lower_bound, upper_bound]

        return param_bounds

    def build_transformations(self, update_jumping_stdev=True):
        """
        Build transformation functions between the parameter space and R^n.
        """
        self.transform = {}
        for i, prior_dict in enumerate(self.iterative_sampling_priors):
            param_name = prior_dict["param_name"]
            self.transform[param_name] = {
                "direct": None,  # param support to R
                "inverse": None,  # R to param space
                "inverse_derivative": None,  # R to R
            }
            lower_bound = self.param_bounds[param_name][0]
            upper_bound = self.param_bounds[param_name][1]

            original_sd = self.iterative_sampling_priors[i][
                "jumping_stdev"
            ]  # we will need to transform the jumping step

            # trivial case of an unbounded parameter
            if lower_bound == -float("inf") and upper_bound == float("inf"):
                self.transform[param_name]["direct"] = lambda x: x
                self.transform[param_name]["inverse"] = lambda x: x
                self.transform[param_name]["inverse_derivative"] = lambda x: 1.0

                representative_point = None
            # case of a lower-bounded parameter with infinite support
            elif upper_bound == float("inf"):
                for func_type in ["direct", "inverse", "inverse_derivative"]:
                    self.transform[param_name][func_type] = make_transform_func_with_lower_bound(
                        lower_bound, func_type
                    )
                representative_point = lower_bound + 10 * original_sd
                if self.starting_point[param_name] <= lower_bound:
                    self.starting_point[param_name] = lower_bound + original_sd / 10

            # case of an upper-bounded parameter with infinite support
            elif lower_bound == -float("inf"):
                for func_type in ["direct", "inverse", "inverse_derivative"]:
                    self.transform[param_name][func_type] = make_transform_func_with_upper_bound(
                        upper_bound, func_type
                    )

                representative_point = upper_bound - 10 * original_sd
                if self.starting_point[param_name] >= upper_bound:
                    self.starting_point[param_name] = upper_bound - original_sd / 10
            # case of a lower- and upper-bounded parameter
            else:
                for func_type in ["direct", "inverse", "inverse_derivative"]:
                    self.transform[param_name][func_type] = make_transform_func_with_two_bounds(
                        lower_bound, upper_bound, func_type
                    )

                representative_point = 0.5 * (lower_bound + upper_bound)
                if self.starting_point[param_name] <= lower_bound:
                    self.starting_point[param_name] = lower_bound + original_sd / 10
                elif self.starting_point[param_name] >= upper_bound:
                    self.starting_point[param_name] = upper_bound - original_sd / 10

            # Don't update jumping if we are resuming (this has already been calculated)
            # FIXME:  We should probably refactor this to update on copies rather than in place
            if representative_point is not None and update_jumping_stdev:
                transformed_low = self.transform[param_name]["direct"](
                    representative_point - original_sd / 4
                )
                transformed_up = self.transform[param_name]["direct"](
                    representative_point + original_sd / 4
                )
                self.iterative_sampling_priors[i]["jumping_stdev"] = abs(
                    transformed_up - transformed_low
                )

    def get_original_params(self, transformed_iterative_params):
        original_iterative_params = []
        for i, prior_dict in enumerate(self.iterative_sampling_priors):
            original_iterative_params.append(
                self.transform[prior_dict["param_name"]]["inverse"](transformed_iterative_params[i])
            )
        return original_iterative_params

    def propose_new_iterative_params_trans(
        self, prev_iterative_params_trans, haario_scaling_factor=2.4
    ):
        """
        calculated the joint log prior
        :param prev_iterative_params_trans: last accepted parameter values as a list ordered using the order of
         self.iterative_sampling_priors
        :return: a new list of parameter values
        """
        new_iterative_params_trans = []
        # if this is the initial step
        if prev_iterative_params_trans is None:
            for prior_dict in self.iterative_sampling_priors:
                start_point = self.starting_point[prior_dict["param_name"]]
                new_iterative_params_trans.append(
                    self.transform[prior_dict["param_name"]]["direct"](start_point)
                )
            return new_iterative_params_trans

        use_adaptive_proposal = (
            self.adaptive_proposal and self.run_num > self.n_steps_fixed_proposal
        )

        if use_adaptive_proposal:
            adaptive_cov_matrix = self.build_adaptive_covariance_matrix(haario_scaling_factor)
            if np.all((adaptive_cov_matrix == 0)):
                use_adaptive_proposal = (
                    False  # we can't use the adaptive method for this step as the covariance is 0.
                )
            else:
                new_iterative_params_trans = sample_from_adaptive_gaussian(
                    prev_iterative_params_trans, adaptive_cov_matrix
                )

        if not use_adaptive_proposal:
            for i, prior_dict in enumerate(self.iterative_sampling_priors):
                sample = np.random.normal(
                    loc=prev_iterative_params_trans[i], scale=prior_dict["jumping_stdev"], size=1
                )[0]
                new_iterative_params_trans.append(sample)

        return new_iterative_params_trans

    def logprior(self, all_params_dict):
        """
        calculated the joint log prior
        :param all_params_dict: model parameters as a dictionary
        :return: the natural log of the joint prior
        """
        logp = 0.0
        for param_name, value in all_params_dict.items():
            prior_dict = [d for d in self.all_priors if d["param_name"] == param_name][0]
            if "skip_evaluation" in prior_dict:
                if prior_dict["skip_evaluation"]:
                    continue
            logp += calculate_prior(prior_dict, value, log=True)

        return logp

    def update_mcmc_trace(self, params_to_store):
        """
        store mcmc iteration into param_trace
        :param params_to_store: model parameters as a list of values ordered using the order of self.iterative_sampling_priors
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
    """

    def __init__(self, chain_id: int, app_name: str, region_name: str):
        self.chain_id = chain_id
        # List of dicts for tracking MCMC progress.
        self.mcmc_runs = []
        self.mcmc_params = []

        # Setup output directory
        project_dir = os.path.join(settings.OUTPUT_DATA_PATH, "calibrate", app_name, region_name)
        timestamp = datetime.now().strftime("%Y-%m-%d")
        # A bit of a hack to write to a different directory when running jobs in AWS.
        self.output_dir = os.environ.get(
            "AUTUMN_CALIBRATE_DIR", os.path.join(project_dir, timestamp)
        )
        db_name = f"chain-{chain_id}"
        self.output_db_path = os.path.join(self.output_dir, db_name)
        if os.path.exists(self.output_db_path):
            # Delete existing data.
            logger.info("File found at %s, recreating %s", self.output_db_path, self.output_dir)
            shutil.rmtree(self.output_dir)

        logger.info("Created data directory at %s", self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.db = db.ParquetDatabase(self.output_db_path)

    @classmethod
    def from_existing(cls, chain_id, output_dir):
        obj = cls.__new__(cls)
        obj.output_dir = output_dir

        db_name = f"chain-{chain_id}"

        obj.output_db_path = os.path.join(obj.output_dir, db_name)
        obj.db = db.ParquetDatabase(obj.output_db_path)

        obj.chain_id = chain_id

        # List of dicts for tracking MCMC progress.
        #obj.mcmc_runs = []
        #obj.mcmc_params = []
        obj.load_mcmc()

        return obj

    def load_mcmc(self):
        """Read MCMC calibration data from disk (for resuming an existing run)
        """
        self.mcmc_runs = self.db.query('mcmc_run').to_dict('records')
        self.mcmc_params = self.db.query('mcmc_params').to_dict('records')

    def write_metadata(self, filename, data):
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, "w") as f:
            yaml.dump(data, f)

    def delete_stored_iterations(self):
        self.db.close()
        self.db.delete_everything()
        self.mcmc_runs = []
        self.mcmc_params = []

    def store_model_outputs(self, model, iter_num: int):
        """
        Record the model outputs for this iteration
        """
        assert model and model.outputs is not None, "No model has been run"
        #outputs_df = db.store.build_outputs_table([model], run_id=iter_num, chain_id=self.chain_id)
        derived_outputs_df = db.store.build_derived_outputs_table(
            [model], run_id=iter_num, chain_id=self.chain_id
        )
        #self.db.append_df(db.store.Table.OUTPUTS, outputs_df)
        self.db.append_df(db.store.Table.DERIVED, derived_outputs_df)

    def store_mcmc_iteration(
        self,
        all_params_dict: dict,
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
            for param_name, value in all_params_dict.items():
                mcmc_params = {
                    "chain": self.chain_id,
                    "run": i_run,
                    "name": param_name,
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

        # Close Parquet writer used to write data for outputs / derived outputs.
        self.db.close()

        with Timer("Writing calibration data to disk."):
            # Write parameters
            mcmc_params_df = pd.DataFrame.from_dict(self.mcmc_params)
            self.db.dump_df(db.store.Table.PARAMS, mcmc_params_df, append=False)
            # Calculate iterations weights, then write to disk
            weight = 0
            for mcmc_run in reversed(self.mcmc_runs):
                weight += 1
                if mcmc_run["accept"]:
                    mcmc_run["weight"] = weight
                    weight = 0

            mcmc_runs_df = pd.DataFrame.from_dict(self.mcmc_runs)
            self.db.dump_df(db.store.Table.MCMC, mcmc_runs_df, append=False)


def check_hierarchical_priors(hierarchical_priors, priors):
    prior_names = [p.name for p in priors]
    for h_p in hierarchical_priors:
        variable_hyper_parameters = h_p.list_variable_hyper_parameters()
        for p_name in variable_hyper_parameters:
            msg = f"{p_name} is defined as a hyper-parameter but is not associated with a prior"
            assert p_name in prior_names, msg

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
    elif prior_dict["distribution"] == "normal":
        lower_bound = - float("inf")
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


def get_parameter_finite_range_from_prior(prior_dict):
    if prior_dict["distribution"] == "uniform":
        prior_low = prior_dict["distri_params"][0]
        prior_high = prior_dict["distri_params"][1]
    elif prior_dict["distribution"] == "lognormal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        prior_low = math.exp(mu + math.sqrt(2) * sd * special.erfinv(2 * 0.025 - 1))
        prior_high = math.exp(mu + math.sqrt(2) * sd * special.erfinv(2 * 0.975 - 1))
    elif prior_dict["distribution"] == "trunc_normal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        bounds = prior_dict["trunc_range"]
        prior_low = stats.truncnorm.ppf(
            0.025, (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
        )
        prior_high = stats.truncnorm.ppf(
            0.975, (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
        )
    elif prior_dict["distribution"] == "normal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        prior_low = stats.norm.ppf(
            0.025, loc=mu, scale=sd
        )
        prior_high = stats.norm.ppf(
            0.975, loc=mu, scale=sd
        )
    elif prior_dict["distribution"] == "beta":
        prior_low = stats.beta.ppf(
            0.025,
            prior_dict["distri_params"][0],
            prior_dict["distri_params"][1],
        )
        prior_high = stats.beta.ppf(
            0.975,
            prior_dict["distri_params"][0],
            prior_dict["distri_params"][1],
        )
    elif prior_dict["distribution"] == "gamma":
        prior_low = stats.gamma.ppf(
            0.025,
            prior_dict["distri_params"][0],
            0.0,
            prior_dict["distri_params"][1],
        )
        prior_high = stats.gamma.ppf(
            0.975,
            prior_dict["distri_params"][0],
            0.0,
            prior_dict["distri_params"][1],
        )
    else:
        raise_error_unsupported_prior(prior_dict["distribution"])

    return prior_low, prior_high


def sample_from_adaptive_gaussian(prev_params, adaptive_cov_matrix):
    return np.random.multivariate_normal(prev_params, adaptive_cov_matrix)


def remove_early_points_to_prevent_crash(target_outputs, priors):
    """
    Trim the beginning of the time series when model start time is varied during the MCMC
    """
    idx = None
    for i, p in enumerate(priors):
        if p["param_name"] == "time.start":
            idx = i
            break

    if idx is not None:
        latest_start_time = priors[idx]["distri_params"][1]
        for target in target_outputs:
            first_idx_to_keep = next(
                t_idx for t_idx, t_val in enumerate(target.data.index) if t_val > latest_start_time
            )
            target.data = target.data.iloc[first_idx_to_keep:]
            #target["values"] = target["values"][first_idx_to_keep:]

    return target_outputs


def set_initial_point(
    priors, model_parameters: dict, chain_idx, total_nb_chains, initialisation_type
):
    """
    Determine the starting point of the MCMC.
    """
    if initialisation_type == MetroInit.LHS:
        # draw samples using LHS based on the prior distributions
        np.random.seed(0)  # Set deterministic random seed for Latin Hypercube Sampling
        starting_points = sample_starting_params_from_lhs(priors, total_nb_chains)

        return starting_points[chain_idx - 1]
    elif initialisation_type == MetroInit.CURRENT_PARAMS:
        # use the current parameter values from the yaml files
        starting_points = read_current_parameter_values(priors, model_parameters)
        return starting_points
    else:
        raise ValueError(f"{initialisation_type} is not a supported Initialisation Type")


def read_current_parameter_values(priors, model_parameters):
    starting_points = {}
    for param_dict in priors:
        if param_dict["param_name"].endswith("dispersion_param"):
            assert param_dict["distribution"] == "uniform"
            starting_points[param_dict["param_name"]] = np.mean(param_dict["distri_params"])
        else:
            starting_points[param_dict["param_name"]] = read_param_value_from_string(
                model_parameters, param_dict["param_name"]
            )

    return starting_points
