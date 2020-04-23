import logging
import os
from time import time
from itertools import chain
from datetime import datetime

import theano
import math
import pandas as pd
import numpy as np
import copy
import pymc3 as pm
import theano.tensor as tt
import autumn.post_processing as post_proc
from scipy.optimize import Bounds, minimize
from scipy import stats, special

from . import constants
from .tb_model import store_tb_database
from .db import Database
from .tool_kit.utils import get_data_hash
from autumn.tool_kit.scenarios import Scenario

from autumn.tb_model.outputs import _unpivot_outputs
import yaml

logger = logging.getLogger("pymc3")
logger.setLevel(logging.DEBUG)

_logger = logging.getLogger("theano.gof.compilelock")
_logger.setLevel(logging.DEBUG)

theano.config.optimizer = "None"


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
    else:
        raise ValueError("prior distribution bounds detection currently not handled.")

    return lower_bound, upper_bound


class Calibration:
    """
    this class handles model calibration using an MCMC algorithm if sampling from the posterior distribution is
    required, or using maximum likelihood estimation if only one calibrated parameter set is required.
    """

    def __init__(
        self,
        model_name: str,
        model_builder,
        priors,
        targeted_outputs,
        multipliers,
        chain_index,
        model_parameters={},
        start_time_range=None,
        record_rejected_outputs=False
    ):

        self.model_name = model_name
        self.model_builder = model_builder  # a function that builds a new model without running it
        self.model_parameters = model_parameters
        self.scenarios = []
        self.initialise_scenario_list()
        self.record_rejected_outputs = record_rejected_outputs
        self.start_time_range = (
            start_time_range  # if specified, we allow start time to vary to achieve the best fit
        )
        self.best_start_time = None
        self.post_processing = None  # a PostProcessing object containing the required outputs of a model that has been run
        self.priors = priors  # a list of dictionaries. Each dictionary describes the prior distribution for a parameter
        self.param_list = [self.priors[i]["param_name"] for i in range(len(self.priors))]
        self.targeted_outputs = (
            targeted_outputs  # a list of dictionaries. Each dictionary describes a target
        )
        self.multipliers = multipliers
        self.chain_index = chain_index

        # Setup output database directory
        project_dir = os.path.join(constants.DATA_PATH, model_name)
        run_hash = get_data_hash(model_name, priors, targeted_outputs, multipliers)
        timestamp = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
        out_db_dir = os.path.join(project_dir, f"calibration-{run_hash}-{timestamp}")
        os.makedirs(out_db_dir, exist_ok=True)
        db_name = f"outputs_calibration_chain_{self.chain_index}.db"
        self.output_db_path = os.path.join(out_db_dir, db_name)

        self.data_as_array = None  # will contain all targeted data points in a single array
        self.loglike = None  # will store theano object

        self.format_data_as_array()
        self.workout_unspecified_target_sds()  # for likelihood definition
        self.workout_unspecified_jumping_sds()  # for proposal function definition

        self.create_loglike_object()
        self.iter_num = 0
        self.run_mode = None
        self.main_table = {}
        self.mcmc_trace = None  # will store the results of the MCMC model calibration
        self.mle_estimates = {}  # will store the results of the maximum-likelihood calibration

        self.evaluated_params_ll = []  # list of tuples:  [(theta_0, ll_0), (theta_1, ll_1), ...]

    def initialise_scenario_list(self):
        base_scenario = Scenario(self.model_builder, 0, copy.deepcopy(self.model_parameters))
        self.scenarios = [base_scenario]
        if 'scenarios' in self.model_parameters:
            for scenario_index in [sc_i for sc_i in self.model_parameters['scenarios'] if int(sc_i) > 0]:
                scenario = Scenario(self.model_builder, scenario_index, copy.deepcopy(self.model_parameters))
                self.scenarios.append(scenario)

    def update_post_processing(self):
        """
        updates self.post_processing attribute based on the newly run model
        :return:
        """
        requested_outputs = [
            self.targeted_outputs[i]["output_key"]
            for i in range(len(self.targeted_outputs))
            if "prevX" in self.targeted_outputs[i]["output_key"]
        ]
        requested_times = {}

        for _output in self.targeted_outputs:
            if "prevX" in _output["output_key"]:
                requested_times[_output["output_key"]] = _output["years"]

        self.post_processing = post_proc.PostProcessing(
            self.scenarios[0].model,
            requested_outputs=requested_outputs,
            requested_times=requested_times,
            multipliers=self.multipliers,
        )

    def store_model_outputs(self, scenario_index):
        """
        Record the model outputs in the database
        """
        _model = self.scenarios[scenario_index].model
        out_df = pd.DataFrame(
            _model.outputs, columns=_model.compartment_names
        )
        derived_output_df = pd.DataFrame.from_dict(_model.derived_outputs)
        store_tb_database(
            derived_output_df,
            table_name="derived_outputs",
            run_idx=self.iter_num,
            database_name=self.output_db_path,
            append=True,
            scenario=scenario_index
        )
        store_tb_database(
            out_df,
            run_idx=self.iter_num,
            times=_model.times,
            database_name=self.output_db_path,
            append=True,
            scenario=scenario_index
        )
        pbi_outputs = _unpivot_outputs(_model)
        store_tb_database(
            pbi_outputs,
            table_name='pbi_scenario_' + str(scenario_index),
            database_name=self.output_db_path,
            scenario=scenario_index,
            run_idx=self.iter_num
        )

    def store_mcmc_iteration_info(self, proposed_params, proposed_loglike, accept, i_run):
        """
        Records the MCMC iteration details
        :param proposed_params: the current parameter values
        :param proposed_loglike: the current loglikelihood
        :param accept: whether the iteration was accepted or not
        :param i_run: the iteration number
        """
        mcmc_run_dict = {k: v for k, v in zip(self.param_list, proposed_params)}
        mcmc_run_dict["loglikelihood"] = proposed_loglike
        mcmc_run_dict["accept"] = 1 if accept else 0
        mcmc_run_colnames = self.param_list.copy()
        mcmc_run_colnames.append("loglikelihood")
        mcmc_run_colnames.append("accept")
        mcmc_run_df = pd.DataFrame(mcmc_run_dict, columns=mcmc_run_colnames, index=[i_run])
        store_tb_database(
            mcmc_run_df,
            table_name="mcmc_run",
            run_idx=i_run,
            database_name=self.output_db_path,
            append=True,
        )

    def run_model_with_params(self, params):
        """
        run the model with a set of params.
        :param params: a list containing the parameter values for update
        """
        update_params = {}  # self.model_parameters
        for i, param_name in enumerate(self.param_list):
            update_params[param_name] = params[i]

        self.scenarios[0].params['default'].update(update_params)
        self.scenarios[0].run()
        self.update_post_processing()

    def run_extra_scenarios_with_params(self, params):
        """
        Run intervention scenarios after accepting a baseline run
        :param params: list of all current MCMC parameter values
        """
        for scenario_idx, scenario_params in self.model_parameters['scenarios'].items():
            if scenario_idx == 0:
                continue
            scenario_params["start_time"] = self.model_parameters['scenario_start_time']

            # Potential update of scenario params if these are among the MCMC params
            updated_scenario_params = copy.copy(scenario_params)
            for param_name in scenario_params.keys():
                if param_name in self.param_list:
                    param_index = self.param_list.index(param_name)
                    updated_scenario_params[param_name] = params[param_index]

            # update default params
            self.scenarios[scenario_idx].params['default'] = copy.deepcopy(self.scenarios[0].params['default'])

            # update scenario params
            self.scenarios[scenario_idx].params['scenarios'][scenario_idx].update(updated_scenario_params)

            # Run scenario
            baseline_model = copy.deepcopy(self.scenarios[0].model)
            self.scenarios[scenario_idx].run(base_model=baseline_model)

            self.store_model_outputs(scenario_idx)

    def loglikelihood(self, params, to_return="best_ll"):
        """
        defines the loglikelihood
        :param params: model parameters
        :return: the loglikelihood
        """
        # run the model
        best_ll = None
        # for evaluated in self.evaluated_params_ll:
        #     if np.array_equal(params, evaluated[0]):
        #         best_ll = evaluated[1]
        #         break

        if best_ll is None:
            self.run_model_with_params(params)

            model_start_time = self.post_processing.derived_outputs["times"][0]
            if self.start_time_range is None:
                considered_start_times = [model_start_time]
            else:
                considered_start_times = np.linspace(
                    self.start_time_range[0],
                    self.start_time_range[1],
                    num=self.start_time_range[1] - self.start_time_range[0] + 1,
                )

            best_ll, best_start_time = (-1.0e60, None)
            if self.run_mode == "lsm":
                best_ll = 1.0e60
            for considered_start_time in considered_start_times:
                time_shift = considered_start_time - model_start_time
                ll = 0  # loglikelihood if using bayesian approach. Sum of squares if using lsm mode
                for target in self.targeted_outputs:
                    key = target["output_key"]
                    data = np.array(target["values"])
                    if key in self.post_processing.generated_outputs:
                        if self.start_time_range is not None:
                            raise ValueError(
                                "variable start time implemented for derived_outputs only"
                            )
                        model_output = np.array(self.post_processing.generated_outputs[key])
                    else:
                        indices = []
                        for year in target["years"]:
                            indices.append(self.scenarios[0].model.times.index(year - time_shift))
                        model_output = np.array(
                            [self.post_processing.derived_outputs[key][index] for index in indices]
                        )

                    if self.run_mode == "lsm":
                        ll += np.sum((data - model_output) ** 2)
                    else:
                        if "loglikelihood_distri" not in target:  # default distribution
                            target["loglikelihood_distri"] = "normal"
                        if target["loglikelihood_distri"] == "normal":
                            ll += -(0.5 / target["sd"] ** 2) * np.sum((data - model_output) ** 2)
                        elif target["loglikelihood_distri"] == "poisson":
                            for i in range(len(data)):
                                ll += (
                                    data[i] * math.log(model_output[i])
                                    - model_output[i]
                                    - math.log(math.factorial(data[i]))
                                )
                        else:
                            raise ValueError("Distribution not supported in loglikelihood_distri")

                if (ll > best_ll and self.run_mode != "lsm") or (
                    ll < best_ll and self.run_mode == "lsm"
                ):
                    best_ll, best_start_time = (ll, considered_start_time)

            if self.run_mode != "autumn_mcmc":
                mcmc_run_dict = {k: v for k, v in zip(self.param_list, params)}
                mcmc_run_dict["loglikelihood"] = best_ll
                mcmc_run_colnames = self.param_list.copy()
                mcmc_run_colnames = mcmc_run_colnames.append("loglikelihood")
                mcmc_run_df = pd.DataFrame(
                    mcmc_run_dict, columns=mcmc_run_colnames, index=[self.iter_num]
                )
                store_tb_database(
                    mcmc_run_df,
                    table_name="mcmc_run",
                    run_idx=self.iter_num,
                    database_name=self.output_db_path,
                    append=True,
                )
            self.evaluated_params_ll.append((copy.copy(params), copy.copy(best_ll)))

        if to_return == "best_ll":
            return best_ll
        elif to_return == "best_start_time":
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
                    self.targeted_outputs[i]["sd"] = 0.5 / 4.0 * np.mean(target["values"])

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
                else:
                    print(
                        "prior_width not specified for "
                        + prior_dict["distribution"]
                        + " distribution at the moment"
                    )

                #  95% of the sampled values within [mu - 2*sd, mu + 2*sd], i.e. interval of witdth 4*sd
                relative_prior_width = (
                    0.25  # fraction of prior_width in which 95% of samples should fall
                )
                self.priors[i]["jumping_sd"] = relative_prior_width * prior_width / 4.0

    def create_loglike_object(self):
        """
        create a 'theano-type' object to compute the likelihood
        """
        self.loglike = LogLike(self.loglikelihood)

    def run_fitting_algorithm(
        self,
        run_mode="mle",
        mcmc_method="Metropolis",
        n_iterations=100,
        n_burned=10,
        n_chains=1,
        available_time=None,
    ):
        """
        master method to run model calibration.

        :param run_mode: either 'pymc_mcmc' (for sampling from the posterior) or 'mle' (maximum likelihood estimation) or
        'lsm' (for least square minimisation using scipy.minimize function)
        :param mcmc_method: if run_mode == 'mcmc' , either 'Metropolis' or 'DEMetropolis'
        :param n_iterations: number of iterations requested for sampling (excluding burn-in phase)
        :param n_burned: number of burned iterations before effective sampling
        :param n_chains: number of chains to be run
        :param available_time: maximal simulation time allowed (in seconds)
        """

        # initialise random seed differently for different chains
        np.random.seed(self.chain_index + int(time()))

        self.run_mode = run_mode
        if run_mode == "autumn_mcmc":
            if n_chains > 1:
                print("autumn_mcmc method does not support multiple-chain runs at the moment")
            t0 = time()

            self.mcmc_trace = {}  # will store param trace and loglikelihood evolution
            for prior_dict in self.priors:
                self.mcmc_trace[prior_dict["param_name"]] = []
            self.mcmc_trace["loglikelihood"] = []

            last_accepted_params = None
            last_acceptance_quantity = None  # acceptance quantity is defined as loglike + logprior
            last_acceptance_loglike = None
            for i_run in range(n_iterations + n_burned):
                # propose new param set
                proposed_params = self.propose_new_params(last_accepted_params)

                # evaluate log-likelihood
                proposed_loglike = self.loglikelihood(proposed_params)

                # evaluate log-prior
                proposed_logprior = self.logprior(proposed_params)

                # decide acceptance
                proposed_acceptance_quantity = proposed_loglike + proposed_logprior
                accept = False
                if (
                    last_acceptance_quantity is None
                    or proposed_acceptance_quantity >= last_acceptance_quantity
                ):
                    accept = True
                else:
                    acc_proba = np.exp(proposed_acceptance_quantity - last_acceptance_quantity)
                    accept = (
                        np.random.binomial(n=1, p=acc_proba, size=1) > 0
                    )  # binomial returns integer. We need a boolean

                # update stored quantities
                if accept:
                    last_accepted_params = proposed_params
                    last_acceptance_quantity = proposed_acceptance_quantity
                    last_acceptance_loglike = proposed_loglike

                self.update_mcmc_trace(last_accepted_params, last_acceptance_loglike)

                # store model outputs
                if accept or self.record_rejected_outputs:
                    self.store_model_outputs(0)

                self.store_mcmc_iteration_info(proposed_params, proposed_loglike, accept, i_run)

                # Run intervention scenarios if accepted run
                if accept:
                    print("Running extra scenarios")
                    self.run_extra_scenarios_with_params(proposed_params)

                self.iter_num += 1

                print(str(i_run + 1) + " MCMC iterations completed.")

                if available_time is not None:
                    elapsed_time = time() - t0
                    if elapsed_time > available_time:
                        print(
                            "Stopping MCMC simulation after "
                            + str(i_run + 1)
                            + " iterations because of time limit"
                        )
                        break

        elif run_mode in ["pymc_mcmc", "mle"]:
            basic_model = pm.Model()
            with basic_model:

                fitted_params = []
                for prior in self.priors:
                    if prior["distribution"] == "uniform":
                        value = pm.Uniform(
                            prior["param_name"],
                            lower=prior["distri_params"][0],
                            upper=prior["distri_params"][1],
                        )
                        fitted_params.append(value)

                theta = tt.as_tensor_variable(fitted_params)

                pm.DensityDist("likelihood", lambda v: self.loglike(v), observed={"v": theta})

                if run_mode == "mle":
                    self.mle_estimates = pm.find_MAP()
                elif run_mode == "pymc_mcmc":  # full MCMC requested
                    if mcmc_method == "Metropolis":
                        mcmc_step = pm.Metropolis(S=np.array([1.0]))
                    elif mcmc_method == "DEMetropolis":
                        mcmc_step = pm.DEMetropolis()
                    else:
                        ValueError(
                            "requested mcmc mode is not supported. Must be one of ['Metropolis', 'DEMetropolis']"
                        )
                    self.mcmc_trace = pm.sample(
                        draws=n_iterations,
                        step=mcmc_step,
                        tune=n_burned,
                        chains=n_chains,
                        progressbar=False,
                    )

                    traceDf = pm.trace_to_dataframe(self.mcmc_trace)
                    traceDf.to_csv("trace.csv")
                    out_database = Database(database_name=self.output_db_path)
                    mcmc_run_df = out_database.db_query("mcmc_run")
                    mcmc_run_df = mcmc_run_df.reset_index(drop=True)
                    traceDf = traceDf.reset_index(drop=True)
                    traceDf["accepted"] = 1
                    mcmc_run_info = pd.merge(
                        mcmc_run_df,
                        traceDf,
                        how="left",
                        left_on=self.param_list,
                        right_on=self.param_list,
                    )
                    mcmc_run_info["accepted"].fillna(0, inplace=True)
                    mcmc_run_info = mcmc_run_info.drop_duplicates()
                    store_tb_database(
                        mcmc_run_info, table_name="mcmc_run_info", database_name=self.output_db_path
                    )
                else:
                    ValueError(
                        "requested run mode is not supported. Must be one of ['pymc_mcmc', 'lme']"
                    )
        elif run_mode == "lsm":
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
            if self.start_time_range is not None:
                self.best_start_time = self.loglikelihood(
                    self.mle_estimates, to_return="best_start_time"
                )
                print("Best start time: " + str(self.best_start_time))

            # FIXME: need to fix dump_mle_params_to_yaml_file
            # print("Best solution:")
            # print(self.mle_estimates)
            # self.dump_mle_params_to_yaml_file()

        else:
            ValueError(
                "requested run mode is not supported. Must be one of ['pymc_mcmc', 'lme', 'autumn_mcmc']"
            )

    def propose_new_params(self, prev_params):
        """
        calculated the joint log prior
        :param prev_params: last accepted parameter values as a list ordered using the order of self.priors
        :return: a new list of parameter values
        """
        # prev_params assumed to be the manually calibrated parameters for first step
        if prev_params is None:
            prev_params = []
            for prior_dict in self.priors:
                prev_params.append(self.model_parameters['default'][prior_dict["param_name"]])

        new_params = []

        for i, prior_dict in enumerate(self.priors):
            # Work out bounds for acceptable values, using the support of the prior distribution
            lower_bound, upper_bound = get_parameter_bounds_from_priors(prior_dict)
            sample = lower_bound - 10.0  # deliberately initialise out of parameter scope
            while not lower_bound <= sample <= upper_bound:
                sample = np.random.normal(
                    loc=prev_params[i], scale=prior_dict["jumping_sd"], size=1
                )[0]
            new_params.append(sample)

        return new_params

    def logprior(self, params):
        """
        calculated the joint log prior
        :param params: model parameters as a list of values ordered using the order of self.priors
        :return: the natural log of the joint prior
        """
        logp = 0.0
        for i, prior_dict in enumerate(self.priors):
            if prior_dict["distribution"] == "uniform":
                logp += math.log(
                    1.0 / (prior_dict["distri_params"][1] - prior_dict["distri_params"][0])
                )
            elif prior_dict["distribution"] == "lognormal":
                mu = prior_dict["distri_params"][0]
                sd = prior_dict["distri_params"][1]
                logp += stats.lognorm.logpdf(
                    x=params[i], s=sd, scale=math.exp(mu)
                )  # see documentation of stats.lognorm for scale
            else:
                print(
                    prior_dict["distribution"]
                    + "distribution not supported in autumn_mcmc at the moment"
                )

        return logp

    def update_mcmc_trace(self, params_to_store, loglike_to_store):
        """
        store mcmc iteration into param_trace
        :param params_to_store: model parameters as a list of values ordered using the order of self.priors
        :param loglike_to_store: current loglikelihood value
        """
        for i, prior_dict in enumerate(self.priors):
            self.mcmc_trace[prior_dict["param_name"]].append(params_to_store[i])
        self.mcmc_trace["loglikelihood"].append(loglike_to_store)

    def dump_mle_params_to_yaml_file(self):

        dict_to_dump = {}
        for i, param_name in enumerate(self.param_list):
            dict_to_dump[param_name] = float(self.mle_estimates[i])
        if self.best_start_time is not None:
            dict_to_dump["start_time"] = self.best_start_time

        # FIXME: DATA_PATH unknown
        # file_path = os.path.join(DATA_PATH, self.model_name, "mle_params.yml")
        # with open(file_path, "w") as outfile:
        #     yaml.dump(dict_to_dump, outfile, default_flow_style=False)


class LogLike(tt.Op):
    """
    Define a theano Op for our likelihood function.
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        :param loglike: The log-likelihood function
        """

        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta)

        outputs[0][0] = np.array(logl)  # output the log-likelihood


# FIXME: Move this script to a smoke test or delete. This file should not depend on mongolia tb model.
# if __name__ == "__main__":
#     par_priors = [{'param_name': 'contact_rate', 'distribution': 'uniform', 'distri_params': [14., 18.]},
#                   # {'param_name': 'rr_transmission_ger', 'distribution': 'uniform', 'distri_params': [1., 5.]},
#                   # {'param_name': 'rr_transmission_urban', 'distribution': 'uniform', 'distri_params': [1., 5.]},
#                   # {'param_name': 'rr_transmission_province', 'distribution': 'uniform', 'distri_params': [.5, 5.]},
#                   {'param_name': 'latency_adjustment', 'distribution': 'uniform', 'distri_params': [1., 3.]},
#                   ]
#     target_outputs = [{'output_key': 'prevXinfectiousXamongXage_15Xage_60', 'years': [2015.], 'values': [560.]},
#                       # {'output_key': 'prevXinfectiousXamongXage_15Xage_60Xhousing_ger', 'years': [2015.], 'values': [613.]},
#                       # {'output_key': 'prevXinfectiousXamongXage_15Xage_60Xlocation_urban', 'years': [2015.], 'values': [586.]},
#                       # {'output_key': 'prevXinfectiousXamongXage_15Xage_60Xlocation_province', 'years': [2015.], 'values': [513.]},
#                       {'output_key': 'prevXlatentXamongXage_5', 'years': [2016.], 'values': [960.]}
#                      ]
#     multipliers = {'prevXlatentXamongXage_5': 1.e4}
#     for i, output in enumerate(target_outputs):
#         if output['output_key'][0:15] == 'prevXinfectious':
#             multipliers[output['output_key']] = 1.e5
#     calib = Calibration(build_mongolia_model, par_priors, target_outputs, multipliers)
#     # calib.run_fitting_algorithm(run_mode='lsm')  # for least square minimization
#     # print(calib.mle_estimates)
#     #
#     # calib.run_fitting_algorithm(run_mode='mle')  # for maximum-likelihood estimation
#     # print(calib.mle_estimates)
#     #
#     calib.run_fitting_algorithm(run_mode='autumn_mcmc', n_iterations=10, n_burned=0, n_chains=1, available_time=10)  # for autumn_mcmc
#     print(calib.mcmc_trace)
