from typing import List

from autumn.tools.utils.outputsbuilder import OutputsBuilder
from .constants import IMMUNITY_STRATA, FlowName
import numpy as np
from scipy import stats


class SmSirOutputsBuilder(OutputsBuilder):

    def request_incidence(self, props_symptomatic: List[float], age_groups: List[int]):
        """
        Calculate incident infections. Will also calculate symptomatic incidence to simplify the calculations of
        notifications and hospitalisations.

        Args:
            props_symptomatic: list of the same length as age_groups
            age_groups: list of the age groups' starting breakpoints

        """

        self.model.request_output_for_flow(name="incidence", flow_name=FlowName.INFECTION)

        """
        Stratified by age group and immunity status
        """

        # First calculate all incident cases, including asymptomatic
        self.request_double_stratified_output_for_flow(
            FlowName.INFECTION,
            [str(group) for group in age_groups],
            "agegroup",
            IMMUNITY_STRATA,
            "immunity",
            name_stem="incidence"
        )

        # Then calculate incident symptomatic cases
        for i_age, agegroup in enumerate(age_groups):
            for immunity_stratum in IMMUNITY_STRATA:
                self.model.request_function_output(
                    name=f"incidence_symptXagegroup_{agegroup}Ximmunity_{immunity_stratum}",
                    func=make_incidence_sympt_func(props_symptomatic[i_age]),
                    sources=[f"incidenceXagegroup_{agegroup}Ximmunity_{immunity_stratum}"],
                    save_results=False
                )

        # Stratified by age group (by aggregating double stratified outputs)
        for agegroup in age_groups:
            for suffix in ["", "_sympt"]:
                sources = [
                    f"incidence{suffix}Xagegroup_{agegroup}Ximmunity_{immunity_stratum}" for
                    immunity_stratum in IMMUNITY_STRATA
                ]
                self.model.request_aggregate_output(
                    name=f"incidence{suffix}Xagegroup_{agegroup}",
                    sources=sources
                )

    def request_notifications(self, prop_symptomatic_infections_notified, time_from_onset_to_notification, model_times, age_groups):
        """
        Request notification calculations.

        Args:
            prop_symptomatic_infections_notified: proportion notified among symptomatic cases (float)
            time_from_onset_to_notification: details of the statistical distribution used to model time to notification
            model_times: model times

        """

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        cdf_gaps = precompute_cdf_gaps(time_from_onset_to_notification, model_times)

        # Request notifications for each age group
        notification_sources = []
        for i_age, agegroup in enumerate(age_groups):
            output_name = f"notificationsXagegroup_{agegroup}"
            notification_sources.append(output_name)
            self.model.request_function_output(
                name=output_name,
                sources=[f"incidence_symptXagegroup_{agegroup}"],
                func=make_calc_notifications_func(
                    prop_symptomatic_infections_notified, cdf_gaps
                )
            )

        # Request aggregated notifications
        self.model.request_aggregate_output(
            name="notifications",
            sources=notification_sources
        )

    def request_random_process_outputs(self,):
        self.model.request_computed_value_output("transformed_random_process")


def build_statistical_distribution(distribution_details):
    """
    Generate a scipy statistical distribution object that can then be used multiple times to evaluate the cdf
    :param distribution_details: a dictionary describing the distribution
    :return: a scipy statistical distribution object
    """
    if distribution_details.distribution == "gamma":
        shape = distribution_details.parameters['shape']
        scale = distribution_details.parameters['mean'] / shape
        return stats.gamma(a=shape, scale=scale)
    else:
        raise ValueError(f"{distribution_details.distribution} distribution not supported")


def precompute_cdf_gaps(distribution_details, model_times):
    """
    Calculate the event probability associated with every possible time interval between two model times.
    :param distribution_details: a dictionary describing the statistical distributions
    :param model_times: the model times
    :return: a list of cdf gap values. Its length is len(model_times) - 1
    """
    distribution = build_statistical_distribution(distribution_details)
    lags = [t - model_times[0] for t in model_times]
    cdf_values = distribution.cdf(lags)
    cdf_gaps = np.diff(cdf_values)
    return cdf_gaps


def apply_convolution(source_output, cdf_gaps, event_proba):
    """
    Calculate a convoluted output.
    :param source_output: the already computed model output on which the calculation is based
    :param cdf_gaps: event occurence probability for every possible time interval
    :param event_proba: overall probability of event occurrence
    :return: a numpy array with the convoluted output
    """
    convoluted_output = np.zeros_like(source_output)
    for i in range(source_output.size):
        trunc_source_output = list(source_output[:i])
        trunc_source_output.reverse()
        convolution_sum = sum([value * cdf_gap for (value, cdf_gap) in zip(trunc_source_output, cdf_gaps[:i])])
        convoluted_output[i] = event_proba * convolution_sum

    return convoluted_output


"""
Below are a few factory functions used when declaring functions within loops. This should prevent issues.
"""


def make_calc_notifications_func(prop_symptomatic_infections_notified, cdf_gaps):

    def notifications_func(incidence_sympt):
        notifications = apply_convolution(incidence_sympt, cdf_gaps, prop_symptomatic_infections_notified)
        return notifications

    return notifications_func


def make_incidence_sympt_func(prop_sympt):

    def incidence_sympt_func(inc):
        return prop_sympt * inc

    return incidence_sympt_func
