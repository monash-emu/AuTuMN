import scipy
from typing import List

from autumn.tools.utils.outputsbuilder import OutputsBuilder
from autumn.models.sm_sir.parameters import TimeDistribution, ImmunityRiskReduction
from .constants import IMMUNITY_STRATA, FlowName, ImmunityStratum
import numpy as np
from scipy import stats


class SmSirOutputsBuilder(OutputsBuilder):

    def request_incidence(self, props_symptomatic: List[float], age_groups: List[int]):
        """
        Calculate incident infections. Will also calculate symptomatic incidence to simplify the calculations of
        notifications and hospitalisations.

        Args:
            props_symptomatic: List of the same length as age_groups
            age_groups: List of the age groups' starting breakpoints

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

    def request_notifications(
            self, prop_symptomatic_infections_notified: float, time_from_onset_to_notification: TimeDistribution,
            model_times: np.ndarray, age_groups: List[int]
    ):
        """
        Request notification calculations.

        Args:
            prop_symptomatic_infections_notified: Proportion notified among symptomatic cases (float)
            time_from_onset_to_notification: Details of the statistical distribution used to model time to notification
            model_times: The model evaluation times
            age_groups: Modelled age group lower breakpoints

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
                ),
                save_results=False
            )

        # Request aggregated notifications
        self.model.request_aggregate_output(
            name="notifications",
            sources=notification_sources
        )

    def request_hospitalisations(
        self,
        prop_hospital_among_symptomatic: List[float],
        hospital_risk_reduction_by_immunity: ImmunityRiskReduction,
        time_from_onset_to_hospitalisation: TimeDistribution,
        model_times: np.ndarray,
        age_groups: List[int]
    ):
        """
        Request hospitalisation-related outputs.

        Args:
            prop_hospital_among_symptomatic: Proportion ever hospitalised among symptomatic cases (float)
            hospital_risk_reduction_by_immunity: Hospital risk reduction according to immunity level
            time_from_onset_to_hospitalisation: Details of the statistical distribution for the time to hospitalisation
            model_times: The model evaluation times
            age_groups: Modelled age group lower breakpoints

        """

        # Prepare a dictionary with hospital risk reduction by level of immunity
        hospital_risk_reduction = {
            ImmunityStratum.NONE: 0.,
            ImmunityStratum.HIGH: hospital_risk_reduction_by_immunity.high,
            ImmunityStratum.LOW: hospital_risk_reduction_by_immunity.low
        }

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        interval_distri_densities = precompute_cdf_gaps(time_from_onset_to_hospitalisation, model_times)

        # Request hospital admissions for each age group
        hospital_admissions_sources = []
        for i_age, agegroup in enumerate(age_groups):
            for immunity_stratum in IMMUNITY_STRATA:
                hospital_risk = prop_hospital_among_symptomatic[i_age] * (1. - hospital_risk_reduction[immunity_stratum])
                output_name = f"hospital_admissionsXagegroup_{agegroup}Ximmunity_{immunity_stratum}"
                hospital_admissions_sources.append(output_name)
                self.model.request_function_output(
                    name=output_name,
                    sources=[f"incidence_symptXagegroup_{agegroup}Ximmunity_{immunity_stratum}"],
                    func=make_calc_hospital_admissions_func(
                        hospital_risk, interval_distri_densities
                    ),
                    save_results=False
                )
        # Request aggregated hospital admissions
        self.model.request_aggregate_output(
            name="hospital_admissions",
            sources=hospital_admissions_sources
        )

    def request_random_process_outputs(self,):
        self.model.request_computed_value_output("transformed_random_process")


def build_statistical_distribution(distribution_details: TimeDistribution):
    """
    Generate a scipy statistical distribution object that can then be used multiple times to evaluate the cdf

    Args:
        distribution_details: User request parameters that define the distribution

    Returns:
        A scipy statistical distribution object

    """

    if distribution_details.distribution == "gamma":
        shape = distribution_details.parameters["shape"]
        scale = distribution_details.parameters["mean"] / shape
        return stats.gamma(a=shape, scale=scale)
    else:
        raise ValueError(f"{distribution_details.distribution} distribution not supported")


def precompute_cdf_gaps(distribution_details, model_times):
    """
    Calculate the event probability associated with every possible time interval between two model times.

    Args:
        distribution_details: User requests for the distribution type
        model_times: The model evaluation times

    Returns:
        A list of the integrals of the PDF of the probability distribution of interest over each time period
        Its length is len(model_times) - 1

    """

    distribution = build_statistical_distribution(distribution_details)
    lags = [t - model_times[0] for t in model_times]
    cdf_values = distribution.cdf(lags)
    interval_distri_densities = np.diff(cdf_values)
    return interval_distri_densities


def apply_convolution(source_output: np.ndarray, cdf_gaps: np.ndarray, event_proba: float):
    """
    Calculate a convolved output.

    Args:
        source_output: Previously computed model output on which the calculation is based
        cdf_gaps: Overall probability distribution of event occurring at a particular time given that it occurs
        event_proba: Total probability of the event occurring

    Retuns:
        A numpy array of the convolved output

    """

    convolved_output = np.zeros_like(source_output)
    for i in range(source_output.size):
        trunc_source_output = list(source_output[:i])
        trunc_source_output.reverse()
        convolution_sum = sum([value * cdf_gap for (value, cdf_gap) in zip(trunc_source_output, cdf_gaps[:i])])
        convolved_output[i] = event_proba * convolution_sum

    return convolved_output


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


def make_calc_hospital_admissions_func(hospital_risk, cdf_gaps):

    def hospital_admissions_func(incidence_sympt):
        hospital_admissions = apply_convolution(incidence_sympt, cdf_gaps, hospital_risk)
        return hospital_admissions

    return hospital_admissions_func
