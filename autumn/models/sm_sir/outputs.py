from scipy import stats
from typing import List
import numpy as np

from autumn.tools.utils.outputsbuilder import OutputsBuilder
from autumn.models.sm_sir.parameters import TimeDistribution, ImmunityRiskReduction
from .constants import IMMUNITY_STRATA, FlowName, ImmunityStratum, Compartment


class SmSirOutputsBuilder(OutputsBuilder):

    def request_incidence(self, compartments, age_groups, clinical_strata, strain_strata):
        """
        Calculate incident disease cases. This is associated with the transition to the first state where individuals are
        potentially symptomatic.

        Args:
            compartments: list of model compartment names (unstratified)
            age_groups: list of modelled age groups
            clinical_strata: list of clinical strata
            strain_strata: list of modelled strains (None if single strain model)
        """

        # Determine what flow will be used to track disease incidence
        if Compartment.INFECTIOUS_LATE in compartments:
            incidence_flow = FlowName.WITHIN_INFECTIOUS
        elif Compartment.EXPOSED in compartments:
            incidence_flow = FlowName.PROGRESSION
        else:
            incidence_flow = FlowName.INFECTION

        self.model.request_output_for_flow(name="incidence", flow_name=incidence_flow)

        """
        Fully stratified incidence outputs
        """
        for agegroup in age_groups:
            for clinical_stratum in clinical_strata:
                for immunity_stratum in IMMUNITY_STRATA:
                    if not strain_strata:
                        self.model.request_output_for_flow(
                            name=f"incidenceXagegroup_{agegroup}Xclinical_{clinical_stratum}Ximmunity_{immunity_stratum}",
                            flow_name=incidence_flow,
                            dest_strata={"agegroup": str(agegroup), "clinical": clinical_stratum, "immunity": immunity_stratum}
                        )
                    else:
                        for strain in strain_strata:
                            self.model.request_output_for_flow(
                                name=f"incidenceXagegroup_{agegroup}Xclinical_{clinical_stratum}Xstrain_{strain}Ximmunity_{immunity_stratum}",
                                flow_name=incidence_flow,
                                dest_strata={"agegroup": str(agegroup), "clinical": clinical_stratum, "strain": strain, "immunity": immunity_stratum}
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
        density_intervals = precompute_density_intervals(time_from_onset_to_notification, model_times)

        # Request notifications for each age group
        notification_sources = []
        for i_age, agegroup in enumerate(age_groups):
            output_name = f"notificationsXagegroup_{agegroup}"
            notification_sources.append(output_name)
            notifications_func = make_calc_notifications_func(prop_symptomatic_infections_notified, density_intervals)
            self.model.request_function_output(
                name=output_name,
                sources=[f"incidence_symptXagegroup_{agegroup}"],
                func=notifications_func,
                save_results=False
            )

        # Request aggregated notifications
        self.model.request_aggregate_output(
            name="notifications",
            sources=notification_sources
        )

    def request_hospitalisations(
        self,
        prop_hospital_among_sympt: List[float],
        hospital_risk_reduction_by_immunity: ImmunityRiskReduction,
        time_from_onset_to_hospitalisation: TimeDistribution,
        model_times: np.ndarray,
        age_groups: List[int]
    ):
        """
        Request hospitalisation-related outputs.

        Args:
            prop_hospital_among_sympt: Proportion ever hospitalised among symptomatic cases (float)
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
        interval_distri_densities = precompute_density_intervals(time_from_onset_to_hospitalisation, model_times)

        # Request hospital admissions for each age group
        hospital_admissions_sources = []
        for i_age, agegroup in enumerate(age_groups):
            for immunity_stratum in IMMUNITY_STRATA:
                hospital_risk = prop_hospital_among_sympt[i_age] * (1. - hospital_risk_reduction[immunity_stratum])
                output_name = f"hospital_admissionsXagegroup_{agegroup}Ximmunity_{immunity_stratum}"
                hospital_admissions_sources.append(output_name)
                hospital_admissions_func = make_calc_hospital_admissions_func(hospital_risk, interval_distri_densities)
                self.model.request_function_output(
                    name=output_name,
                    sources=[f"incidence_symptXagegroup_{agegroup}Ximmunity_{immunity_stratum}"],
                    func=hospital_admissions_func,
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


def precompute_density_intervals(distribution_details, model_times):
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


def apply_convolution(source_output: np.ndarray, density_intervals: np.ndarray, event_proba: float):
    """
    Calculate a convolved output.

    Args:
        source_output: Previously computed model output on which the calculation is based
        density_intervals: Overall probability distribution of event occurring at a particular time given that it occurs
        event_proba: Total probability of the event occurring

    Retuns:
        A numpy array of the convolved output

    """

    convolved_output = np.zeros_like(source_output)
    for i in range(source_output.size):
        trunc_source_output = list(source_output[:i])
        trunc_source_output.reverse()
        convolution_sum = sum([value * cdf_gap for (value, cdf_gap) in zip(trunc_source_output, density_intervals[:i])])
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
