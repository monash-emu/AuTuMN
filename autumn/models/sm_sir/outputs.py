from scipy import stats
from typing import List
import numpy as np

from autumn.tools.utils.outputsbuilder import OutputsBuilder
from autumn.models.sm_sir.parameters import TimeDistribution, ImmunityRiskReduction
from .constants import IMMUNITY_STRATA, FlowName, ImmunityStratum, Compartment, ClinicalStratum
from autumn.tools.utils.utils import apply_odds_ratio_to_props


class SmSirOutputsBuilder(OutputsBuilder):

    def request_incidence(self, compartments, age_groups, clinical_strata, strain_strata):
        """
        Calculate incident disease cases. This is associated with the transition to infectiousness if there is only one
        infectious compartment, or transition between the two if there are two.
        Note that this differs from the approach in the covid_19 model, which took entry to the first "active"
        compartment to represent the onset of symptoms, which infectiousness starting before this.

        Args:
            compartments: list of model compartment names (unstratified)
            age_groups: list of modelled age groups
            clinical_strata: list of clinical strata
            strain_strata: list of modelled strains (None if single strain model)

        """

        # Determine what flow will be used to track disease incidence
        if Compartment.INFECTIOUS_LATE in compartments:
            incidence_flow = FlowName.WITHIN_INFECTIOUS
        elif Compartment.LATENT in compartments:
            incidence_flow = FlowName.PROGRESSION
        else:
            incidence_flow = FlowName.INFECTION

        self.model.request_output_for_flow(name="incidence", flow_name=incidence_flow)

        """
        Fully stratified incidence outputs
        """

        clinical_strata = [""] if not clinical_strata else clinical_strata
        strain_strata = [""] if not strain_strata else strain_strata
        detected_incidence_sources = []
        incidence_sympt_sources_by_age_and_immunity = {}
        for agegroup in age_groups:
            incidence_sympt_sources_by_age_and_immunity[agegroup] = {}
            for immunity_stratum in IMMUNITY_STRATA:
                incidence_sympt_sources_by_age_and_immunity[agegroup][immunity_stratum] = []
                for clinical_stratum in clinical_strata:
                    for strain in strain_strata:
                        output_name = f"incidenceXagegroup_{agegroup}Ximmunity_{immunity_stratum}"
                        dest_strata = {"agegroup": str(agegroup), "immunity": immunity_stratum}
                        if len(clinical_stratum) > 0:
                            output_name += f"Xclinical_{clinical_stratum}"
                            dest_strata["clinical"] = clinical_stratum
                        if len(strain) > 0:
                            output_name += f"Xstrain_{strain}"
                            dest_strata["strain"] = strain

                        self.model.request_output_for_flow(
                            name=output_name,
                            flow_name=incidence_flow,
                            dest_strata=dest_strata
                        )

                        if clinical_stratum in ["", ClinicalStratum.DETECT]:
                            detected_incidence_sources.append(output_name)

                        if clinical_stratum in ["", ClinicalStratum.SYMPT_NON_DETECT, ClinicalStratum.DETECT]:
                            incidence_sympt_sources_by_age_and_immunity[agegroup][immunity_stratum].append(output_name)

        # Compute detected incidence to prepare for notifications calculations
        self.model.request_aggregate_output(
            name="ever_detected_incidence",
            sources=detected_incidence_sources
        )

        # Compute symptomatic incidence by age and immunity status to prepare for hospital outputs calculations
        for agegroup in age_groups:
            for immunity_stratum in IMMUNITY_STRATA:
                self.model.request_aggregate_output(
                    name=f"incidence_symptXagegroup_{agegroup}Ximmunity_{immunity_stratum}",
                    sources=incidence_sympt_sources_by_age_and_immunity[agegroup][immunity_stratum]
                )

    def request_notifications(
            self, time_from_onset_to_notification: TimeDistribution, model_times: np.ndarray
    ):
        """
        Request notification calculations.

        Args:
            time_from_onset_to_notification: Details of the statistical distribution used to model time to notification
            model_times: The model evaluation times

        """

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        density_intervals = precompute_density_intervals(time_from_onset_to_notification, model_times)

        # Request notifications
        notifications_func = make_calc_notifications_func(density_intervals)
        self.model.request_function_output(
            name="notifications",
            sources=["ever_detected_incidence"],
            func=notifications_func,
        )

    def request_hospitalisations(
        self,
        prop_hospital_among_sympt: List[float],
        hospital_prop_multiplier: float,
        hospital_risk_reduction_by_immunity: ImmunityRiskReduction,
        time_from_onset_to_hospitalisation: TimeDistribution,
        hospital_stay_duration: TimeDistribution,
        model_times: np.ndarray,
        age_groups: List[int]
    ):
        """
        Request hospitalisation-related outputs.

        Args:
            prop_hospital_among_sympt: Proportion ever hospitalised among symptomatic cases (float)
            hospital_prop_multiplier: Multiplier applied as an odds ratio adjustment
            hospital_risk_reduction_by_immunity: Hospital risk reduction according to immunity level
            time_from_onset_to_hospitalisation: Details of the statistical distribution for the time to hospitalisation
            hospital_stay_duration: Details of the statistical distribution for hospitalisation stay duration
            model_times: The model evaluation times
            age_groups: Modelled age group lower breakpoints

        """

        # Adjusted hospital proportions
        adjusted_prop_hosp_among_sympt = apply_odds_ratio_to_props(prop_hospital_among_sympt, hospital_prop_multiplier)

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
                risk_reduction = 1. - hospital_risk_reduction[immunity_stratum]
                hospital_risk = adjusted_prop_hosp_among_sympt[i_age] * risk_reduction
                output_name = f"hospital_admissionsXagegroup_{agegroup}Ximmunity_{immunity_stratum}"
                hospital_admissions_sources.append(output_name)
                hospital_admissions_func = make_calc_admissions_func(hospital_risk, interval_distri_densities)
                self.model.request_function_output(
                    name=output_name,
                    sources=[f"incidence_symptXagegroup_{agegroup}Ximmunity_{immunity_stratum}"],
                    func=hospital_admissions_func
                )

        # Request aggregated hospital admissions
        self.model.request_aggregate_output(
            name="hospital_admissions",
            sources=hospital_admissions_sources
        )

        # Request aggregated hospital occupancy
        probas_stay_greater_than = precompute_probas_stay_greater_than(hospital_stay_duration, model_times)
        hospital_occupancy_func = make_calc_occupancy_func(probas_stay_greater_than)
        self.model.request_function_output(
            name="hospital_occupancy",
            sources=["hospital_admissions"],
            func=hospital_occupancy_func
        )

    def request_icu_outputs(
        self,
        prop_icu_among_hospitalised: float,
        time_from_hospitalisation_to_icu: TimeDistribution,
        icu_stay_duration: TimeDistribution,
        model_times: np.ndarray
    ):
        """
        Request ICU-related outputs.

        Args:
            prop_icu_among_hospitalised: Proportion ever requiring ICU stay among hospitalised cases (float)
            time_from_hospitalisation_to_icu: Details of the statistical distribution for the time to ICU admission
            icu_stay_duration: Details of the statistical distribution for ICU stay duration
            model_times: The model evaluation times

        """

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        interval_distri_densities = precompute_density_intervals(time_from_hospitalisation_to_icu, model_times)

        # Request ICU admissions
        icu_admissions_func = make_calc_admissions_func(prop_icu_among_hospitalised, interval_distri_densities)
        self.model.request_function_output(
            name="icu_admissions",
            sources=["hospital_admissions"],
            func=icu_admissions_func
        )

        # Request ICU occupancy
        probas_stay_greater_than = precompute_probas_stay_greater_than(icu_stay_duration, model_times)
        icu_occupancy_func = make_calc_occupancy_func(probas_stay_greater_than)
        self.model.request_function_output(
            name="icu_occupancy",
            sources=["icu_admissions"],
            func=icu_occupancy_func
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


def precompute_probas_stay_greater_than(distribution_details, model_times):
    """
    Calculate the probability that duration of stay is greater than every possible time interval between two model times

    Args:
        distribution_details: User requests for the distribution type
        model_times: The model evaluation times

    Returns:
        A list of the values of 1 - CDF for the probability distribution of interest over each time period
        Its length is len(model_times)

    """

    distribution = build_statistical_distribution(distribution_details)
    lags = [t - model_times[0] for t in model_times]
    cdf_values = distribution.cdf(lags)
    probas_stay_greater_than = 1 - cdf_values
    return probas_stay_greater_than

def convolve_probability(source_output: np.ndarray, density_intervals: np.ndarray, scale: float = 1.0, lag: int = 1) -> np.ndarray:
    """
    Calculate a convolved output.

    Args:
        source_output: Previously computed model output on which the calculation is based
        density_intervals: Overall probability density of occurence at particular timestep
        scale: Total probability scale of event occuring
        lag: Lag in output; defaults to 1 to reflect that events cannot occur at the same time

    Returns:
        A numpy array of the convolved output

    """
    convolved_output = np.zeros_like(source_output)

    offset = 1 - lag
    
    for i in range(source_output.size):
        trunc_source_output = source_output[:i+offset][::-1]
        trunc_intervals = density_intervals[:i+offset]
        convolution_sum = (trunc_source_output * trunc_intervals).sum()
        convolved_output[i] = scale * convolution_sum

    return convolved_output


def apply_convolution_for_event(source_output: np.ndarray, density_intervals: np.ndarray, event_proba: float):
    """
    Calculate a convolved output.

    Args:
        source_output: Previously computed model output on which the calculation is based
        density_intervals: Overall probability distribution of event occurring at a particular time given that it occurs
        event_proba: Total probability of the event occurring

    Returns:
        A numpy array of the convolved output

    """

    return convolve_probability(source_output, density_intervals, scale=event_proba)


def apply_convolution_for_occupancy(source_output: np.ndarray, probas_stay_greater_than: np.ndarray):
    """
    Calculate a convolved output.

    Args:
        source_output: Previously computed model output on which the calculation is based
        probas_stay_greater_than: Probability that duration of stay is greater than each possible time interval between
        two model times

    Returns:
        A numpy array for the convolved output

    """

    return convolve_probability(source_output, probas_stay_greater_than, lag=0)


"""
Below are a few factory functions used when declaring functions within loops. This should prevent issues.
"""


def make_calc_notifications_func(density_intervals):

    def notifications_func(detected_incidence):
        notifications = apply_convolution_for_event(detected_incidence, density_intervals, 1.)
        return notifications

    return notifications_func


def make_calc_admissions_func(admission_risk, density_intervals):

    def admissions_func(reference_output):
        admissions = apply_convolution_for_event(reference_output, density_intervals, admission_risk)
        return admissions

    return admissions_func


def make_calc_occupancy_func(probas_stay_greater_than):

    def occupancy_func(admissions):
        occupancy = apply_convolution_for_occupancy(admissions, probas_stay_greater_than)
        return occupancy

    return occupancy_func
