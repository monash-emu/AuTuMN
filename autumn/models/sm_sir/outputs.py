from scipy import stats
from typing import List, Dict, Optional
import numpy as np

from autumn.tools.utils.outputsbuilder import OutputsBuilder
from autumn.models.sm_sir.parameters import TimeDistribution, ImmunityRiskReduction, VocComponent
from .constants import IMMUNITY_STRATA, FlowName, ImmunityStratum, Compartment, ClinicalStratum
from autumn.tools.utils.utils import apply_odds_ratio_to_props, weighted_average
from autumn.models.sm_sir.strat_processing.agegroup import convert_param_agegroups


def get_immunity_prop_modifiers(
        source_pop_immunity_dist: Dict[str, float],
        source_pop_protection: Dict[str, float],
):
    """
    Work out how much we are going to adjust the age-specific hospitalisation and mortality rates by to account for pre-
    existing immunity protection against hospitalisation and mortality.
    The aim is to get the final parameters to come out so that the weighted average effect comes out to that reported by
    Nyberg et al., while also acknowledging that there is vaccine-related immunity present in the population.
    To do this, we estimate the proportion of the population in each of the three modelled immunity categories:
        - The "none" immunity stratum - representing the entirely unvaccinated
        - The "low" immunity stratum - representing people who have received two doses of an effective vaccine
            (i.e. the ones used in the UK: ChAdOx, BNT162b2, mRNA1273)
        - The "high" immunity stratum - representing people who have received their third dose of an mRNA vaccine
    The calculation proceeds by working out the proportion of the UK population who would have been in each of these
    three categories at the mid-point of the Nyberg study and using the estimated effect of the vaccine-related immunity
    around then for each of these three groups.
    We then work out the hospitalisation and mortality effect for each of these three groups that would be needed to
    make the weighted average of the hospitalisation proportions come out to the original parameter values (and check
    that this is indeed the case).

    Returns:
        The multipliers for the age-specific proportions for each of the immunity strata

    """

    # Take the complement
    immunity_effect = {k: 1. - v for k, v in source_pop_protection.items()}

    # Work out the adjustments based on the protection provided by immunity
    effective_weights = [immunity_effect[stratum] * source_pop_immunity_dist[stratum] for stratum in IMMUNITY_STRATA]
    no_immunity_modifier = 1. / (sum(effective_weights))
    immune_modifiers = {strat: no_immunity_modifier * immunity_effect[strat] for strat in IMMUNITY_STRATA}

    # Unnecessary check that the weighted average we have calculated does come out to one
    msg = "Values used don't appear to create a weighted average with weights summing to one, something went wrong"
    assert weighted_average(immune_modifiers, source_pop_immunity_dist) == 1., msg

    return immune_modifiers


class SmSirOutputsBuilder(OutputsBuilder):

    def request_cdr(self):
        """
        Register the CDR computed value as a derived output.

        """
        self.model.request_computed_value_output("cdr")

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
        detected_incidence_sources = []
        sympt_incidence_sources = {}
        for agegroup in age_groups:
            sympt_incidence_sources[agegroup] = {}
            for immunity_stratum in IMMUNITY_STRATA:
                sympt_incidence_sources[agegroup][immunity_stratum] = {}
                for strain in strain_strata:
                    sympt_incidence_sources[agegroup][immunity_stratum][strain] = []
                    for clinical_stratum in clinical_strata:

                        # Work out the incidence stratification string
                        stem = f"incidence"
                        agegroup_string = f"Xagegroup_{agegroup}"
                        immunity_string = f"Ximmunity_{immunity_stratum}"
                        clinical_string = f"Xclinical_{clinical_stratum}" if clinical_stratum else ""
                        strain_string = f"Xstrain_{strain}" if strain else ""
                        output_name = stem + agegroup_string + immunity_string + clinical_string + strain_string

                        # Work out the destination criteria
                        dest_filter = {"agegroup": str(agegroup), "immunity": immunity_stratum}
                        clinical_filter = {"clinical": clinical_stratum} if clinical_stratum else {}
                        strain_filter = {"strain": strain} if strain else {}
                        dest_filter.update(clinical_filter)
                        dest_filter.update(strain_filter)

                        # Get the most highly stratified incidence calculation
                        self.model.request_output_for_flow(
                            name=output_name,
                            flow_name=incidence_flow,
                            dest_strata=dest_filter
                        )

                        # Update the dictionaries of which outputs are relevant
                        if clinical_stratum in ["", ClinicalStratum.DETECT]:
                            detected_incidence_sources.append(output_name)
                        if clinical_stratum in ["", ClinicalStratum.SYMPT_NON_DETECT, ClinicalStratum.DETECT]:
                            sympt_incidence_sources[agegroup][immunity_stratum][strain].append(output_name)

        # Compute detected incidence to prepare for notifications calculations
        self.model.request_aggregate_output(
            name="ever_detected_incidence",
            sources=detected_incidence_sources
        )

        # Compute symptomatic incidence by age, immunity and strain status to prepare for hospital outputs calculations
        for agegroup in age_groups:
            for immunity_stratum in IMMUNITY_STRATA:
                for strain in strain_strata:
                    stem = f"incidence_sympt"
                    agegroup_string = f"Xagegroup_{agegroup}"
                    immunity_string = f"Ximmunity_{immunity_stratum}"
                    strain_string = f"Xstrain_{strain}" if strain else ""
                    sympt_inc_name = stem + agegroup_string + immunity_string + strain_string
                    self.model.request_aggregate_output(
                        name=sympt_inc_name,
                        sources=sympt_incidence_sources[agegroup][immunity_stratum][strain]
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

    def request_infection_deaths(
            self,
            ifr_props_params,
            time_from_onset_to_death: TimeDistribution,
            model_times: np.ndarray,
            age_groups: List[int],
            clinical_strata: List[str],
            strain_strata: List[str],
            voc_params: Optional[Dict[str, VocComponent]],
            iso3,
            region,
    ):
        """
        Request infection death-related outputs.

        Args:
            ifr_prop: Infection fatality rates
            time_from_onset_to_death: Details of the statistical distribution for the time to death
            model_times: The model evaluation times
            age_groups: Modelled age group lower breakpoints
            strain_strata: The names of the strains being implemented (or a list of an empty string if no strains)
            clinical_strata: The clinical strata being implemented
            voc_params: The parameters pertaining to the VoCs being implemented in the model

        """

        ifr_request = list(ifr_props_params.values.values())
        ifr_prop = convert_param_agegroups(ifr_request, iso3, region, age_groups, is_80_plus=True)

        # Get the adjustments to the hospitalisation rates according to immunity status
        immune_hosp_modifiers = get_immunity_prop_modifiers(
            ifr_props_params.source_immunity_distribution,
            ifr_props_params.source_immunity_protection,
        )

        # Collate age- and immunity-structured IFRs into a single dictionary
        adj_prop_hosp_among_sympt = {}
        for immunity_stratum in IMMUNITY_STRATA:
            multiplier = immune_hosp_modifiers[immunity_stratum]
            adj_prop = [i_prop * multiplier for i_prop in ifr_prop]
            adj_prop_hosp_among_sympt[immunity_stratum] = apply_odds_ratio_to_props(
                adj_prop,
                ifr_props_params.multiplier
            )

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        interval_distri_densities = precompute_density_intervals(time_from_onset_to_death, model_times)

        # Request infection deaths for each age group
        infection_deaths_sources = []
        for i_age, agegroup in enumerate(age_groups):
            for immunity_stratum in IMMUNITY_STRATA:
                for strain in strain_strata:
                    for clinical_stratum in clinical_strata:

                        # Find the strata we are working with and work out the strings to refer to
                        agegroup_string = f"Xagegroup_{agegroup}"
                        immunity_string = f"Ximmunity_{immunity_stratum}"
                        clinical_string = f"Xclinical_{clinical_stratum}" if clinical_stratum else ""
                        strain_string = f"Xstrain_{strain}" if strain else ""
                        strata_string = agegroup_string + immunity_string + clinical_string + strain_string
                        output_name = "infection_deaths" + strata_string
                        infection_deaths_sources.append(output_name)
                        incidence_name = "incidence" + strata_string

                        # Calculate the multiplier based on age, immunity and strain
                        immunity_risk_modifier = 1. - adj_prop_hosp_among_sympt[immunity_stratum][i_age]
                        strain_risk_modifier = 1. if not strain else 1. - voc_params[strain].death_protection
                        death_risk = ifr_prop[i_age] * immunity_risk_modifier * strain_risk_modifier

                        # Get the infection deaths function
                        infection_deaths_func = make_calc_deaths_func(death_risk, interval_distri_densities)

                        # Request the output
                        self.model.request_function_output(
                            name=output_name,
                            sources=[incidence_name],
                            func=infection_deaths_func
                        )

        # Request aggregated infection deaths
        self.model.request_aggregate_output(
            name="infection_deaths",
            sources=infection_deaths_sources
        )

    def request_hospitalisations(
            self,
            prop_hospital_among_sympt,
            time_from_onset_to_hospitalisation: TimeDistribution,
            hospital_stay_duration: TimeDistribution,
            model_times: np.ndarray,
            age_groups: List[int],
            strain_strata: List[str],
            voc_params: Optional[Dict[str, VocComponent]],
            iso3,
            region,
    ):
        """
        Request hospitalisation-related outputs.

        Args:
            prop_hospital_among_sympt: Proportion ever hospitalised among symptomatic cases (float)
            time_from_onset_to_hospitalisation: Details of the statistical distribution for the time to hospitalisation
            hospital_stay_duration: Details of the statistical distribution for hospitalisation stay duration
            model_times: The model evaluation times
            age_groups: Modelled age group lower breakpoints
            strain_strata: The names of the strains being implemented (or a list of an empty string if no strains)
            voc_params: The parameters pertaining to the VoCs being implemented in the model

        """

        hosp_request = list(prop_hospital_among_sympt.values.values())
        hosp_props = convert_param_agegroups(hosp_request, iso3, region, age_groups, is_80_plus=True)

        # Get the adjustments to the hospitalisation rates according to immunity status
        immune_hosp_modifiers = get_immunity_prop_modifiers(
            prop_hospital_among_sympt.source_immunity_distribution,
            prop_hospital_among_sympt.source_immunity_protection,
        )

        # Collate age- and immunity-structured hospitalisation rates into a single dictionary
        adj_prop_hosp_among_sympt = {}
        for immunity_stratum in IMMUNITY_STRATA:
            multiplier = immune_hosp_modifiers[immunity_stratum]
            adj_prop = [i_prop * multiplier for i_prop in hosp_props]
            adj_prop_hosp_among_sympt[immunity_stratum] = apply_odds_ratio_to_props(
                adj_prop,
                prop_hospital_among_sympt.multiplier
            )

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        interval_distri_densities = precompute_density_intervals(time_from_onset_to_hospitalisation, model_times)

        # Request hospital admissions for each age group
        hospital_admissions_sources = []
        for i_age, agegroup in enumerate(age_groups):
            for immunity_stratum in IMMUNITY_STRATA:
                for strain in strain_strata:

                    # Find the strata we are working with and work out the strings to refer to
                    agegroup_string = f"Xagegroup_{agegroup}"
                    immunity_string = f"Ximmunity_{immunity_stratum}"
                    strain_string = f"Xstrain_{strain}" if strain else ""
                    strata_string = agegroup_string + immunity_string + strain_string
                    output_name = "hospital_admissions" + strata_string
                    hospital_admissions_sources.append(output_name)
                    incidence_name = "incidence_sympt" + strata_string

                    # Calculate the multiplier based on age, immunity and strain
                    strain_risk_modifier = 1. if not strain else 1. - voc_params[strain].hosp_protection
                    hospital_risk = adj_prop_hosp_among_sympt[immunity_stratum][i_age] * strain_risk_modifier

                    # Get the hospitalisation function
                    hospital_admissions_func = make_calc_admissions_func(hospital_risk, interval_distri_densities)

                    # Request the output
                    self.model.request_function_output(
                        name=output_name,
                        sources=[incidence_name],
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

    def request_recovered_proportion(self, base_comps: List[str]):
        """
        Track the total population ever infected and the proportion of the total population.

        Args:
             base_comps: The unstratified model compartments

        """

        # All the compartments other than the fully susceptible have been infected at least once
        recovered_compartments = [comp for comp in base_comps if comp != Compartment.SUSCEPTIBLE]

        self.model.request_output_for_compartments(
            "ever_infected",
            recovered_compartments,
        )
        self.model.request_function_output(
            "prop_ever_infected",
            lambda infected, total: infected / total,
            sources=["ever_infected", "total_population"],
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


def make_calc_deaths_func(death_risk, density_intervals):
    def deaths_func(detected_incidence):
        deaths = apply_convolution_for_event(detected_incidence, density_intervals, death_risk)
        return deaths

    return deaths_func


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
