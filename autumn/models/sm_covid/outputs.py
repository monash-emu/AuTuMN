from typing import List, Dict, Optional, Union

from scipy import stats
import numpy as np
from numba import jit

from autumn.model_features.outputs import OutputsBuilder
from autumn.models.sm_sir.parameters import TimeDistribution, VocComponent, AgeSpecificProps
from .constants import IMMUNITY_STRATA, Compartment, ImmunityStratum
from autumn.core.utils.utils import weighted_average, get_apply_odds_ratio_to_prop
from autumn.models.sm_sir.stratifications.agegroup import convert_param_agegroups


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
    no_immunity_modifier = 1. / sum(effective_weights)
    immune_modifiers = {strat: no_immunity_modifier * immunity_effect[strat] for strat in IMMUNITY_STRATA}

    # Unnecessary check that the weighted average we have calculated does come out to one
    msg = "Values used don't appear to create a weighted average with weights summing to one, something went wrong"
    assert weighted_average(immune_modifiers, source_pop_immunity_dist, rounding=4) == 1., msg

    return immune_modifiers


class SmCovidOutputsBuilder(OutputsBuilder):
   
    def request_incidence(
            self,      
            age_groups: List[str],     
            strain_strata: List[str], 
            incidence_flow: str,
            request_incidence_by_age: bool,
    ):
        """
        Calculate incident disease cases. This is associated with the transition to infectiousness if there is only one
        infectious compartment, or transition between the two if there are two.
        Note that this differs from the approach in the covid_19 model, which took entry to the first "active"
        compartment to represent the onset of symptoms, which infectiousness starting before this.

        Args:
            age_groups: The modelled age groups
            strain_strata: The modelled strains
            incidence_flow: The name of the flow representing incident cases
            request_incidence_by_age: Whether to save outputs for incidence by age

        """

        # Unstratified
        self.model.request_output_for_flow(name="incidence", flow_name=incidence_flow)

        # Stratified
        for agegroup in age_groups:
            agegroup_string = f"Xagegroup_{agegroup}"        
            age_incidence_sources = []    

            for immunity_stratum in IMMUNITY_STRATA:
                immunity_string = f"Ximmunity_{immunity_stratum}"                
                
                for strain in strain_strata:
                    strain_string = f"Xstrain_{strain}" if strain else ""
                    strain_filter = {"strain": strain} if strain else {}

                    dest_filter = {"agegroup": agegroup, "immunity": immunity_stratum}
                    dest_filter.update(strain_filter)

                    output_name = f"incidence{agegroup_string}{immunity_string}{strain_string}"
                    age_incidence_sources.append(output_name)

                    self.model.request_output_for_flow(
                        name=output_name,
                        flow_name=incidence_flow,
                        dest_strata=dest_filter,
                        save_results=False,
                    ) 

            # Aggregated incidence by age
            if request_incidence_by_age:
                self.model.request_aggregate_output(
                    name=f"incidence{agegroup_string}",
                    sources=age_incidence_sources,
                    save_results=True,
                )     


    def request_infection_deaths(
            self,
            model_times: np.ndarray,
            age_groups: List[str],
            strain_strata: List[str],
            iso3: str,
            region: Union[str, None],
            ifr_prop_requests: AgeSpecificProps,
            ve_death: float,
            time_from_onset_to_death: TimeDistribution,
            voc_params: Optional[Dict[str, VocComponent]],
    ):
        """
        Request infection death-related outputs.

        Args:
            model_times: The model evaluation times
            age_groups: Modelled age group lower breakpoints
            strain_strata: The modelled strains
            iso3: The ISO3 code of the country being simulated
            region: The sub-region being simulated, if any
            ifr_prop_requests: All the CFR-related requests, including the proportions themselves
            ve_death: Vaccine efficacy against mortality
            time_from_onset_to_death: Details of the statistical distribution for the time to death
            voc_params: The strain-specific parameters

        """

        age_ifr_request = ifr_prop_requests.values
        age_ifr_props = convert_param_agegroups(iso3, region, age_ifr_request, age_groups)
        ifr_multiplier = ifr_prop_requests.multiplier

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        interval_distri_densities = precompute_density_intervals(time_from_onset_to_death, model_times)

        # Prepare immunity modifiers
        immune_death_modifiers = {
            ImmunityStratum.UNVACCINATED: 1.,
            ImmunityStratum.VACCINATED: 1. - ve_death,
        }

        # Request infection deaths for each age group
        infection_deaths_sources = []
        for agegroup in age_groups:
            agegroup_string = f"Xagegroup_{agegroup}"            

            for immunity_stratum in IMMUNITY_STRATA:
                immunity_string = f"Ximmunity_{immunity_stratum}"

                # Adjust CFR proportions for immunity
                age_immune_death_props = age_ifr_props * immune_death_modifiers[immunity_stratum]

                for strain in strain_strata:
                    strain_string = f"Xstrain_{strain}" if strain else ""

                    # Find the strata we are working with and work out the strings to refer to
                    strata_string = f"{agegroup_string}{immunity_string}{strain_string}"

                    output_name = f"infection_deaths{strata_string}"
                    infection_deaths_sources.append(output_name)

                    # Calculate the multiplier based on age, immunity and strain
                    strain_risk_modifier = 1. if not strain else voc_params[strain].death_risk_adjuster
                    death_risk = age_immune_death_props[agegroup] * strain_risk_modifier * ifr_multiplier

                    assert death_risk <= 1., "The overall death risk is greater than one."

                    # Get the infection deaths function for convolution
                    infection_deaths_func = make_calc_deaths_func(death_risk, interval_distri_densities)

                    # Request the output
                    self.model.request_function_output(
                        name=output_name,
                        sources=[f"incidence{strata_string}"],
                        func=infection_deaths_func,
                        save_results=False,
                    )

        # Request aggregated infection deaths
        self.model.request_aggregate_output(
            name="infection_deaths",
            sources=infection_deaths_sources
        )

    def request_hospitalisations(
            self,
            model_times: np.ndarray,
            age_groups: List[int],
            strain_strata: List[str],
            iso3: str,
            region: Union[str, None],
            symptomatic_prop_requests: Dict[str, float],
            hosp_prop_requests: AgeSpecificProps,
            ve_hospitalisation: float,
            time_from_onset_to_hospitalisation: TimeDistribution,
            hospital_stay_duration: TimeDistribution,
            voc_params: Optional[Dict[str, VocComponent]],
    ):
        """
        Request hospitalisation-related outputs.

        Args:
            model_times: The model evaluation times
            age_groups: Modelled age group lower breakpoints
            strain_strata: The names of the strains being implemented (or a list of an empty string if no strains)
            iso3: The ISO3 code of the country being simulated
            region: The sub-region being simulated, if any
            symptomatic_prop_requests: The symptomatic proportions by age
            hosp_prop_requests: The hospitalisation proportions given symptomatic infection
            time_from_onset_to_hospitalisation: Details of the statistical distribution for the time to hospitalisation
            hospital_stay_duration: Details of the statistical distribution for hospitalisation stay duration
            voc_params: The parameters pertaining to the VoCs being implemented in the model

        """

        symptomatic_props = convert_param_agegroups(iso3, region, symptomatic_prop_requests, age_groups)
        hosp_props = convert_param_agegroups(iso3, region, hosp_prop_requests.values, age_groups)

        # Get the adjustments to the hospitalisation rates according to immunity status
        or_adjuster_func = get_apply_odds_ratio_to_prop(hosp_prop_requests.multiplier)

        # Prepare immunity modifiers
        immune_hosp_modifiers = {
            ImmunityStratum.UNVACCINATED: 1.,
            ImmunityStratum.VACCINATED: 1. - ve_hospitalisation,
        }

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        interval_distri_densities = precompute_density_intervals(time_from_onset_to_hospitalisation, model_times)

        # Request hospital admissions for each age group
        hospital_admissions_sources = []
        for agegroup in age_groups:
            agegroup_string = f"Xagegroup_{agegroup}"

            for immunity_stratum in IMMUNITY_STRATA:
                immunity_string = f"Ximmunity_{immunity_stratum}"

                # Adjust the hospitalisation proportions for immunity
                adj_hosp_props = hosp_props * immune_hosp_modifiers[immunity_stratum]
                adj_hosp_props = adj_hosp_props.apply(or_adjuster_func)

                for strain in strain_strata:
                    strain_string = f"Xstrain_{strain}" if strain else ""

                    # Find the strata we are working with and work out the strings to refer to
                    strata_string = f"{agegroup_string}{immunity_string}{strain_string}"
                    output_name = f"hospital_admissions{strata_string}"
                    hospital_admissions_sources.append(output_name)

                    # Calculate the multiplier based on age, immunity and strain
                    strain_risk_modifier = 1. if not strain else voc_params[strain].hosp_risk_adjuster                    
                    hospital_risk_given_symptoms = adj_hosp_props[agegroup] * strain_risk_modifier
                    hospital_risk_given_infection = hospital_risk_given_symptoms * symptomatic_props[agegroup]

                    # Get the hospitalisation function
                    hospital_admissions_func = make_calc_admissions_func(hospital_risk_given_infection, interval_distri_densities)

                    # Request the output
                    self.model.request_function_output(
                        name=output_name,
                        sources=[f"incidence{strata_string}"],
                        func=hospital_admissions_func,
                        save_results=False,
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


    def request_peak_hospital_occupancy(self):
        """
        Create an output for the peak hospital occupancy. This is stored as a timeseries, although this is 
        actually a constant output.
        """
        self.model.request_function_output(
            'peak_hospital_occupancy',
            lambda hosp_occupancy: np.repeat(hosp_occupancy.max(), hosp_occupancy.size),
            ["hospital_occupancy"],
        )

    # def request_icu_outputs(
    #     self,
    #     prop_icu_among_hospitalised: float,
    #     time_from_hospitalisation_to_icu: TimeDistribution,
    #     icu_stay_duration: TimeDistribution,
    #     strain_strata: List[str],
    #     model_times: np.ndarray,
    #     voc_params: Optional[Dict[str, VocComponent]],
    #     age_groups: List[int],
    # ):
    #     """
    #     Request ICU-related outputs.

    #     Args:
    #         prop_icu_among_hospitalised: Proportion ever requiring ICU stay among hospitalised cases (float)
    #         time_from_hospitalisation_to_icu: Details of the statistical distribution for the time to ICU admission
    #         icu_stay_duration: Details of the statistical distribution for ICU stay duration
    #         strain_strata: The names of the strains being implemented (or a list of an empty string if no strains)
    #         model_times: The model evaluation times
    #         voc_params: The parameters pertaining to the VoCs being implemented in the model
    #         age_groups: Modelled age group lower breakpoints
    #     """

    #     # Pre-compute the probabilities of event occurrence within each time interval between model times
    #     interval_distri_densities = precompute_density_intervals(time_from_hospitalisation_to_icu, model_times)

    #     icu_admissions_sources = []
    #     for agegroup in age_groups:
    #         agegroup_string = f"Xagegroup_{agegroup}"

    #         for immunity_stratum in IMMUNITY_STRATA:
    #             immunity_string = f"Ximmunity_{immunity_stratum}"

    #             for strain in strain_strata:
    #                 strain_string = f"Xstrain_{strain}" if strain else ""
    #                 strata_string = f"{agegroup_string}{immunity_string}{strain_string}"
    #                 output_name = f"icu_admissions{strata_string}"
    #                 icu_admissions_sources.append(output_name)

    #                 # Calculate the multiplier based on age, immunity and strain
    #                 strain_risk_modifier = 1. if not strain else voc_params[strain].icu_multiplier
    #                 icu_risk = prop_icu_among_hospitalised * strain_risk_modifier

    #                 # Request ICU admissions
    #                 icu_admissions_func = make_calc_admissions_func(icu_risk, interval_distri_densities)
    #                 self.model.request_function_output(
    #                     name=output_name,
    #                     sources=[f"hospital_admissions{strata_string}"],
    #                     func=icu_admissions_func,
    #                     save_results=False,
    #                 )

    #     # Request aggregated icu admissions
    #     self.model.request_aggregate_output(
    #         name="icu_admissions",
    #         sources=icu_admissions_sources,
    #     )

    #     # Request ICU occupancy
    #     probas_stay_greater_than = precompute_probas_stay_greater_than(icu_stay_duration, model_times)
    #     icu_occupancy_func = make_calc_occupancy_func(probas_stay_greater_than)
    #     self.model.request_function_output(
    #         name="icu_occupancy",
    #         sources=["icu_admissions"],
    #         func=icu_occupancy_func,
    #     )

    def request_recovered_proportion(self, base_comps: List[str]):
        """
        Track the total population ever infected and the proportion of the total population.

        Args:
             base_comps: The unstratified model compartments

        """

        # All the compartments other than the fully susceptible have been infected at least once
        ever_infected_compartments = [comp for comp in base_comps if comp != Compartment.SUSCEPTIBLE]

        self.model.request_output_for_compartments(
            "ever_infected",
            ever_infected_compartments,
        )
        self.model.request_function_output(
            "prop_ever_infected",
            lambda infected, total: infected / total,
            sources=["ever_infected", "total_population"],
        )


    def request_random_process_outputs(self,):
        self.model.request_computed_value_output("transformed_random_process")


    def request_immunity_props(self, immunity_strata, age_pops, request_immune_prop_by_age):
        """
        Track population distribution across immunity stratification, to make sure vaccination stratification is working
        correctly.

        Args:
            strata: Immunity strata being implemented in the model
            age_pops: Population size by age group
            request_immune_prop_by_age: Whether to request age-specific immunity proportions

        """

        for immunity_stratum in immunity_strata:
            n_immune_name = f"n_immune_{immunity_stratum}"
            prop_immune_name = f"prop_immune_{immunity_stratum}"
            self.model.request_output_for_compartments(
                n_immune_name,
                self.compartments,
                {"immunity": immunity_stratum},
            )
            self.model.request_function_output(
                prop_immune_name,
                lambda num, total: num / total,
                [n_immune_name, "total_population"],
            )

            # Calculate age-specific proportions if requested
            if request_immune_prop_by_age:
                for agegroup, popsize in age_pops.items():
                    n_age_immune_name = f"n_immune_{immunity_stratum}Xagegroup_{agegroup}"
                    prop_age_immune_name = f"prop_immune_{immunity_stratum}Xagegroup_{agegroup}"
                    self.model.request_output_for_compartments(
                        n_age_immune_name,
                        self.compartments,
                        {"immunity": immunity_stratum, "agegroup": agegroup},
                        save_results=False,
                    )
                    self.model.request_function_output(
                        prop_age_immune_name,
                        make_age_immune_prop_func(popsize),
                        [n_age_immune_name],
                    )


    def request_cumulative_outputs(self, requested_cumulative_outputs, cumulative_start_time):
        """
        Compute cumulative outputs for requested outputs.

        Args:
            requested_cumulative_outputs: List of requested derived outputs to accumulate
            cumulative_start_time: reference time for cumulative output calculation
        """

        for output in requested_cumulative_outputs:
            self.model.request_cumulative_output(name=f"cumulative_{output}", source=output, start_time=cumulative_start_time)


    def request_student_weeks_missed_output(self, n_student_weeks_missed):
        """
            Store the number of students*weeks of school missed. This is a single float that will be stored as a derived output
        """
        self.model.request_function_output(
            'student_weeks_missed',
            lambda total: np.repeat(n_student_weeks_missed, total.size),  # constant function
            ["total_population"],  # could be anything here, really...
        )


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

@jit
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


def make_age_immune_prop_func(popsize):
    """
    Create a simple function to calculate immune proportion for a given age group 

    Args:
        popsize: Population size of the relevant age group
    
    Returns:
        A function converitng a number of individuals into the associated proportion among the relevant age group 
    """
    def age_immune_prop_func(n_immune):
        return n_immune / popsize

    return age_immune_prop_func