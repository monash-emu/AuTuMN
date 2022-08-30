from typing import List, Dict, Optional, Union

# from scipy import stats
import numpy as np

from numba import jit as numbajit

from autumn.model_features.outputs import OutputsBuilder
from autumn.models.sm_jax.parameters import TimeDistribution, VocComponent, AgeSpecificProps
from .constants import IMMUNITY_STRATA, Compartment, ClinicalStratum
from autumn.core.utils.utils import weighted_average, get_apply_odds_ratio_to_prop
from autumn.models.sm_jax.stratifications.agegroup import convert_param_agegroups

from summer.parameters.params import Function, Data, Parameter, DerivedOutput

from computegraph import jaxify

modules = jaxify.get_modules()
scipy = modules["scipy"]
jnp = modules["numpy"]


def gamma_cdf(shape, scale, x):
    return scipy.special.gammainc(shape, x / scale)


def apply_odds_ratio_to_proportion(proportion: float, odds_ratio: float) -> float:
    """
    Use an odds ratio to adjust a proportion.

    Starts from the premise that the odds associated with the original proportion (p1) = p1 / (1 - p1)
    and similarly, that the odds associated with the adjusted proportion (p2) = p2 / (1 - p2)
    We want to multiply the odds associated with p1 by a certain odds ratio.
    That, is we need to solve the following equation for p2:
        p1 / (1 - p1) * OR = p2 / (1 - p2)
    Solving the above for p2:
        p2 = p1 * OR / (p1 * (OR - 1) + 1)

    Args:
        proportion: The original proportion (p1 in the description above)
        odds_ratio: The odds ratio to adjust by (OR above)
    Returns:
        The adjusted proportion (p2 above)

    """

    # Transform and return
    return proportion * odds_ratio / (proportion * (odds_ratio - 1.0) + 1.0)


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
    immunity_effect = {k: 1.0 - v for k, v in source_pop_protection.items()}

    # Work out the adjustments based on the protection provided by immunity
    effective_weights = [
        immunity_effect[stratum] * source_pop_immunity_dist[stratum] for stratum in IMMUNITY_STRATA
    ]
    no_immunity_modifier = 1.0 / sum(effective_weights)
    immune_modifiers = {
        strat: no_immunity_modifier * immunity_effect[strat] for strat in IMMUNITY_STRATA
    }

    # Unnecessary check that the weighted average we have calculated does come out to one
    msg = "Values used don't appear to create a weighted average with weights summing to one, something went wrong"
    assert weighted_average(immune_modifiers, source_pop_immunity_dist, rounding=4) == 1.0, msg

    return immune_modifiers


class SmSirOutputsBuilder(OutputsBuilder):
    def get_strata_for_strat(self, strat_name: str):
        stratification = self.model.get_stratification(strat_name)
        if stratification:
            return stratification.strata
        else:
            return None

    def request_cdr(self):
        """
        Register the CDR computed value as a derived output.

        """
        self.model.request_computed_value_output("cdr")

    def request_incidence(
        self,
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
            clinical_strata: The clinical strata implemented
            strain_strata: The modelled strains, or None if model is not stratified by strain
            incidence_flow: The name of the flow representing incident cases
            request_incidence_by_age: Whether to save outputs for incidence by age

        """

        age_groups = self.get_strata_for_strat("agegroup")
        clinical_strata = self.get_strata_for_strat("clinical") or [""]
        strain_strata = self.get_strata_for_strat("strain") or [""]
        immunity_strata = self.get_strata_for_strat("immunity") or [""]

        # Unstratified
        self.model.request_output_for_flow(name="incidence", flow_name=incidence_flow)

        # Stratified
        detected_incidence_sources = (
            []
        )  # Collect detected incidence unstratified for notifications calculation

        for agegroup in age_groups:
            agegroup_string = f"Xagegroup_{agegroup}"
            agegroup_filter = {"agegroup": agegroup}

            age_incidence_sources = []

            for immunity_stratum in immunity_strata:
                immunity_string = f"Ximmunity_{immunity_stratum}" if immunity_stratum else ""
                immunity_filter = {"immunity": immunity_stratum} if immunity_stratum else {}

                for strain in strain_strata:
                    strain_string = f"Xstrain_{strain}" if strain else ""
                    strain_filter = {"strain": strain} if strain else {}
                    sympt_incidence_sources = []

                    for clinical_stratum in clinical_strata:
                        clinical_string = (
                            f"Xclinical_{clinical_stratum}" if clinical_stratum else ""
                        )

                        # Work out what to filter by, depending on whether these stratifications have been implemented
                        dest_filter = {"clinical": clinical_stratum} if clinical_stratum else {}
                        dest_filter.update(agegroup_filter)
                        dest_filter.update(immunity_filter)
                        dest_filter.update(strain_filter)

                        # Work out the fully stratified incidence string
                        output_name = f"incidence{agegroup_string}{immunity_string}{clinical_string}{strain_string}"
                        age_incidence_sources.append(output_name)

                        # Get the most highly stratified incidence calculation
                        self.model.request_output_for_flow(
                            name=output_name,
                            flow_name=incidence_flow,
                            dest_strata=dest_filter,
                            save_results=False,
                        )

                        # Update the dictionaries of which outputs are relevant
                        if clinical_stratum in ["", ClinicalStratum.DETECT]:
                            detected_incidence_sources.append(output_name)
                        if clinical_stratum in [
                            "",
                            ClinicalStratum.SYMPT_NON_DETECT,
                            ClinicalStratum.DETECT,
                        ]:
                            sympt_incidence_sources.append(output_name)

                    # Calculate the incidence of symptomatic disease in preparation for hospitalisation and deaths
                    sympt_inc_name = (
                        f"incidence_sympt{agegroup_string}{immunity_string}{strain_string}"
                    )
                    self.model.request_aggregate_output(
                        name=sympt_inc_name,
                        sources=sympt_incidence_sources,
                        save_results=False,
                    )

            if request_incidence_by_age:
                self.model.request_aggregate_output(
                    name=f"incidence{agegroup_string}",
                    sources=age_incidence_sources,
                    save_results=True,
                )

        # Compute detected incidence to prepare for notifications calculations
        self.model.request_aggregate_output(
            name="ever_detected_incidence", sources=detected_incidence_sources
        )

    def request_notifications(self, time_from_onset_to_notification: TimeDistribution):
        """
        Request notification calculations.

        Args:
            time_from_onset_to_notification: Details of the statistical distribution used to model time to notification
            model_times: The model evaluation times

        """

        model_times = self.model.times

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        density_intervals = precompute_density_intervals(
            time_from_onset_to_notification, model_times
        )

        notifications_density_intervals = Data(density_intervals)

        # Request notifications
        # notifications_func = make_calc_notifications_func(density_intervals)

        notifications_func = Function(
            convolve_probability,
            [
                DerivedOutput("ever_detected_incidence"),
                notifications_density_intervals,
            ],
        )

        self.model.request_param_function_output(
            name="notifications",
            func=notifications_func,
        )

    def request_infection_deaths(
        self,
        model_times: np.ndarray,
        age_groups: List[str],
        strain_strata: List[str],
        iso3: str,
        region: Union[str, None],
        cfr_prop_requests: AgeSpecificProps,
        time_from_onset_to_death: TimeDistribution,
        voc_params: Optional[Dict[str, VocComponent]],
        request_infection_deaths_by_age: bool,
    ):
        """
        Request infection death-related outputs.

        Args:
            model_times: The model evaluation times
            age_groups: Modelled age group lower breakpoints
            strain_strata: The names of the strains being implemented (or a list of an empty string if no strains)
            iso3: The ISO3 code of the country being simulated
            region: The sub-region being simulated, if any
            cfr_prop_requests: All the CFR-related requests, including the proportions themselves
            time_from_onset_to_death: Details of the statistical distribution for the time to death
            voc_params: The parameters pertaining to the VoCs being implemented in the model
            request_infection_deaths_by_age: Whether to save outputs for infection deaths by age

        """

        cfr_request = cfr_prop_requests.values
        cfr_props = convert_param_agegroups(iso3, region, cfr_request, age_groups)

        # Get the adjustments to the hospitalisation rates according to immunity status
        source_immunity_dist = cfr_prop_requests.source_immunity_distribution
        source_immunity_protection = cfr_prop_requests.source_immunity_protection
        immune_death_modifiers = get_immunity_prop_modifiers(
            source_immunity_dist, source_immunity_protection
        )

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        interval_distri_densities = precompute_density_intervals(
            time_from_onset_to_death, model_times
        )

        # Request infection deaths for each age group
        infection_deaths_sources = []
        for agegroup in age_groups:
            agegroup_string = f"Xagegroup_{agegroup}"

            age_infection_deaths_sources = []

            for immunity_stratum in IMMUNITY_STRATA:
                immunity_string = f"Ximmunity_{immunity_stratum}"

                # Adjust CFR proportions for immunity
                adj_death_props = cfr_props * immune_death_modifiers[immunity_stratum]

                or_adjuster_func = get_apply_odds_ratio_to_prop(cfr_prop_requests.multiplier)
                odds_ratio = cfr_prop_requests.multiplier
                # adj_death_props = {k:}
                adj_death_props = adj_death_props.apply(or_adjuster_func)

                for strain in strain_strata:
                    strain_string = f"Xstrain_{strain}" if strain else ""

                    # Find the strata we are working with and work out the strings to refer to
                    strata_string = f"{agegroup_string}{immunity_string}{strain_string}"
                    output_name = f"infection_deaths{strata_string}"

                    infection_deaths_sources.append(output_name)
                    age_infection_deaths_sources.append(output_name)

                    # Calculate the multiplier based on age, immunity and strain
                    strain_risk_modifier = (
                        1.0 if not strain else 1.0 - voc_params[strain].death_protection
                    )
                    death_risk = adj_death_props[agegroup] * strain_risk_modifier

                    # Get the infection deaths function for convolution
                    infection_deaths_func = make_calc_deaths_func(
                        death_risk, interval_distri_densities
                    )

                    # Request the output
                    self.model.request_function_output(
                        name=output_name,
                        sources=[f"incidence_sympt{strata_string}"],
                        func=infection_deaths_func,
                        save_results=False,
                    )

            # Request infection deaths by age
            if request_infection_deaths_by_age:
                self.model.request_aggregate_output(
                    name=f"infection_deaths{agegroup_string}",
                    sources=age_infection_deaths_sources,
                    save_results=True,
                )

        # Request param func output that consumes all

        # Request aggregated infection deaths
        self.model.request_aggregate_output(
            name="infection_deaths", sources=infection_deaths_sources
        )

    def request_hospitalisations(
        self,
        model_times: np.ndarray,
        age_groups: List[int],
        strain_strata: List[str],
        iso3: str,
        region: Union[str, None],
        hosp_prop_requests: AgeSpecificProps,
        time_from_onset_to_hospitalisation: TimeDistribution,
        hospital_stay_duration: TimeDistribution,
        voc_params: Optional[Dict[str, VocComponent]],
        request_hospital_admissions_by_age: bool,
        request_hospital_occupancy_by_age: bool,
    ):
        """
        Request hospitalisation-related outputs.

        Args:
            model_times: The model evaluation times
            age_groups: Modelled age group lower breakpoints
            strain_strata: The names of the strains being implemented (or a list of an empty string if no strains)
            iso3: The ISO3 code of the country being simulated
            region: The sub-region being simulated, if any
            hosp_prop_requests: The hospitalisation proportion-related requests, including the proportions themselves
            time_from_onset_to_hospitalisation: Details of the statistical distribution for the time to hospitalisation
            hospital_stay_duration: Details of the statistical distribution for hospitalisation stay duration
            voc_params: The parameters pertaining to the VoCs being implemented in the model
            request_hospital_admissions_by_age: Whether to save outputs for hospital admissions by age
            request_hospital_occupancy_by_age: Whether to save outputs for hospital occupancy by age

        """

        hosp_request = hosp_prop_requests.values
        hosp_props = convert_param_agegroups(iso3, region, hosp_request, age_groups)

        # Get the adjustments to the hospitalisation rates according to immunity status
        source_immunity_dist = hosp_prop_requests.source_immunity_distribution
        source_immunity_protection = hosp_prop_requests.source_immunity_protection
        immune_hosp_modifiers = get_immunity_prop_modifiers(
            source_immunity_dist, source_immunity_protection
        )

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        interval_distri_densities = precompute_density_intervals(
            time_from_onset_to_hospitalisation, model_times
        )

        # Functions to retrieve hospital occupancy from hospital admissions
        probas_stay_greater_than = precompute_probas_stay_greater_than(
            hospital_stay_duration, model_times
        )
        hospital_occupancy_func = make_calc_occupancy_func(probas_stay_greater_than)

        # Request hospital admissions for each age group
        hospital_admissions_sources = []
        for agegroup in age_groups:
            agegroup_string = f"Xagegroup_{agegroup}"

            age_hospital_admissions_sources = []

            for immunity_stratum in IMMUNITY_STRATA:
                immunity_string = f"Ximmunity_{immunity_stratum}"

                # Adjust the hospitalisation proportions for immunity
                adj_hosp_props = hosp_props * immune_hosp_modifiers[immunity_stratum]
                or_adjuster_func = get_apply_odds_ratio_to_prop(hosp_prop_requests.multiplier)
                adj_hosp_props = adj_hosp_props.apply(or_adjuster_func)

                for strain in strain_strata:
                    strain_string = f"Xstrain_{strain}" if strain else ""

                    # Find the strata we are working with and work out the strings to refer to
                    strata_string = f"{agegroup_string}{immunity_string}{strain_string}"
                    output_name = f"hospital_admissions{strata_string}"
                    hospital_admissions_sources.append(output_name)
                    age_hospital_admissions_sources.append(output_name)

                    # Calculate the multiplier based on age, immunity and strain
                    strain_risk_modifier = (
                        1.0 if not strain else 1.0 - voc_params[strain].hosp_protection
                    )
                    hospital_risk = adj_hosp_props[agegroup] * strain_risk_modifier

                    # Get the hospitalisation function
                    hospital_admissions_func = make_calc_admissions_func(
                        hospital_risk, interval_distri_densities
                    )

                    # Request the output
                    self.model.request_function_output(
                        name=output_name,
                        sources=[f"incidence_sympt{strata_string}"],
                        func=hospital_admissions_func,
                        save_results=False,
                    )

            # Request hospital admissions by age
            if request_hospital_admissions_by_age:
                self.model.request_aggregate_output(
                    name=f"hospital_admissions{agegroup_string}",
                    sources=age_hospital_admissions_sources,
                    save_results=True,
                )

            # Request hospital occupancy by age
            if request_hospital_occupancy_by_age:
                self.model.request_function_output(
                    name=f"hospital_occupancy{agegroup_string}",
                    sources=[f"hospital_admissions{agegroup_string}"],
                    func=hospital_occupancy_func,
                )

        # Request aggregated hospital admissions
        self.model.request_aggregate_output(
            name="hospital_admissions", sources=hospital_admissions_sources
        )

        # Request aggregated hospital occupancy
        self.model.request_function_output(
            name="hospital_occupancy", sources=["hospital_admissions"], func=hospital_occupancy_func
        )

    def request_icu_outputs(
        self,
        prop_icu_among_hospitalised: float,
        time_from_hospitalisation_to_icu: TimeDistribution,
        icu_stay_duration: TimeDistribution,
        strain_strata: List[str],
        model_times: np.ndarray,
        voc_params: Optional[Dict[str, VocComponent]],
        age_groups: List[int],
        request_icu_admissions_by_age: bool,
        request_icu_occupancy_by_age: bool,
    ):
        """
        Request ICU-related outputs.

        Args:
            prop_icu_among_hospitalised: Proportion ever requiring ICU stay among hospitalised cases (float)
            time_from_hospitalisation_to_icu: Details of the statistical distribution for the time to ICU admission
            icu_stay_duration: Details of the statistical distribution for ICU stay duration
            strain_strata: The names of the strains being implemented (or a list of an empty string if no strains)
            model_times: The model evaluation times
            voc_params: The parameters pertaining to the VoCs being implemented in the model
            age_groups: Modelled age group lower breakpoints
            request_icu_admissions_by_age: Whether to save outputs for ICU admissions by age
            request_icu_occupancy_by_age: Whether to save outputs for ICU occupancy by age

        """

        # Pre-compute the probabilities of event occurrence within each time interval between model times
        interval_distri_densities = precompute_density_intervals(
            time_from_hospitalisation_to_icu, model_times
        )

        # Functions to retrieve ICU occupancy from ICU admissions
        probas_stay_greater_than = precompute_probas_stay_greater_than(
            icu_stay_duration, model_times
        )
        icu_occupancy_func = make_calc_occupancy_func(probas_stay_greater_than)

        icu_admissions_sources = []
        for agegroup in age_groups:
            agegroup_string = f"Xagegroup_{agegroup}"

            age_icu_admissions_sources = []

            for immunity_stratum in IMMUNITY_STRATA:
                immunity_string = f"Ximmunity_{immunity_stratum}"

                for strain in strain_strata:
                    strain_string = f"Xstrain_{strain}" if strain else ""
                    strata_string = f"{agegroup_string}{immunity_string}{strain_string}"
                    output_name = f"icu_admissions{strata_string}"
                    icu_admissions_sources.append(output_name)
                    age_icu_admissions_sources.append(output_name)

                    # Calculate the multiplier based on age, immunity and strain
                    strain_risk_modifier = 1.0 if not strain else voc_params[strain].icu_multiplier
                    icu_risk = prop_icu_among_hospitalised * strain_risk_modifier

                    # Request ICU admissions
                    icu_admissions_func = make_calc_admissions_func(
                        icu_risk, interval_distri_densities
                    )
                    self.model.request_function_output(
                        name=output_name,
                        sources=[f"hospital_admissions{strata_string}"],
                        func=icu_admissions_func,
                        save_results=False,
                    )

            # Request ICU admissions by age
            if request_icu_admissions_by_age:
                self.model.request_aggregate_output(
                    name=f"icu_admissions{agegroup_string}",
                    sources=age_icu_admissions_sources,
                    save_results=True,
                )

            # Request ICU occupancy by age
            if request_icu_occupancy_by_age:
                self.model.request_function_output(
                    name=f"icu_occupancy{agegroup_string}",
                    sources=[f"icu_admissions{agegroup_string}"],
                    func=icu_occupancy_func,
                )

        # Request aggregated icu admissions
        self.model.request_aggregate_output(
            name="icu_admissions",
            sources=icu_admissions_sources,
        )

        # Request ICU occupancy
        self.model.request_function_output(
            name="icu_occupancy",
            sources=["icu_admissions"],
            func=icu_occupancy_func,
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

    def request_random_process_outputs(
        self,
    ):
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
            self.model.request_cumulative_output(
                name=f"cumulative_{output}", source=output, start_time=cumulative_start_time
            )


def build_statistical_distribution(distribution_details: TimeDistribution):
    """
    Generate a scipy statistical distribution object that can then be used multiple times to evaluate the cdf

    Args:
        distribution_details: User request parameters that define the distribution

    Returns:
        A scipy statistical distribution object

    """
    from scipy import stats

    if distribution_details.distribution == "gamma":
        shape = distribution_details.shape.value
        scale = distribution_details.mean.value / shape
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
    lags = model_times - model_times[0]
    cdf_values = distribution.cdf(lags)
    interval_distri_densities = np.gradient(cdf_values)
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
    lags = model_times - model_times[0]
    cdf_values = distribution.cdf(lags)
    probas_stay_greater_than = 1.0 - cdf_values
    return probas_stay_greater_than


@numbajit
def _convolve_probability(
    source_output: np.ndarray, density_intervals: np.ndarray, scale: float = 1.0, lag: int = 1
) -> np.ndarray:
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
        trunc_source_output = source_output[: i + offset][::-1]
        trunc_intervals = density_intervals[: i + offset]
        convolution_sum = (trunc_source_output * trunc_intervals).sum()
        convolved_output[i] = scale * convolution_sum

    return convolved_output


def convolve_probability(source_output, density_kernel):
    return jnp.convolve(source_output, density_kernel)[0 : len(source_output)]


def apply_convolution_for_event(
    source_output: np.ndarray, density_intervals: np.ndarray, event_proba: float
):
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


def apply_convolution_for_occupancy(
    source_output: np.ndarray, probas_stay_greater_than: np.ndarray
):
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
        notifications = apply_convolution_for_event(detected_incidence, density_intervals, 1.0)
        return notifications

    return notifications_func


def make_calc_deaths_func(death_risk, density_intervals):
    def deaths_func(detected_incidence):
        deaths = apply_convolution_for_event(detected_incidence, density_intervals, death_risk)
        return deaths

    return deaths_func


def make_calc_admissions_func(admission_risk, density_intervals):
    def admissions_func(reference_output):
        admissions = apply_convolution_for_event(
            reference_output, density_intervals, admission_risk
        )
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
