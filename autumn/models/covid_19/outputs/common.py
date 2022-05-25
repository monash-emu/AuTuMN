from typing import List

from summer.compute import ComputedValueProcessor

from autumn.models.covid_19.constants import (
    INFECT_DEATH, INFECTION, Compartment, NOTIFICATIONS, HISTORY_STRATA, INFECTION_DEATHS,
    COMPARTMENTS, Vaccination, PROGRESS, Clinical, History, Tracing, NOTIFICATION_CLINICAL_STRATA,
    HOSTPIALISED_CLINICAL_STRATA,
)
from autumn.models.covid_19.parameters import Sojourn, VaccinationRisk
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.models.covid_19.constants import INCIDENCE
from autumn.models.covid_19.stratifications.strains import Strain
from autumn.tools.utils.utils import get_complement_prop
from autumn.model_features.outputs import OutputsBuilder


class TimeProcess(ComputedValueProcessor):
    """
    This is just sitting here ready to go, in case anyone wants to produce any outputs that are dependent on modelled
    time.

    """

    def process(self, compartment_values, computed_values, time):
        return time


class CovidOutputsBuilder(OutputsBuilder):
    """
    The object responsible for collating and requesting all the derived outputs for the model.

    Attributes:
        is_contact_tracing: Whether contact tracing is active in this model

    """

    def __init__(self, model, compartments, is_contact_tracing):
        OutputsBuilder.__init__(self, model, compartments)
        self.is_contact_tracing = is_contact_tracing
        self.untraced_stratum = {"tracing": Tracing.UNTRACED} if is_contact_tracing else {}

    def request_incidence(self):
        """
        Incidence is the transition from late exposed to early active - i.e. the rate of onset of an "episode".

        Generates the following derived outputs both overall and by age group:
            incidence: Rate of onset of new episodes

        """

        # Unstratified
        self.model.request_output_for_flow(name=INCIDENCE, flow_name=INCIDENCE)

        # Stratified by age group
        self.request_stratified_output_for_flow(INCIDENCE, AGEGROUP_STRATA, "agegroup")

        # Stratified by age group and by clinical stratum
        self.request_double_stratified_output_for_flow(
            INCIDENCE, AGEGROUP_STRATA, "agegroup", CLINICAL_STRATA, "clinical"
        )

    def request_infection(self):
        """
        Track the rate at which people are newly infected.
        Infection is the transition from susceptible to early exposed, of course.

        Generates the following derived outputs both overall and by age group:
            infection: Rate of new infections by time

        """

        self.model.request_output_for_flow("infection", INFECTION)

    def request_notifications(self, cumul_inc_start_time: float, hospital_reporting: float):
        """
        Calculate the rate of notifications over time.

        Args:
            cumul_inc_start_time: The starting time to start the cumulative calculations
            hospital_reporting: The proportion of hospitalisations notified (defaults to one)

        Generates the following derived outputs:
            notifications overall: Progressions to active for those in the last three strata or any strata if traced
            notifications by age group: As previous, split by age

        """

        # Age-specific notifications
        for agegroup in AGEGROUP_STRATA:
            notification_pathways = []

            # First track all traced cases (regardless of clinical stratum, on the assumption that all traced patients
            # are identified as they are in quarantine)
            if self.is_contact_tracing:
                traced_dest_strata = {"agegroup": agegroup, "tracing": Tracing.TRACED}
                name = f"progress_{Tracing.TRACED}Xagegroup_{agegroup}"
                notification_pathways.append(name)
                self.model.request_output_for_flow(
                    name=name, flow_name=PROGRESS, dest_strata=traced_dest_strata, save_results=False,
                )

            # Then track untraced cases that are passively detected - either regardless of tracing status,
            # (empty dictionary if tracing not active) or just in the untraced group otherwise
            for clinical in NOTIFICATION_CLINICAL_STRATA:

                # Include reporting adjustment if requested
                reporting = hospital_reporting if clinical in HOSTPIALISED_CLINICAL_STRATA else 1.
                untraced_dest_strata = {"clinical": clinical, "agegroup": agegroup}
                untraced_dest_strata.update(self.untraced_stratum)
                notified_hospitalisations = f"_progress_{Tracing.UNTRACED}Xagegroup_{agegroup}Xclinical_{clinical}"
                self.model.request_output_for_flow(
                    name=notified_hospitalisations, flow_name=PROGRESS, dest_strata=untraced_dest_strata,
                    save_results=False,
                )

                # Has to have a different name to the flow output to avoid summer error
                name = notified_hospitalisations[1:]
                notification_pathways.append(name)
                self.model.request_function_output(
                    name=name, func=lambda rate: rate * reporting, sources=(notified_hospitalisations,),
                    save_results=False,
                )

            final_name = f"{NOTIFICATIONS}Xagegroup_{agegroup}"
            self.model.request_aggregate_output(name=final_name, sources=notification_pathways)

        notifications_by_agegroup = [f"{NOTIFICATIONS}Xagegroup_{i_age}" for i_age in AGEGROUP_STRATA]
        self.model.request_aggregate_output(name=NOTIFICATIONS, sources=notifications_by_agegroup)

        # Cumulative unstratified notifications
        if cumul_inc_start_time:
            self.model.request_cumulative_output(
                name=f"accum_{NOTIFICATIONS}", source=NOTIFICATIONS, start_time=cumul_inc_start_time,
            )

    def request_non_hosp_notifications(self):
        """
        Calculate the rate of non-hospitalised notifications over time and
        the prevalence of the non-hospitalised notifications to reflect the number of quarantined persons.

        Generates the following derived outputs:
            non hospitalised notifications: Age specific notifications in Symptomatic ambulatory ever detected
            prevalence non hospitalised notifications: Age specific outputs. For the traced everyone in notification
            clinical strata and infectious compartments. For the untraced symptomatic detected in active compartments.

        """

        # Age-specific non-hospitalised notifications
        for agegroup in AGEGROUP_STRATA:
            age_notification_pathways = []

            # First track all traced cases (in Symptomatic ambulatory ever detected)
            if self.is_contact_tracing:
                name = f"progress_traced_non_hospitalisedXagegroup_{agegroup}"
                age_notification_pathways.append(name)
                self.model.request_output_for_flow(
                    name=name,
                    flow_name=PROGRESS,
                    dest_strata={"clinical": Clinical.SYMPT_ISOLATE, "tracing": Tracing.TRACED, "agegroup": agegroup},
                    save_results=False,
                )

            # Then track untraced cases in Symptomatic ambulatory ever detected
            name = f"progress_non_hosp_Xclinical_{Clinical.SYMPT_ISOLATE}Xagegroup_{agegroup}Xtraced_{Tracing.UNTRACED}"
            dest_strata = {"clinical": Clinical.SYMPT_ISOLATE, "agegroup": agegroup}.update(self.untraced_stratum)
            age_notification_pathways.append(name)
            self.model.request_output_for_flow(
                name=name,
                flow_name=PROGRESS,
                dest_strata=dest_strata,
                save_results=False,
            )
            agg_name = f"non_hospitalised_notificationsXagegroup_{agegroup}"
            self.model.request_aggregate_output(name=agg_name, sources=age_notification_pathways)

        # calculating the prevalence of the non hospitalised notifications by age group
        for agegroup in AGEGROUP_STRATA:
            age_notification_pathways = []

            # First track traced cases in all clinical strata except hospitalisations
            if self.is_contact_tracing:
                for clinical in NOTIFICATION_CLINICAL_STRATA:
                    name = f"progress_prevalence_traced_X{agegroup}X{clinical}"
                    age_notification_pathways.append(name)
                    self.model.request_output_for_compartments(
                        name=name,
                        compartments=[Compartment.LATE_ACTIVE],
                        strata={"clinical": clinical, "agegroup": agegroup, "tracing": Tracing.TRACED},
                    )

            # Then track untraced cases (everyone in notified clinical stratum)
            dest_strata = {"clinical": Clinical.SYMPT_ISOLATE, "agegroup": agegroup}.update(self.untraced_stratum)
            name = f"progress_prevalence_{Tracing.UNTRACED}Xagegroup_{agegroup}Xclinical_{Clinical.SYMPT_ISOLATE}"
            age_notification_pathways.append(name)
            self.model.request_output_for_compartments(
                name=name,
                compartments=[Compartment.LATE_ACTIVE],
                strata=dest_strata,
            )

            self.model.request_aggregate_output(
                name=f"prevalence_non_hospitalised_notificationsXagegroup_{agegroup}", sources=age_notification_pathways
            )

    def request_adult_paeds_notifications(self):
        """
        Split the age-specific notifications previously generated into paediatric and adult.

        Generates the following derived outputs:
            notificationsXpaediatric: Notifications for those aged under 15
            notificationsXadult: Notifications for those aged 15 and above

        """

        # Split by child and adult
        paed_notifications = [f"notificationsXagegroup_{agegroup}" for agegroup in AGEGROUP_STRATA[:3]]
        adult_notifications = [f"notificationsXagegroup_{agegroup}" for agegroup in AGEGROUP_STRATA[3:]]
        self.model.request_aggregate_output(name="notificationsXpaediatric", sources=paed_notifications)
        self.model.request_aggregate_output(name="notificationsXadult", sources=adult_notifications)

    def request_cdr(self):
        """
        Just make the computed value CDR (case detection rate) available as a derived output.

        """

        self.model.request_computed_value_output("cdr")

    def request_deaths(self):
        """
        Track COVID-19-related deaths.

        Generates the following derived outputs both overall and by age group:
            infection_deaths: Rate of deaths over time
            accum_deaths: Cumulative deaths that have accrued by that point in time

        """

        # Unstratified
        self.model.request_output_for_flow(name=INFECTION_DEATHS, flow_name=INFECT_DEATH)
        self.model.request_cumulative_output(name="accum_deaths", source="infection_deaths")

        # Stratified by age
        self.request_stratified_output_for_flow(
            INFECT_DEATH, AGEGROUP_STRATA, "agegroup", name_stem=INFECTION_DEATHS, filter_on="source"
        )
        for agegroup in AGEGROUP_STRATA:
            self.model.request_cumulative_output(
                name=f"accum_deathsXagegroup_{agegroup}",
                source=f"infection_deathsXagegroup_{agegroup}",
            )

        # Stratified by age and clinical stratum
        self.request_double_stratified_output_for_flow(
            INFECT_DEATH, AGEGROUP_STRATA, "agegroup", CLINICAL_STRATA, "clinical", name_stem=INFECTION_DEATHS,
            filter_on="source"
        )

    def request_admissions(self):
        """
        Track COVID-19-attributable admissions to hospital and to ICU.

        Generates the following derived outputs both overall and by age group:
            icu_admissions: Only those being admitted to ICU
            hospital_admissions: All those being admitted to hospital

        """

        # Track non-ICU hospital admissions (transition from early to late active in hospital, non-ICU stratum)
        self.model.request_output_for_flow(
            name="non_icu_admissions",
            flow_name=PROGRESS,
            source_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
            dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
            save_results=False,
        )

        # Track ICU admissions (transition from early to late active in ICU stratum)
        self.model.request_output_for_flow(
            name="icu_admissions",
            flow_name=PROGRESS,
            source_strata={"clinical": Clinical.ICU},
            dest_strata={"clinical": Clinical.ICU},
        )

        # Track all hospitalisations as the sum of hospital non-ICU and ICU
        self.model.request_aggregate_output("hospital_admissions", sources=["icu_admissions", "non_icu_admissions"])

        for agegroup in AGEGROUP_STRATA:
            self.model.request_output_for_flow(
                name=f"non_icu_admissionsXagegroup_{agegroup}",
                flow_name=PROGRESS,
                source_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "agegroup": agegroup},
                dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "agegroup": agegroup},
                save_results=False,
            )
            self.model.request_output_for_flow(
                name=f"icu_admissionsXagegroup_{agegroup}",
                flow_name=PROGRESS,
                source_strata={"clinical": Clinical.ICU, "agegroup": agegroup},
                dest_strata={"clinical": Clinical.ICU, "agegroup": agegroup},
                save_results=False,
            )
            self.model.request_aggregate_output(
                f"hospital_admissionsXagegroup_{agegroup}",
                sources=[f"icu_admissionsXagegroup_{agegroup}", f"non_icu_admissionsXagegroup_{agegroup}"]
            )

    def request_occupancy(self, sojourn_periods: Sojourn):
        """
        Track the number of people in hospital or in ICU over time.

        Args:
            sojourn_periods: The sojourn periods for the hospitalised compartments

        Generates the following derived outputs both overall and by age group:
            icu_occupancy: Only those currently in ICU
            hospital_occupancy: All those currently in hospital

        """

        # Hospital occupancy is represented as all ICU, all hospital late active, and some early active ICU cases
        compartment_periods = sojourn_periods
        icu_early_period = compartment_periods["icu_early"]
        hospital_early_period = compartment_periods["hospital_early"]
        period_icu_patients_in_hospital = max(icu_early_period - hospital_early_period, 0.)
        proportion_icu_patients_in_hospital = period_icu_patients_in_hospital / icu_early_period

        # Unstratified calculations
        self.model.request_output_for_compartments(
            "icu_occupancy",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.ICU},
        )
        self.model.request_output_for_compartments(
            "_late_active_hospital",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.HOSPITAL_NON_ICU},
            save_results=False,
        )
        self.model.request_output_for_compartments(
            "_early_active_icu",
            compartments=[Compartment.EARLY_ACTIVE],
            strata={"clinical": Clinical.ICU},
            save_results=False,
        )
        self.model.request_function_output(
            name="_early_active_icu_proportion",
            func=lambda patients: patients * proportion_icu_patients_in_hospital,
            sources=["_early_active_icu"],
            save_results=False,
        )
        self.model.request_aggregate_output(
            name="hospital_occupancy",
            sources=[
                "_late_active_hospital",
                "icu_occupancy",
                "_early_active_icu_proportion",
            ],
        )

        # Stratified by age group
        for agegroup in AGEGROUP_STRATA:

            age_icu_name = f"icu_occupancyXagegroup_{agegroup}"
            self.model.request_output_for_compartments(
                name=age_icu_name,
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.ICU, "agegroup": agegroup},
                save_results=True,
            )
            age_late_hospital_name = f"late_hospitalXagegroup_{agegroup}"
            self.model.request_output_for_compartments(
                name=age_late_hospital_name,
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.HOSPITAL_NON_ICU, "agegroup": agegroup},
                save_results=False,
            )
            age_icu_ealy_name = f"early_active_icuXagegroup_{agegroup}"
            self.model.request_output_for_compartments(
                name=age_icu_ealy_name,
                compartments=[Compartment.EARLY_ACTIVE],
                strata={"clinical": Clinical.ICU, "agegroup": agegroup},
                save_results=False,
            )
            age_icu_early_in_hospital_name = f"early_active_icu_in_hospitalXagegroup_{agegroup}"
            self.model.request_function_output(
                name=age_icu_early_in_hospital_name,
                func=lambda patients: patients * proportion_icu_patients_in_hospital,
                sources=[age_icu_ealy_name],
                save_results=False,
            )
            self.model.request_aggregate_output(
                name=f"hospital_occupancyXagegroup_{agegroup}",
                sources=[
                    age_icu_name,
                    age_late_hospital_name,
                    age_icu_early_in_hospital_name,
                ],
            )

    def request_tracing(self):
        """
        Collate up all the computed values used during the process of working out the effect of contact tracing.

        Additionally generates the following derived outputs:
            prop_contacts_quarantined: The proportion of all contacts (including those of undetected cases) identified

        """

        # Standard calculations always computed when contact tracing requested
        self.model.request_computed_value_output("prevalence")
        self.model.request_computed_value_output("prop_detected_traced")
        self.model.request_computed_value_output("prop_contacts_with_detected_index")
        self.model.request_computed_value_output("traced_flow_rate")

        # Proportion of quarantined contacts among all contacts
        self.model.request_function_output(
            name="prop_contacts_quarantined",
            func=lambda prop_detected_traced, prop_detected_index: prop_detected_traced * prop_detected_index,
            sources=["prop_detected_traced", "prop_contacts_with_detected_index"],
        )

    def request_strains(self, voc_names: List[str]):
        """
        VoC-related outputs.

        Args:
            voc_names: The names of all the VoCs being implemented in the model

        Generates the following derived outputs:
            incidence (by strain): See above for definition
            prop_incidence (by strain): Proportion of incidence attributable to that strain

        """

        # Incidence rate for each strain implemented
        all_strains = [Strain.WILD_TYPE] + voc_names
        self.request_stratified_output_for_flow(INCIDENCE, all_strains, "strain")

        # Convert to a proportion
        for strain in all_strains:
            self.model.request_function_output(
                name=f"prop_{INCIDENCE}_strain_{strain}",
                func=lambda strain_inc, total_inc: strain_inc / total_inc,
                sources=[f"{INCIDENCE}Xstrain_{strain}", INCIDENCE]
            )

    def request_vaccination(self, is_dosing_active: bool, vacc_strata: List[str]):
        """
        Vaccination-related outputs

        Args:
            is_dosing_active: Are we including an additional model stratum for those who have been fully vaccinated
            vacc_strata: All the vaccination strata being implemented

        Generates the following derived outputs:
            proportions of people (by vaccination status): See the population distribution by vaccination status
            at_least_one_dose_prop: To facilitate calculating total vaccination coverage, regardless of model structure

        """

        # Track proportions vaccinated by vaccination status (depends on total_population previously being requested)
        for vacc_stratum in vacc_strata:
            self.model.request_output_for_compartments(
                name=f"_{vacc_stratum}",
                compartments=COMPARTMENTS,
                strata={"vaccination": vacc_stratum},
                save_results=False,
            )
            self.model.request_function_output(
                name=f"proportion_{vacc_stratum}",
                sources=[f"_{vacc_stratum}", "total_population"],
                func=lambda vaccinated, total: vaccinated / total,
            )

        if is_dosing_active:
            self.model.request_function_output(
                name="at_least_one_dose_prop",
                func=lambda unvacc, pop: 1. - unvacc / pop,
                sources=[f"_{Vaccination.UNVACCINATED}", "total_population"]
            )

    def request_vacc_aefis(self, vacc_risk_params: VaccinationRisk):
        """
        Calculate the risk of TTS and myocarditis for the vaccines predominantly responsible for these.

        Args:
            vacc_risk_params: The parameters defining vaccine-related risk (i.e. AEFIs)

        """

        risk_multiplier = vacc_risk_params.risk_multiplier

        # Track the rate of adverse events and hospitalisations by age, if adverse events calculations are requested
        self.request_stratified_output_for_flow("vaccination", AGEGROUP_STRATA, "agegroup", filter_on="source")

        for agegroup in AGEGROUP_STRATA:
            agegroup_string = f"agegroup_{agegroup}"

            # TTS for AstraZeneca vaccines
            self.model.request_function_output(
                name=f"tts_casesX{agegroup_string}",
                sources=[f"vaccinationX{agegroup_string}"],
                func=lambda vaccinated:
                vaccinated * vacc_risk_params.tts_rate[agegroup] * vacc_risk_params.prop_astrazeneca * risk_multiplier
            )
            self.model.request_function_output(
                name=f"tts_deathsX{agegroup_string}",
                sources=[f"tts_casesX{agegroup_string}"],
                func=lambda tts_cases:
                tts_cases * vacc_risk_params.tts_fatality_ratio[agegroup]
            )

            # Myocarditis for mRNA vaccines
            self.model.request_function_output(
                name=f"myocarditis_casesX{agegroup_string}",
                sources=[f"vaccinationX{agegroup_string}"],
                func=lambda vaccinated:
                vaccinated * vacc_risk_params.myocarditis_rate[agegroup] * vacc_risk_params.prop_mrna * risk_multiplier
            )

        # Aggregate using larger age-groups
        aggregated_age_groups = {"15_19": ["15"]}
        for age_min in [20 + i * 10 for i in range(5)]:
            age_max = age_min + 9
            aggregated_age_groups[f"{age_min}_{age_max}"] = [str(age_min), str(age_min + 5)]
        aggregated_age_groups["70_plus"] = ["70", "75"]

        cumul_start_time = None if not vacc_risk_params.cumul_start_time else vacc_risk_params.cumul_start_time
        for output_type in ["tts_cases", "tts_deaths", "myocarditis_cases", "hospital_admissions"]:
            for aggregated_age_group, agegroups in aggregated_age_groups.items():
                agg_output = f"{output_type}Xagg_age_{aggregated_age_group}"
                agg_output_sources = [f"{output_type}Xagegroup_{agegroup}" for agegroup in agegroups]
                self.model.request_aggregate_output(name=agg_output, sources=agg_output_sources)

                # Cumulative output calculation
                self.model.request_cumulative_output(
                    name=f"cumulative_{agg_output}",
                    source=agg_output,
                    start_time=cumul_start_time
                )

    def request_experienced(self):
        """
        History status-related outputs.
        Arguably the naive group should exclude the people with active infection - this is not currently being done.

        Generates the following derived outputs:
            proportions of people (by history status)
            prop_ever_infected: Everyone who is not infection-naive

        """

        self.request_stratified_output_for_compartment("prop", COMPARTMENTS, HISTORY_STRATA, "history")
        self.model.request_function_output(
            name="prop_ever_infected",
            func=get_complement_prop,
            sources=[f"propXhistory_{History.NAIVE}", "total_population"],
        )

        # Stratified by age group
        self.request_stratified_output_for_compartment(
            "total_population", COMPARTMENTS, AGEGROUP_STRATA, "agegroup", save_results=False
        )
        for agegroup in AGEGROUP_STRATA:
            for stratum in HISTORY_STRATA:
                name = f"prop_{stratum}Xagegroup{agegroup}"
                self.model.request_output_for_compartments(
                    name=name,
                    compartments=COMPARTMENTS,
                    strata={"history": stratum, "agegroup": agegroup},
                )
            self.model.request_function_output(
                name=f"prop_ever_infectedXagegroup_{agegroup}",
                func=get_complement_prop,
                sources=[f"prop_{History.NAIVE}Xagegroup{agegroup}", f"total_populationXagegroup_{agegroup}"],
            )

    def request_time(self):
        """
        Currently not used, but available if needed.

        """

        self.model.request_computed_value_output("time_process")
