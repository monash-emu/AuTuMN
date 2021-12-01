from autumn.models.covid_19.constants import (
    INFECT_DEATH, INFECTION, Compartment, NOTIFICATIONS, NOTIFICATION_CLINICAL_STRATA,
    COMPARTMENTS, Vaccination, PROGRESS, Clinical, History
)
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.models.covid_19.constants import INCIDENCE
from autumn.models.covid_19.stratifications.strains import Strain
from autumn.tools.utils.utils import list_element_wise_division, get_prop_two_numerators
from autumn.tools.utils.outputsbuilder import OutputsBuilder


class CovidOutputsBuilder(OutputsBuilder):

    def request_incidence(self):

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
        Request infection rate calculations.

        Note that susceptible_infection_rate functions only work for SEIR structure, would need to change for SEIRS,
        SEIS, etc.
        """

        self.model.request_output_for_flow("infection", INFECTION)

    def request_notifications(self, contact_tracing_params, cumul_inc_start_time):

        # Unstratified
        notification_pathways = []

        # First track all traced cases (regardless of clinical stratum)
        if contact_tracing_params:
            name = "progress_traced"
            notification_pathways.append(name)
            self.model.request_output_for_flow(
                name=name,
                flow_name="progress",
                dest_strata={"tracing": "traced"},
                save_results=False,
            )

        # Then track untraced cases that are passively detected (depending on clinical stratum)
        for clinical in NOTIFICATION_CLINICAL_STRATA:
            name = f"progress_untracedX{clinical}"
            dest_strata = {"clinical": clinical, "tracing": "untraced"} if \
                contact_tracing_params else \
                {"clinical": clinical}
            notification_pathways.append(name)
            self.model.request_output_for_flow(
                name=name,
                flow_name="progress",
                dest_strata=dest_strata,
                save_results=False,
            )
        self.model.request_aggregate_output(name="notifications", sources=notification_pathways)

        # Cumulative unstratified notifications
        if cumul_inc_start_time:
            self.model.request_cumulative_output(
                name=f"accum_{NOTIFICATIONS}",
                source=NOTIFICATIONS,
                start_time=cumul_inc_start_time,
            )

        # Age-specific notifications
        for agegroup in AGEGROUP_STRATA:
            age_notification_pathways = []

            # First track all traced cases (regardless of clinical stratum)
            if contact_tracing_params:
                name = f"progress_tracedX{agegroup}"
                age_notification_pathways.append(name)
                self.model.request_output_for_flow(
                    name=name,
                    flow_name="progress",
                    dest_strata={"tracing": "traced", "agegroup": agegroup},
                    save_results=False,
                )

            # Then track untraced cases that are passively detected (depending on clinical stratum)
            for clinical in NOTIFICATION_CLINICAL_STRATA:
                name = f"progress_untracedXagegroup_{agegroup}X{clinical}"
                dest_strata = {"clinical": clinical, "tracing": "untraced", "agegroup": agegroup} if \
                    contact_tracing_params else \
                    {"clinical": clinical, "agegroup": agegroup}
                age_notification_pathways.append(name)
                self.model.request_output_for_flow(
                    name=name,
                    flow_name="progress",
                    dest_strata=dest_strata,
                    save_results=False,
                )
            self.model.request_aggregate_output(
                name=f"notificationsXagegroup_{agegroup}", sources=age_notification_pathways
            )

        # Age-specific non-hospitalised notifications
        for agegroup in AGEGROUP_STRATA:
            age_notification_pathways = []

            # First track all traced cases (in Symptomatic ambulatory ever detected)
            if contact_tracing_params:
                name = f"progress_traced_non_hospitalisedX{agegroup}"
                age_notification_pathways.append(name)
                self.model.request_output_for_flow(
                    name=name,
                    flow_name="progress",
                    dest_strata={"clinical": Clinical.SYMPT_ISOLATE, "tracing": "traced", "agegroup": agegroup},
                    save_results=False,
                )

            # Then track untraced cases that are passively detected (depending on clinical stratum)
            name = f"progress_untracedXagegroup__non_hospitalised_{agegroup}XClinical.SYMPT_ISOLATE"
            dest_strata = {"clinical": clinical, "tracing": "untraced", "agegroup": agegroup} if \
                contact_tracing_params else \
                {"clinical": Clinical.SYMPT_ISOLATE, "agegroup": agegroup}
            age_notification_pathways.append(name)
            self.model.request_output_for_flow(
                name=name,
                flow_name="progress",
                dest_strata=dest_strata,
                save_results=False,
            )
            self.model.request_aggregate_output(
                name=f"non_hospitalised_notificationsXagegroup_{agegroup}", sources=age_notification_pathways
            )

        # calculating the prevalence of the non hospitalised notifications by age group
        for agegroup in AGEGROUP_STRATA:
            age_notification_pathways = []

            # First track traced cases in all clinical strata except hospitalisations
            if contact_tracing_params:
                for clinical in NOTIFICATION_CLINICAL_STRATA:
                    name = f"progress_prevalence_traced_X{agegroup}X{clinical}"
                    age_notification_pathways.append(name)
                    self.model.request_output_for_flow(
                        name=name,
                        flow_name="progress",
                        dest_strata={"clinical": clinical, "tracing": "traced", "agegroup": agegroup},
                        save_results=False,
                    )

            # Then track untraced cases (everyone in notified clinical stratum)

            name = f"progress_prevalence_untracedXagegroup_{agegroup}XClinical.SYMPT_ISOLATE"
            dest_strata = {"clinical": Clinical.SYMPT_ISOLATE, "tracing": "untraced", "agegroup": agegroup} if \
                contact_tracing_params else \
                {"clinical": Clinical.SYMPT_ISOLATE, "agegroup": agegroup}
            age_notification_pathways.append(name)
            self.model.request_output_for_flow(
                 name=name,
                 flow_name="progress",
                 dest_strata=dest_strata,
                 save_results=False,
                )
            self.model.request_aggregate_output(
                name=f"prevalence_non_hospitalised_notificationsXagegroup_{agegroup}", sources=age_notification_pathways
            )

        # Split by child and adult
        paed_notifications = [f"notificationsXagegroup_{agegroup}" for agegroup in AGEGROUP_STRATA[:3]]
        adult_notifications = [f"notificationsXagegroup_{agegroup}" for agegroup in AGEGROUP_STRATA[3:]]
        self.model.request_aggregate_output(name="notificationsXpaediatric", sources=paed_notifications)
        self.model.request_aggregate_output(name="notificationsXadult", sources=adult_notifications)

        self.request_extra_notifications(contact_tracing_params, cumul_inc_start_time)

    def request_extra_notifications(self, contact_tracing_params, cumul_inc_start_time):
        pass

    def request_progression(self):

        # Unstratified
        self.model.request_output_for_flow(name=PROGRESS, flow_name=PROGRESS)

        # Stratified by age group and clinical status
        self.request_double_stratified_output_for_flow(
            PROGRESS, AGEGROUP_STRATA, "agegroup", NOTIFICATION_CLINICAL_STRATA, "clinical"
        )

        self.request_extra_progression()

    def request_extra_progression(self):
        pass

    def request_cdr(self):

        self.model.request_computed_value_output("cdr")

    def request_deaths(self):

        # Unstratified
        self.model.request_output_for_flow(name="infection_deaths", flow_name=INFECT_DEATH)
        self.model.request_cumulative_output(name="accum_deaths", source="infection_deaths")

        # Stratified by age
        self.request_stratified_output_for_flow(
            INFECT_DEATH, AGEGROUP_STRATA, "agegroup", name_stem="infection_deaths", filter_on="source"
        )
        for agegroup in AGEGROUP_STRATA:
            self.model.request_cumulative_output(
                name=f"accum_deathsXagegroup_{agegroup}",
                source=f"infection_deathsXagegroup_{agegroup}",
            )

        # Stratified by age and clinical stratum
        self.request_double_stratified_output_for_flow(
            INFECT_DEATH, AGEGROUP_STRATA, "agegroup",
            CLINICAL_STRATA, "clinical", name_stem="infection_deaths", filter_on="source"
        )

        self.request_extra_deaths()

    def request_extra_deaths(self):

        pass

    def request_admissions(self):

        # Track non-ICU hospital admissions (transition from early to late active in hospital, non-ICU stratum)
        self.model.request_output_for_flow(
            name="non_icu_admissions",
            flow_name="progress",
            source_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
            dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
            save_results=False,
        )

        # Track ICU admissions (transition from early to late active in ICU stratum)
        self.model.request_output_for_flow(
            name="icu_admissions",
            flow_name="progress",
            source_strata={"clinical": Clinical.ICU},
            dest_strata={"clinical": Clinical.ICU},
        )

        # Create hospitalisation functions as sum of hospital non-ICU and ICU
        self.model.request_aggregate_output(
            "hospital_admissions",
            sources=["icu_admissions", "non_icu_admissions"]
        )

        self.request_extra_admissions()

    def request_extra_admissions(self):

        pass

    def request_occupancy(self, sojourn_periods):

        # Hospital occupancy is represented as all ICU, all hospital late active, and some early active ICU cases
        compartment_periods = sojourn_periods
        icu_early_period = compartment_periods["icu_early"]
        hospital_early_period = compartment_periods["hospital_early"]
        period_icu_patients_in_hospital = max(icu_early_period - hospital_early_period, 0.)
        self.proportion_icu_patients_in_hospital = period_icu_patients_in_hospital / icu_early_period

        # Unstratified calculations
        self.model.request_output_for_compartments(
            "_late_active_hospital",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.HOSPITAL_NON_ICU},
            save_results=False,
        )
        self.model.request_output_for_compartments(
            "icu_occupancy",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.ICU},
        )
        self.model.request_output_for_compartments(
            "_early_active_icu",
            compartments=[Compartment.EARLY_ACTIVE],
            strata={"clinical": Clinical.ICU},
            save_results=False,
        )
        self.model.request_function_output(
            name="_early_active_icu_proportion",
            func=lambda patients: patients * self.proportion_icu_patients_in_hospital,
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
        # stratified by age group
        for agegroup in AGEGROUP_STRATA:

            # ## calculate age-specific ICU occupancies
            age_icu_name = f"icu_occupancyXagegroup_{agegroup}"
            self.model.request_output_for_compartments(
                name=age_icu_name,
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.ICU, "agegroup": agegroup},
                save_results=True,
            )

            # ## Calculate age-specific hospital occupancies
            # starting with hospital non-ICU
            age_late_hospital_name = f"late_hospitalXagegroup_{agegroup}"
            self.model.request_output_for_compartments(
                name=age_late_hospital_name,
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.HOSPITAL_NON_ICU, "agegroup": agegroup},
                save_results=False,
            )

            # calculating hospitalisation from early ICU compartment
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
                func=lambda patients: patients * self.proportion_icu_patients_in_hospital,
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

        self.request_extra_occupancy()

    def request_extra_occupancy(self):

        pass

    def request_tracing(self):

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

    def request_strains(self, voc_names):

        # Incidence rate for each strain implemented
        all_strains = [Strain.WILD_TYPE] + voc_names
        self.request_stratified_output_for_flow(INCIDENCE, all_strains, "strain")

        # Convert to a proportion
        for strain in all_strains:
            self.model.request_function_output(
                name=f"prop_{INCIDENCE}_strain_{strain}",
                func=lambda strain_inc, total_inc: list_element_wise_division(strain_inc, total_inc),
                sources=[f"{INCIDENCE}Xstrain_{strain}", INCIDENCE]
            )

    def request_vaccination(self, is_dosing_active, vacc_strata):

        # Track proportions vaccinated by vaccination status (depends on _total_population previously being requested)
        for vacc_stratum in vacc_strata:
            self.model.request_output_for_compartments(
                name=f"_{vacc_stratum}",
                compartments=COMPARTMENTS,
                strata={"vaccination": vacc_stratum},
                save_results=False,
            )
            self.model.request_function_output(
                name=f"proportion_{vacc_stratum}",
                sources=[f"_{vacc_stratum}", "_total_population"],
                func=lambda vaccinated, total: vaccinated / total,
            )

        if is_dosing_active:
            self.model.request_function_output(
                name="at_least_one_dose_prop",
                func=lambda vacc, one_dose, pop: (vacc + one_dose) / pop,
                sources=[f"_{Vaccination.VACCINATED}", f"_{Vaccination.ONE_DOSE_ONLY}", "_total_population"]
            )

    def request_vacc_aefis(self, vacc_risk_params):
        risk_multiplier = vacc_risk_params.risk_multiplier

        # Track the rate of adverse events and hospitalisations by age, if adverse events calculations are requested
        hospital_sources = []
        self.request_stratified_output_for_flow(
            "vaccination", AGEGROUP_STRATA, "agegroup", filter_on="source"
        )

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
            hospital_sources += [
                f"{PROGRESS}X{agegroup_string}Xclinical_{Clinical.ICU}",
                f"{PROGRESS}X{agegroup_string}Xclinical_{Clinical.HOSPITAL_NON_ICU}",
            ]

            # Hospitalisations by age
            hospital_sources_this_age = [s for s in hospital_sources if f"X{agegroup_string}X" in s]
            self.model.request_aggregate_output(
                name=f"hospital_admissionsX{agegroup_string}",
                sources=hospital_sources_this_age
            )

        # Aggregate using larger age-groups
        aggregated_age_groups = {
            "15_19": ["15"],
        }
        for age_min in [20 + i*10 for i in range(5)]:
            age_max = age_min + 9
            aggregated_age_groups[f"{age_min}_{age_max}"] = [str(age_min), str(age_min + 5)]
        aggregated_age_groups["70_plus"] = ["70", "75"]

        cumul_start_time = None if not vacc_risk_params.cumul_start_time else vacc_risk_params.cumul_start_time
        for output_type in ["tts_cases", "tts_deaths", "myocarditis_cases", "hospital_admissions"]:
            for aggregated_age_group, agegroups in aggregated_age_groups.items():
                agg_output = f"{output_type}Xagg_age_{aggregated_age_group}"
                agg_output_sources = [f"{output_type}Xagegroup_{agegroup}" for agegroup in agegroups]
                self.model.request_aggregate_output(
                    name=agg_output,
                    sources=agg_output_sources
                )

                # cumulative output calculation
                self.model.request_cumulative_output(
                    name=f"cumulative_{agg_output}",
                    source=agg_output,
                    start_time=cumul_start_time
                )

    def request_experienced(self):

        # Unstratified
        self.model.request_output_for_compartments(
            name=f"_{History.EXPERIENCED}",
            compartments=COMPARTMENTS,
            strata={"history": History.EXPERIENCED},
            save_results=False
        )
        self.model.request_output_for_compartments(
            name=f"_{History.WANED}",
            compartments=COMPARTMENTS,
            strata={"history": History.WANED},
            save_results=False
        )
        self.model.request_function_output(
            name="prop_ever_infected",
            sources=[f"_{History.EXPERIENCED}", f"_{History.WANED}", "_total_population"],
            func=get_prop_two_numerators,
        )

        self.request_stratified_output_for_compartment(
            "_total_population", COMPARTMENTS, AGEGROUP_STRATA, "agegroup", save_results=False
        )

        # Stratified by age group
        for agegroup in AGEGROUP_STRATA:
            experienced_name = f"_{History.EXPERIENCED}Xagegroup_{agegroup}"
            waned_name = f"_{History.WANED}Xagegroup_{agegroup}"
            total_name = f"_total_populationXagegroup_{agegroup}"
            self.model.request_output_for_compartments(
                name=experienced_name,
                compartments=COMPARTMENTS,
                strata={"history": History.EXPERIENCED, "agegroup": agegroup},
                save_results=False,
            )
            self.model.request_output_for_compartments(
                name=waned_name,
                compartments=COMPARTMENTS,
                strata={"history": History.WANED, "agegroup": agegroup},
                save_results=False,
            )
            self.model.request_function_output(
                name=f"prop_ever_infectedXagegroup_{agegroup}",
                sources=[experienced_name, waned_name, total_name],
                func=get_prop_two_numerators,
            )
