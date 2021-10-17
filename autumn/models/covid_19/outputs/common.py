from autumn.models.covid_19.constants import (
    INFECT_DEATH, INFECTION, Compartment, PROGRESS, NOTIFICATIONS, NOTIFICATION_CLINICAL_STRATA
)
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.models.covid_19.constants import Clinical
from autumn.models.covid_19.constants import INCIDENCE
from autumn.models.covid_19.stratifications.strains import Strain
from autumn.tools.utils.utils import list_element_wise_division

""" *** I have no idea why I can't kill the following function *** """


def cant_kill_function():
    pass


def request_stratified_output_for_flow(
        model, flow, strata, stratification, name_stem=None, filter_on="destination"
):
    """
    Standardise looping over stratum to pull out stratified outputs for flow.
    """

    stem = name_stem if name_stem else flow
    for stratum in strata:
        if filter_on == "destination":
            model.request_output_for_flow(
                name=f"{stem}X{stratification}_{stratum}",
                flow_name=flow,
                dest_strata={stratification: stratum},
            )
        elif filter_on == "source":
            model.request_output_for_flow(
                name=f"{stem}X{stratification}_{stratum}",
                flow_name=flow,
                source_strata={stratification: stratum},
            )
        else:
            raise ValueError(f"filter_on should be either 'source' or 'destination', found {filter_on}")


def request_double_stratified_output_for_flow(
        model, flow, strata_1, stratification_1, strata_2, stratification_2, name_stem=None, filter_on="destination"
):
    """
    As for previous function, but looping over two stratifications.
    """

    stem = name_stem if name_stem else flow
    for stratum_1 in strata_1:
        for stratum_2 in strata_2:
            name = f"{stem}X{stratification_1}_{stratum_1}X{stratification_2}_{stratum_2}"
            if filter_on == "destination":
                model.request_output_for_flow(
                    name=name,
                    flow_name=flow,
                    dest_strata={
                        stratification_1: stratum_1,
                        stratification_2: stratum_2,
                    }
                )
            elif filter_on == "source":
                model.request_output_for_flow(
                    name=name,
                    flow_name=flow,
                    source_strata={
                        stratification_1: stratum_1,
                        stratification_2: stratum_2,
                    }
                )
            else:
                raise ValueError(f"filter_on should be either 'source' or 'destination', found {filter_on}")


def request_stratified_output_for_compartment(
        model, request_name, compartments, strata, stratification, save_results=True
):
    for stratum in strata:
        full_request_name = f"{request_name}X{stratification}_{stratum}"
        model.request_output_for_compartments(
            name=full_request_name,
            compartments=compartments,
            strata={stratification: stratum},
            save_results=save_results,
        )


class Outputs:
    """
    This class is not specific to Covid, so should be moved out of this file - but not sure whether to move it to
    somewhere in autumn or in summer.
    """

    def __init__(self, model, COMPARTMENTS):
        self.model = model
        self.model.request_output_for_compartments(
            name="_total_population", compartments=COMPARTMENTS, save_results=False
        )

    def request_stratified_output_for_flow(self, flow, strata, stratification, name_stem=None, filter_on="destination"):
        """
        Standardise looping over stratum to pull out stratified outputs for flow.
        """

        stem = name_stem if name_stem else flow
        for stratum in strata:
            if filter_on == "destination":
                self.model.request_output_for_flow(
                    name=f"{stem}X{stratification}_{stratum}",
                    flow_name=flow,
                    dest_strata={stratification: stratum},
                )
            elif filter_on == "source":
                self.model.request_output_for_flow(
                    name=f"{stem}X{stratification}_{stratum}",
                    flow_name=flow,
                    source_strata={stratification: stratum},
                )
            else:
                raise ValueError(f"filter_on should be either 'source' or 'destination', found {filter_on}")

    def request_double_stratified_output_for_flow(
            self, flow, strata_1, stratification_1, strata_2, stratification_2, name_stem=None, filter_on="destination"
    ):
        """
        As for previous function, but looping over two stratifications.
        """

        stem = name_stem if name_stem else flow
        for stratum_1 in strata_1:
            for stratum_2 in strata_2:
                name = f"{stem}X{stratification_1}_{stratum_1}X{stratification_2}_{stratum_2}"
                if filter_on == "destination":
                    self.model.request_output_for_flow(
                        name=name,
                        flow_name=flow,
                        dest_strata={
                            stratification_1: stratum_1,
                            stratification_2: stratum_2,
                        }
                    )
                elif filter_on == "source":
                    self.model.request_output_for_flow(
                        name=name,
                        flow_name=flow,
                        source_strata={
                            stratification_1: stratum_1,
                            stratification_2: stratum_2,
                        }
                    )
                else:
                    raise ValueError(f"filter_on should be either 'source' or 'destination', found {filter_on}")


class CovidOutputs(Outputs):

    def request_incidence(self):
        """
        Request incidence rate calculations.
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
        request_double_stratified_output_for_flow(
            self.model, PROGRESS, AGEGROUP_STRATA, "agegroup", NOTIFICATION_CLINICAL_STRATA, "clinical"
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
        request_double_stratified_output_for_flow(
            self.model, INFECT_DEATH, AGEGROUP_STRATA, "agegroup",
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
        request_stratified_output_for_flow(self.model, INCIDENCE, all_strains, "strain")

        # Convert to a proportion
        for strain in all_strains:
            self.model.request_function_output(
                name=f"prop_{INCIDENCE}_strain_{strain}",
                func=lambda strain_inc, total_inc: list_element_wise_division(strain_inc, total_inc),
                sources=[f"{INCIDENCE}Xstrain_{strain}", INCIDENCE]
            )
