from summer import CompartmentalModel

from autumn.models.covid_19.constants import (
    COMPARTMENTS,
    NOTIFICATION_CLINICAL_STRATA,
    Clinical,
    Compartment,
    Strain,
)
from autumn.projects.covid_19.mixing_optimisation.constants import Region
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.models.covid_19.stratifications.history import History
from autumn.models.covid_19.stratifications.tracing import Tracing
from autumn.models.covid_19.stratifications.vaccination import VACCINATION_STRATA
from autumn.tools.utils.utils import list_element_wise_division


NOTIFICATIONS = "notifications"
INFECTION = "infection"
INCIDENCE = "incidence"
PROGRESS = "progress"
INFECT_DEATH = "infect_death"


def find_vaccinated_agegroups(roll_out_components):
    """
    Find all the age groups that are getting vaccinated under any of the roll-out components.
    """

    relevant_agegroups = set()
    for component in range(len(roll_out_components)):
        age_min = roll_out_components[component].age_min
        age_max = roll_out_components[component].age_max
        min_value = age_min if age_min else 0.
        max_value = age_max if age_max else 200.
        vaccinated_agegroups = [age for age in AGEGROUP_STRATA if min_value <= float(age) < max_value]
        relevant_agegroups = set.union(relevant_agegroups, set(vaccinated_agegroups))
    return relevant_agegroups


def request_stratified_output_for_flow(
        model, flow, strata, stratification, name_stem=None
):
    """
    Standardise looping over stratum to pull out stratified outputs for flow.
    """

    stem = name_stem if name_stem else flow
    for stratum in strata:
        model.request_output_for_flow(
            name=f"{stem}X{stratification}_{stratum}",
            flow_name=flow,
            dest_strata={stratification: stratum},
        )


def request_double_stratified_output_for_flow(
        model, flow, strata_1, stratification_1, strata_2, stratification_2, name_stem=None
):
    """
    As for previous function, but looping over two stratifications.
    """

    stem = name_stem if name_stem else flow
    for stratum_1 in strata_1:
        for stratum_2 in strata_2:
            name = f"{stem}X{stratification_1}_{stratum_1}X{stratification_2}_{stratum_2}"
            model.request_output_for_flow(
                name=name,
                flow_name=flow,
                dest_strata={
                    stratification_1: stratum_1,
                    stratification_2: stratum_2,
                }
            )


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


def request_standard_outputs(
    model: CompartmentalModel,
    params: Parameters,
):
    """
    Request all of the standard model outputs for the COVID-19 model, other than some specific ones used in Victoria.
    """

    # Different output requests for Victoria model
    is_region_vic = params.population.region and Region.to_name(params.population.region) in Region.VICTORIA_SUBREGIONS

    """
    Infection
    """

    model.request_output_for_flow(INFECTION, INFECTION)
    if params.vaccination:
        request_stratified_output_for_flow(model, INFECTION, VACCINATION_STRATA, "vaccination")

    """
    Incidence
    """

    # Overall, disaggregated by age group and disaggregated by both age group and clinical status
    model.request_output_for_flow(name=INCIDENCE, flow_name=INCIDENCE)
    request_stratified_output_for_flow(model, INCIDENCE, AGEGROUP_STRATA, "agegroup")
    request_double_stratified_output_for_flow(
        model, INCIDENCE, AGEGROUP_STRATA, "agegroup", CLINICAL_STRATA, "clinical"
    )

    # We also need to capture traced cases that are not already captured with NOTIFICATION_CLINICAL_STRATA
    if params.contact_tracing:
        non_notified_clinical_strata = [strat for strat in CLINICAL_STRATA if strat not in NOTIFICATION_CLINICAL_STRATA]
        for agegroup in AGEGROUP_STRATA:
            for clinical in non_notified_clinical_strata:
                name = f"{INCIDENCE}_tracedXagegroup_{agegroup}Xclinical_{clinical}"
                model.request_output_for_flow(
                    name=name,
                    flow_name=INCIDENCE,
                    dest_strata={"agegroup": agegroup, "clinical": clinical, "tracing": Tracing.TRACED},
                    save_results=False,
                )

    """
    Progression
    """

    model.request_output_for_flow(name=PROGRESS, flow_name=PROGRESS)
    request_double_stratified_output_for_flow(
        model, PROGRESS, AGEGROUP_STRATA, "agegroup", NOTIFICATION_CLINICAL_STRATA, "clinical"
    )

    """
    Healthcare admissions
    """

    hospital_sources, icu_sources = [], []
    for agegroup in AGEGROUP_STRATA:
        hospital_sources += [
            f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.ICU}",
            f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.HOSPITAL_NON_ICU}",
        ]
        icu_sources.append(f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.ICU}")
    model.request_aggregate_output(name="new_hospital_admissions", sources=hospital_sources)
    model.request_aggregate_output(name="new_icu_admissions", sources=icu_sources)

    notification_sources = [
        f"{PROGRESS}Xagegroup_{age}Xclinical_{clinical}"
        for age in AGEGROUP_STRATA
        for clinical in NOTIFICATION_CLINICAL_STRATA
    ]

    # We also need to capture traced cases that are not already captured with NOTIFICATION_CLINICAL_STRATA
    notifications_traced_by_age_sources = {}
    if params.contact_tracing:
        for agegroup in AGEGROUP_STRATA:
            notifications_traced_by_age_sources[agegroup] = []
            for clinical in [s for s in CLINICAL_STRATA if s not in NOTIFICATION_CLINICAL_STRATA]:
                name = f"{PROGRESS}_tracedXagegroup_{agegroup}Xclinical_{clinical}"
                model.request_output_for_flow(
                    name=name,
                    flow_name=PROGRESS,
                    dest_strata={"agegroup": agegroup, "clinical": clinical, "tracing": Tracing.TRACED},
                    save_results=False,
                )
                notification_sources.append(name)
                notifications_traced_by_age_sources[agegroup].append(name)

    model.request_aggregate_output(name=NOTIFICATIONS, sources=notification_sources)

    # Cumulative incidence and notifications
    if params.cumul_incidence_start_time:
        for existing_output in [INCIDENCE, NOTIFICATIONS]:
            model.request_cumulative_output(
                name=f"accum_{existing_output}",
                source=existing_output,
                start_time=params.cumul_incidence_start_time,
            )

    # Notifications by age group
    for agegroup in AGEGROUP_STRATA:
        sympt_isolate_name = f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.SYMPT_ISOLATE}"
        hospital_non_icu_name = f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.HOSPITAL_NON_ICU}"
        icu_name = f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.ICU}"
        notifications_by_age_sources = [sympt_isolate_name, hospital_non_icu_name, icu_name]
        if params.contact_tracing:
            notifications_by_age_sources += notifications_traced_by_age_sources[agegroup]

        model.request_aggregate_output(
            name=f"{NOTIFICATIONS}Xagegroup_{agegroup}",
            sources=notifications_by_age_sources
        )

    """
    Deaths
    """

    model.request_output_for_flow(
        name="infection_deaths", flow_name=INFECT_DEATH
    )
    request_stratified_output_for_flow(
        model, INFECT_DEATH, AGEGROUP_STRATA, "agegroup", name_stem="infection_deaths"
    )
    request_double_stratified_output_for_flow(
        model, INFECT_DEATH, AGEGROUP_STRATA, "agegroup", CLINICAL_STRATA, "clinical", name_stem="infection_deaths"
    )
    model.request_cumulative_output(name="accum_deaths", source="infection_deaths")
    if not is_region_vic:
        for agegroup in AGEGROUP_STRATA:
            model.request_cumulative_output(
                name=f"accum_deathsXagegroup_{agegroup}",
                source=f"infection_deathsXagegroup_{agegroup}",
            )

    """
    Healthcare occupancy
    """

    # Hospital occupancy represented as all ICU, all hospital late active, and some early active ICU cases
    compartment_periods = params.sojourn.compartment_periods
    icu_early_period = compartment_periods["icu_early"]
    hospital_early_period = compartment_periods["hospital_early"]
    period_icu_patients_in_hospital = max(icu_early_period - hospital_early_period, 0.)
    proportion_icu_patients_in_hospital = period_icu_patients_in_hospital / icu_early_period
    model.request_output_for_compartments(
        "_late_active_hospital",
        compartments=[Compartment.LATE_ACTIVE],
        strata={"clinical": Clinical.HOSPITAL_NON_ICU},
        save_results=False,
    )
    model.request_output_for_compartments(
        "icu_occupancy",
        compartments=[Compartment.LATE_ACTIVE],
        strata={"clinical": Clinical.ICU},
    )
    model.request_output_for_compartments(
        "_early_active_icu",
        compartments=[Compartment.EARLY_ACTIVE],
        strata={"clinical": Clinical.ICU},
        save_results=False,
    )
    model.request_function_output(
        name="_early_active_icu_proportion",
        func=lambda patients: patients * proportion_icu_patients_in_hospital,
        sources=["_early_active_icu"],
        save_results=False,
    )
    model.request_aggregate_output(
        name="hospital_occupancy",
        sources=[
            "_late_active_hospital",
            "icu_occupancy",
            "_early_active_icu_proportion",
        ],
    )

    """
    Proportion seropositive/recovered
    """

    model.request_output_for_compartments(
        name="_total_population", compartments=COMPARTMENTS, save_results=False
    )
    if params.stratify_by_infection_history:

        # Note these people are called "naive", but they have actually had past Covid, immunity just hasn't yet waned
        model.request_output_for_compartments(
            name="_recovered",
            compartments=[Compartment.RECOVERED],
            strata={"history": History.NAIVE},
            save_results=False,
        )
        model.request_output_for_compartments(
            name="_experienced",
            compartments=COMPARTMENTS,
            strata={"history": History.EXPERIENCED},
            save_results=False,
        )
        model.request_function_output(
            name="proportion_seropositive",
            sources=["_recovered", "_experienced", "_total_population"],
            func=lambda recovered, experienced, total: (recovered + experienced) / total,
        )
    else:
        model.request_output_for_compartments(
            name="_recovered", compartments=[Compartment.RECOVERED], save_results=False
        )
        model.request_function_output(
            name="proportion_seropositive",
            sources=["_recovered", "_total_population"],
            func=lambda recovered, total: recovered / total,
        )
    if not is_region_vic:
        request_stratified_output_for_compartment(
            model, "_total_population", COMPARTMENTS, AGEGROUP_STRATA, "agegroup", save_results=False
        )
        for agegroup in AGEGROUP_STRATA:
            recovered_name = f"_recoveredXagegroup_{agegroup}"
            total_name = f"_total_populationXagegroup_{agegroup}"
            if params.stratify_by_infection_history:
                experienced_name = f"_experiencedXagegroup_{agegroup}"
                model.request_output_for_compartments(
                    name=recovered_name,
                    compartments=[Compartment.RECOVERED],
                    strata={"agegroup": agegroup, "history": History.EXPERIENCED},
                    save_results=False,
                )
                model.request_output_for_compartments(
                    name=experienced_name,
                    compartments=COMPARTMENTS,
                    strata={"agegroup": agegroup, "history": History.NAIVE},
                    save_results=False,
                )
                model.request_function_output(
                    name=f"proportion_seropositiveXagegroup_{agegroup}",
                    sources=[recovered_name, experienced_name, total_name],
                    func=lambda recovered, experienced, total: (recovered + experienced) / total,
                )
            else:
                model.request_output_for_compartments(
                    name=recovered_name,
                    compartments=[Compartment.RECOVERED],
                    strata={"agegroup": agegroup},
                    save_results=False,
                )
                model.request_function_output(
                    name=f"proportion_seropositiveXagegroup_{agegroup}",
                    sources=[recovered_name, total_name],
                    func=lambda recovered, total: recovered / total,
                )

    """
    Vaccination
    """

    if params.vaccination and len(params.vaccination.roll_out_components) > 0:
        request_stratified_output_for_flow(
            model, "vaccination", find_vaccinated_agegroups(params.vaccination.roll_out_components), "agegroup"
        )

    # Calculate the incidence by strain
    if params.voc_emergence:
        voc_names = list(params.voc_emergence.keys())
        all_strains = [Strain.WILD_TYPE] + voc_names
        for strain in all_strains:
            incidence_key = f"{INCIDENCE}_strain_{strain}"
            model.request_output_for_flow(
                name=incidence_key,
                flow_name=INCIDENCE,
                dest_strata={"strain": strain},
                save_results=False
            )
            # Calculate the proportion of incident cases that are VoC
            model.request_function_output(
                name=f"prop_{INCIDENCE}_strain_{strain}",
                func=lambda strain_inc, total_inc: list_element_wise_division(strain_inc, total_inc),
                sources=[incidence_key, INCIDENCE]
            )

    """
    Case detection rate
    """

    model.request_computed_value_output("cdr")
