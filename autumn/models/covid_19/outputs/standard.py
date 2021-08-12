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
from autumn.tools.utils.utils import list_element_wise_division


NOTIFICATIONS = "notifications"
INCIDENCE = "incidence"
PROGRESS = "progress"


def request_standard_outputs(
    model: CompartmentalModel,
    params: Parameters,
):
    """
    Request all of the standard model outputs for the COVID-19 model, other than some specific ones used in Victoria
    """

    # Different output requests for Victoria model
    is_region_vic = params.population.region and Region.to_name(params.population.region) in Region.VICTORIA_SUBREGIONS

    # COVID-19 episode incidence - overall, disaggregated by age group and by both age group and clinical status
    model.request_output_for_flow(name=INCIDENCE, flow_name=INCIDENCE)
    notification_at_sympt_onset_sources = []

    for agegroup in AGEGROUP_STRATA:
        model.request_output_for_flow(
            name=f"{INCIDENCE}Xagegroup_{agegroup}",
            flow_name=INCIDENCE,
            dest_strata={"agegroup": agegroup},
        )

        for clinical in CLINICAL_STRATA:
            name = f"{INCIDENCE}Xagegroup_{agegroup}Xclinical_{clinical}"
            model.request_output_for_flow(
                name=name,
                flow_name=INCIDENCE,
                dest_strata={"agegroup": agegroup, "clinical": clinical},
            )
            if clinical in NOTIFICATION_CLINICAL_STRATA:
                notification_at_sympt_onset_sources.append(name)

        # We also need to capture traced cases that are not already captured with NOTIFICATION_CLINICAL_STRATA
        if params.contact_tracing:
            for clinical in [strat for strat in CLINICAL_STRATA if strat not in NOTIFICATION_CLINICAL_STRATA]:
                name = f"{INCIDENCE}_tracedXagegroup_{agegroup}Xclinical_{clinical}"
                model.request_output_for_flow(
                    name=name,
                    flow_name=INCIDENCE,
                    dest_strata={"agegroup": agegroup, "clinical": clinical, "tracing": "traced"},
                    save_results=False,
                )
                notification_at_sympt_onset_sources.append(name)

    # Within active progression
    model.request_output_for_flow(name=PROGRESS, flow_name=PROGRESS)
    for agegroup in AGEGROUP_STRATA:
        for clinical in NOTIFICATION_CLINICAL_STRATA:
            model.request_output_for_flow(
                name=f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{clinical}",
                flow_name=PROGRESS,
                dest_strata={"agegroup": agegroup, "clinical": clinical},
            )

    # New hospital and ICU admissions
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
                    dest_strata={"agegroup": agegroup, "clinical": clinical, "tracing": "traced"},
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

    # Covid-related deaths
    model.request_output_for_flow(name="infection_deaths", flow_name="infect_death")
    for agegroup in AGEGROUP_STRATA:
        model.request_output_for_flow(
            name=f"infection_deathsXagegroup_{agegroup}",
            flow_name="infect_death",
            source_strata={"agegroup": agegroup},
            save_results=False,
        )
        for clinical in CLINICAL_STRATA:
            model.request_output_for_flow(
                name=f"infection_deathsXagegroup_{agegroup}Xclinical_{clinical}",
                flow_name="infect_death",
                source_strata={"agegroup": agegroup, "clinical": clinical},
            )

    model.request_cumulative_output(name="accum_deaths", source="infection_deaths")
    if not is_region_vic:
        for agegroup in AGEGROUP_STRATA:
            model.request_cumulative_output(
                name=f"accum_deathsXagegroup_{agegroup}",
                source=f"infection_deathsXagegroup_{agegroup}",
            )

    # Track hospital occupancy - as all ICU and hospital late active compartments and some of the early active ICU cases
    compartment_periods = params.sojourn.compartment_periods
    icu_early_period = compartment_periods["icu_early"]
    hospital_early_period = compartment_periods["hospital_early"]
    period_icu_patients_in_hospital = max(icu_early_period - hospital_early_period, 0.0)
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

    # Proportion seropositive
    model.request_output_for_compartments(
        name="_total_population", compartments=COMPARTMENTS, save_results=False
    )
    if params.stratify_by_infection_history:
        model.request_output_for_compartments(
            name="_recovered",
            compartments=[Compartment.RECOVERED],
            strata={"history": "naive"},
            save_results=False,
        )
        model.request_output_for_compartments(
            name="_experienced",
            compartments=COMPARTMENTS,
            strata={"history": "experienced"},
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
        for agegroup in AGEGROUP_STRATA:
            recovered_name = f"_recoveredXagegroup_{agegroup}"
            total_name = f"_total_populationXagegroup_{agegroup}"
            model.request_output_for_compartments(
                name=total_name,
                compartments=COMPARTMENTS,
                strata={"agegroup": agegroup},
                save_results=False,
            )
            if params.stratify_by_infection_history:
                experienced_name = f"_experiencedXagegroup_{agegroup}"
                model.request_output_for_compartments(
                    name=recovered_name,
                    compartments=[Compartment.RECOVERED],
                    strata={"agegroup": agegroup, "history": "experienced"},
                    save_results=False,
                )
                model.request_output_for_compartments(
                    name=experienced_name,
                    compartments=COMPARTMENTS,
                    strata={"agegroup": agegroup, "history": "naive"},
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

    if params.vaccination and len(params.vaccination.roll_out_components) > 0:
        # FIXME: I don't think is universal yet
        age_min = params.vaccination.roll_out_components[0].age_min
        age_max = params.vaccination.roll_out_components[0].age_max
        vaccinated_agegroups = [age for age in AGEGROUP_STRATA if age_max <= float(agegroup) < age_min]
        for agegroup in vaccinated_agegroups:
            model.request_output_for_flow(
                name=f"vaccinationXagegroup_{agegroup}",
                flow_name="vaccination",
                source_strata={"agegroup": agegroup}
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

    # Track CDR
    model.request_computed_value_output("cdr")
