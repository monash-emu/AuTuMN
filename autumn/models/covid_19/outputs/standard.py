from summer import CompartmentalModel

from autumn.models.covid_19.constants import (
    COMPARTMENTS,
    NOTIFICATION_CLINICAL_STRATA,
    Clinical,
    Compartment,
)
from autumn.projects.covid_19.mixing_optimisation.constants import OPTI_ISO3S, Region
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.tools import inputs

from autumn.tools.utils.utils import list_element_wise_division


def request_standard_outputs(
    model: CompartmentalModel,
    params: Parameters,
):
    country = params.country
    pop = params.population
    is_region_vic = pop.region and Region.to_name(pop.region) in Region.VICTORIA_SUBREGIONS

    # Disease incidence
    model.request_output_for_flow(name="incidence", flow_name="incidence")
    notification_at_sympt_onset_sources = []
    for agegroup in AGEGROUP_STRATA:
        # Track incidence for each agegroup.
        model.request_output_for_flow(
            name=f"incidenceXagegroup_{agegroup}",
            flow_name="incidence",
            dest_strata={"agegroup": agegroup},
        )
        for clinical in CLINICAL_STRATA:
            # Track incidence for each agegroup and clinical status.
            name = f"incidenceXagegroup_{agegroup}Xclinical_{clinical}"
            model.request_output_for_flow(
                name=name,
                flow_name="incidence",
                dest_strata={"agegroup": agegroup, "clinical": clinical},
            )
            if clinical in NOTIFICATION_CLINICAL_STRATA:
                notification_at_sympt_onset_sources.append(name)

        # We also need to capture traced cases that are not already captured with NOTIFICATION_CLINICAL_STRATA
        if params.contact_tracing:
            for clinical in [s for s in CLINICAL_STRATA if s not in NOTIFICATION_CLINICAL_STRATA]:
                name = f"incidence_tracedXagegroup_{agegroup}Xclinical_{clinical}"
                model.request_output_for_flow(
                    name=name,
                    flow_name="incidence",
                    dest_strata={"agegroup": agegroup, "clinical": clinical, "tracing": "traced"},
                    save_results=False,
                )
                notification_at_sympt_onset_sources.append(name)

    # Notifications at symptom onset.
    model.request_aggregate_output(
        name="notifications_at_sympt_onset", sources=notification_at_sympt_onset_sources
    )

    # Disease progression
    model.request_output_for_flow(name="progress", flow_name="progress")
    for agegroup in AGEGROUP_STRATA:
        for clinical in NOTIFICATION_CLINICAL_STRATA:
            model.request_output_for_flow(
                name=f"progressXagegroup_{agegroup}Xclinical_{clinical}",
                flow_name="progress",
                dest_strata={"agegroup": agegroup, "clinical": clinical},
            )

    # New hospital admissions
    hospital_sources = []
    icu_sources = []
    for agegroup in AGEGROUP_STRATA:
        icu_sources.append(f"progressXagegroup_{agegroup}Xclinical_{Clinical.ICU}")
        hospital_sources += [
            f"progressXagegroup_{agegroup}Xclinical_{Clinical.ICU}",
            f"progressXagegroup_{agegroup}Xclinical_{Clinical.HOSPITAL_NON_ICU}",
        ]

    model.request_aggregate_output(
        name="new_hospital_admissions",
        sources=hospital_sources,
    )
    model.request_aggregate_output(name="new_icu_admissions", sources=icu_sources)

    # Get notifications, which may included people detected in-country as they progress, or imported cases which are detected.
    notification_sources = [
        f"progressXagegroup_{a}Xclinical_{c}" for a in AGEGROUP_STRATA for c in NOTIFICATION_CLINICAL_STRATA
    ]

    # We also need to capture traced cases that are not already captured with NOTIFICATION_CLINICAL_STRATA
    notifications_traced_by_age_sources = {}
    if params.contact_tracing:
        for agegroup in AGEGROUP_STRATA:
            notifications_traced_by_age_sources[agegroup] = []
            for clinical in [s for s in CLINICAL_STRATA if s not in NOTIFICATION_CLINICAL_STRATA]:
                name = f"progress_tracedXagegroup_{agegroup}Xclinical_{clinical}"
                model.request_output_for_flow(
                    name=name,
                    flow_name="progress",
                    dest_strata={"agegroup": agegroup, "clinical": clinical, "tracing": "traced"},
                    save_results=False,
                )
                notification_sources.append(name)
                notifications_traced_by_age_sources[agegroup].append(name)

    model.request_aggregate_output(name="local_notifications", sources=notification_sources)
    model.request_aggregate_output(
        name="notifications", sources=notification_sources
    )  # Used to be different coz we had imports.

    # cumulative incidence and notifications
    if params.cumul_incidence_start_time:
        for existing_output in ["incidence", "notifications"]:
            model.request_cumulative_output(
                name=f"accum_{existing_output}",
                source=existing_output,
                start_time=params.cumul_incidence_start_time,
            )

    # Notification by age group
    for agegroup in AGEGROUP_STRATA:
        sympt_isolate_name = f"progressXagegroup_{agegroup}Xclinical_sympt_isolate"
        hospital_non_icu_name = f"progressXagegroup_{agegroup}Xclinical_hospital_non_icu"
        icu_name = f"progressXagegroup_{agegroup}Xclinical_icu"
        notifications_by_age_sources = [sympt_isolate_name, hospital_non_icu_name, icu_name]
        if params.contact_tracing:
            notifications_by_age_sources += notifications_traced_by_age_sources[agegroup]

        model.request_aggregate_output(
            name=f"notificationsXagegroup_{agegroup}",
            sources=notifications_by_age_sources
        )

    # Infection deaths.
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

    # Track years of life lost per year.
    life_expectancy = inputs.get_life_expectancy_by_agegroup(AGEGROUP_STRATA, country.iso3)[0]
    life_expectancy_latest = [life_expectancy[agegroup][-1] for agegroup in life_expectancy]
    yoll_sources = []
    for idx, agegroup in enumerate(AGEGROUP_STRATA):
        # Use default parameter to bind loop variable to function.
        l = life_expectancy_latest[idx]

        def get_yoll(deaths, life_exp=l):
            return deaths * life_exp

        yoll_source = f"_yoll_{agegroup}"
        yoll_sources.append(yoll_source)
        model.request_function_output(
            name=yoll_source,
            func=get_yoll,
            sources=[f"infection_deathsXagegroup_{agegroup}"],
            save_results=False,
        )

    model.request_aggregate_output(name="years_of_life_lost", sources=yoll_sources)
    if params.country.iso3 in OPTI_ISO3S:
        # Derived outputs for the optimization project.
        model.request_cumulative_output(
            name="accum_years_of_life_lost", source="years_of_life_lost"
        )

    # Track hospital occupancy.
    # We count all ICU and hospital late active compartments and a proportion of early active ICU cases.
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
        for agegroup in AGEGROUP_STRATA:
            model.request_output_for_flow(
                name=f"vaccinationXagegroup_{agegroup}",
                flow_name="vaccination",
                source_strata={"agegroup": agegroup}
                )

    if params.voc_emergence:
        # Calculate the incidence of VoC cases
        model.request_output_for_flow(
            name=f"incidence_voc",
            flow_name="incidence",
            dest_strata={"strain": "voc"},
            save_results=False
        )
        # Calculate the proportion of incident cases that are VoC
        model.request_function_output(
            name="prop_voc_incidence",
            func=lambda voc, total: list_element_wise_division(voc, total),
            sources=["incidence_voc", "incidence"]
        )

    # track CDR
    model.request_computed_value_output("cdr")
