from typing import List
import numpy as np

from summer2 import CompartmentalModel

from autumn import inputs

from apps.covid_19.model.parameters import Parameters
from apps.covid_19.constants import Compartment, Clinical, COMPARTMENTS, NOTIFICATION_STRATA
from apps.covid_19.mixing_optimisation.constants import OPTI_ISO3S, Region
from apps.covid_19.model.stratifications.clinical import CLINICAL_STRATA
from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA

from apps.covid_19.model.preprocess.importation import build_abs_detection_proportion_imported


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
            if clinical in NOTIFICATION_STRATA:
                notification_at_sympt_onset_sources.append(name)

    # Notifications at symptom onset.
    model.request_aggregate_output(
        name="notifications_at_sympt_onset", sources=notification_at_sympt_onset_sources
    )

    # Disease progresssion
    model.request_output_for_flow(name="progress", flow_name="progress")
    for clinical in NOTIFICATION_STRATA:
        model.request_output_for_flow(
            name=f"progressXclinical_{clinical}",
            flow_name="progress",
            dest_strata={"clinical": clinical},
        )

    # New hospital admissions
    model.request_aggregate_output(
        name="new_hospital_admissions",
        sources=[
            f"progressXclinical_{Clinical.ICU}",
            f"progressXclinical_{Clinical.HOSPITAL_NON_ICU}",
        ],
    )
    model.request_aggregate_output(
        name="new_icu_admissions", sources=[f"progressXclinical_{Clinical.ICU}"]
    )

    # Get notifications, which may included people detected in-country as they progress, or imported cases which are detected.
    notification_sources = [f"progressXclinical_{c}" for c in NOTIFICATION_STRATA]
    model.request_aggregate_output(name="local_notifications", sources=notification_sources)
    if params.importation:
        # Include *detected* imported cases in notifications.
        model.request_output_for_flow(
            name="_importation", flow_name="importation", save_results=False
        )
        get_abs_detection_proportion_imported = build_abs_detection_proportion_imported(
            params, AGEGROUP_STRATA
        )
        props_imports_detected = np.array(
            [get_abs_detection_proportion_imported(t) for t in model.times]
        )
        get_count_detected_imports = lambda imports: imports * props_imports_detected
        notification_sources = [*notification_sources, "_importation_detected"]
        model.request_function_output(
            name="_importation_detected",
            func=get_count_detected_imports,
            sources=["_importation"],
            save_results=False,
        )

    model.request_aggregate_output(name="notifications", sources=notification_sources)

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

    # Proprotion seropositive
    model.request_output_for_compartments(
        name="_total_population", compartments=COMPARTMENTS, save_results=False
    )
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
            total_name = f"_total_populationXagegroup_{agegroup}"
            recovered_name = f"_recoveredXagegroup_{agegroup}"
            model.request_output_for_compartments(
                name=total_name,
                compartments=COMPARTMENTS,
                strata={"agegroup": agegroup},
                save_results=False,
            )
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
