
from summer.model import StratifiedModel
from summer.model.utils.string import find_name_components
from apps.covid_19.constants import Compartment, ClinicalStratum


NOTIFICATION_STRATUM = [ClinicalStratum.SYMPT_ISOLATE, ClinicalStratum.HOSPITAL_NON_ICU, ClinicalStratum.ICU]


def get_calc_notifications_covid(
    include_importation, prop_detected_func,
):
    def calculate_notifications_covid(model: StratifiedModel, time: float):
        """
        Returns the number of notifications for a given time.
        The fully stratified incidence outputs must be available before calling this function
        """
        notifications_count = 0.
        time_idx = model.times.index(time)
        for key, value in model.derived_outputs.items():
            is_progress = "progress" in find_name_components(key)
            is_notify_stratum = any([stratum in key for stratum in NOTIFICATION_STRATUM])
            if is_progress and is_notify_stratum:
                notifications_count += value[time_idx]

        if include_importation:
            notifications_count += (
                    model.time_variants["crude_birth_rate"](time)
                    * sum(model.compartment_values)
                    * prop_detected_func(time)
            )

        return notifications_count

    return calculate_notifications_covid


def calculate_new_hospital_admissions_covid(model, time):
    time_idx = model.times.index(time)
    hosp_admissions = 0.0
    for key, value in model.derived_outputs.items():
        if "progress" in find_name_components(key) and ClinicalStratum.ICU in key:
            hosp_admissions += value[time_idx]
    return hosp_admissions


def calculate_new_icu_admissions_covid(model, time):
    time_idx = model.times.index(time)
    icu_admissions = 0.0
    for key, value in model.derived_outputs.items():
        if "progress" in find_name_components(key) and \
                f"clinical_{ClinicalStratum.ICU}" in find_name_components(key):
            icu_admissions += value[time_idx]
    return icu_admissions


def calculate_icu_prev(model, time):
    icu_prev = 0
    for i_comp, comp_name in enumerate(model.compartment_names):
        if Compartment.LATE_ACTIVE in find_name_components(comp_name) and \
                f"clinical_{ClinicalStratum.ICU}" in find_name_components(comp_name):
            icu_prev += model.compartment_values[i_comp]
    return icu_prev


def calculate_hospital_occupancy(model, time):
    hospital_prev = 0.
    period_icu_patients_in_hospital = \
        max(
            model.parameters["compartment_periods"]["icu_early"] -
            model.parameters["compartment_periods"]["hospital_early"],
            0.
        )
    proportion_icu_patients_in_hospital = \
        period_icu_patients_in_hospital / \
        model.parameters["compartment_periods"]["icu_early"]
    for i_comp, comp_name in enumerate(model.compartment_names):

        # Both ICU and hospital late active compartments
        if Compartment.LATE_ACTIVE in find_name_components(comp_name) and \
                (f"clinical_{ClinicalStratum.ICU}" in find_name_components(comp_name) or
                 f"clinical_{ClinicalStratum.HOSPITAL_NON_ICU}" in find_name_components(comp_name)):
            hospital_prev += model.compartment_values[i_comp]

        # A proportion of the early active ICU compartment
        if Compartment.EARLY_ACTIVE in find_name_components(comp_name) and \
                f"clinical_{ClinicalStratum.ICU}" in comp_name:
            hospital_prev += \
                model.compartment_values[i_comp] * \
                proportion_icu_patients_in_hospital

    return hospital_prev


def calculate_icu_occupancy(model, time):
    icu_prev = 0
    for i_comp, comp_name in enumerate(model.compartment_names):
        if Compartment.LATE_ACTIVE in find_name_components(comp_name) and \
                f"clinical_{ClinicalStratum.ICU}" in find_name_components(comp_name):
            icu_prev += model.compartment_values[i_comp]
    return icu_prev


def calculate_proportion_seropositive(model, time):
    n_seropositive = 0.
    for i_comp, comp_name in enumerate(model.compartment_names):
        if Compartment.RECOVERED in find_name_components(comp_name):
            n_seropositive += model.compartment_values[i_comp]
    return n_seropositive / sum(model.compartment_values)


def get_calculate_years_of_life_lost(life_expectancy_by_agegroup):

    def calculate_years_of_life_lost(model, time):
        time_idx = model.times.index(time)
        total_yoll = 0.
        for i_group, agegroup in enumerate(model.all_stratifications["agegroup"]):
            for derived_output in model.derived_outputs:
                if f"infection_deathsXagegroup_{agegroup}" in derived_output:
                    total_yoll += \
                        model.derived_outputs[derived_output][time_idx] * \
                        life_expectancy_by_agegroup[i_group]

        return total_yoll

    return calculate_years_of_life_lost


def get_progress_connections(stratum_names: str):
    """
    Track "progress": flow from early infectious cases to late infectious cases.
    """
    progress_connections = {
        "progress": {
            "origin": Compartment.EARLY_ACTIVE,
            "to": Compartment.LATE_ACTIVE,
            "origin_condition": "",
            "to_condition": "",
        }
    }
    for stratum_name in stratum_names:
        output_key = f"progressX{stratum_name}"
        progress_connections[output_key] = {
            "origin": Compartment.EARLY_ACTIVE,
            "to": Compartment.LATE_ACTIVE,
            "origin_condition": "",
            "to_condition": stratum_name,
        }

    return progress_connections


def get_incidence_connections(stratum_names: str):
    """
    Track "incidence": flow from presymptomatic cases to infectious cases.
    """
    incidence_connections = {
        "incidence": {
            "origin": Compartment.LATE_EXPOSED,
            "to": Compartment.EARLY_ACTIVE,
            "origin_condition": "",
            "to_condition": "",
        }
    }
    for stratum_name in stratum_names:
        output_key = f"incidenceX{stratum_name}"
        incidence_connections[output_key] = {
            "origin": Compartment.LATE_EXPOSED,
            "to": Compartment.EARLY_ACTIVE,
            "origin_condition": "",
            "to_condition": stratum_name,
        }

    return incidence_connections
