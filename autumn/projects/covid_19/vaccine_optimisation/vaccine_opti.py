from autumn.projects.covid_19.vaccine_optimisation.constants import (
    SCENARIO_START_TIME,
    PHASE_2_DURATION,
    PHASE_3_DURATION,
    SEVERITY_EFFICACY,
    INFECTION_EFFICACY,
    TOTAL_DAILY_DOSES,
    VACC_AGE_GROUPS,
)
from autumn.optimisation.opti import Opti

APP_NAME = "covid_19"
PHASE_2_END = SCENARIO_START_TIME + PHASE_2_DURATION
SIMULATION_END_TIME = PHASE_2_END + PHASE_3_DURATION
ROOT_MODEL_PARAMS = {
    "time": {"end": SCENARIO_START_TIME + 1},
}


def get_decision_vars_names():
    names = []
    for n_phase in [2, 3]:
        for age_group in VACC_AGE_GROUPS:
            age_gorup_name = f"age_{age_group[0]}"
            if age_group[1] is None:
                age_gorup_name += "+"
            else:
                age_gorup_name += f"_{age_group[1]}"

            names.append(f"prop_{age_gorup_name}_phase_{n_phase}")
    names.append("relaxation")

    return names


def initialise_opti_object(country):
    opti_object = Opti(APP_NAME, country, root_model_params=ROOT_MODEL_PARAMS)
    root_params = opti_object.run_root_model()  # run the baseline
    opti_object.scenario_func = make_scenario_func(root_params, country)
    opti_object.objective_func = make_objective_func()

    return opti_object


def get_vacc_roll_out_func(decision_vars, country):
    """
    Build the vaccine roll-out components based on the decision variables
    :param decision_vars: list of 16 floats representing the age-specific allocation proportions for phase 2 and phase 3
    :return: list of vaccine roll-out components
    """
    roll_out_components = []

    for i_age_group, age_group in enumerate(VACC_AGE_GROUPS):
        daily_doses_phase_2 = float(TOTAL_DAILY_DOSES[country] * decision_vars[i_age_group])
        daily_doses_phase_3 = float(
            TOTAL_DAILY_DOSES[country] * decision_vars[i_age_group + len(VACC_AGE_GROUPS)]
        )
        component = {
            "age_min": age_group[0],
            "supply_timeseries": {
                "times": [
                    SCENARIO_START_TIME - 1,
                    SCENARIO_START_TIME,
                    PHASE_2_END,
                    PHASE_2_END + 1,
                ],
                "values": [0.0, daily_doses_phase_2, daily_doses_phase_2, daily_doses_phase_3],
            },
        }
        if age_group[1] is not None:
            component["age_max"] = age_group[1]
        roll_out_components.append(component)

    return roll_out_components


def make_scenario_func(root_params, country):
    def scenario_func(decision_vars):
        """
        The decision vars are the age-specific allocation proportions
        :param decision_vars: a list of length 16, summing to 1
        :return: a dictionary defining the optimisation scenario
        """

        gap_close = decision_vars[-1]

        sc_dict = {
            "time": {"start": SCENARIO_START_TIME, "end": SIMULATION_END_TIME},
            # VACCINATION
            "vaccination": {
                "severity_efficacy": SEVERITY_EFFICACY,
                "infection_efficacy": INFECTION_EFFICACY,
                "roll_out_components": get_vacc_roll_out_func(decision_vars, country),
            },
            # Prepare placeholder for mobility
            "mobility": {"mixing": {}},
        }

        # LIFTING RESTRICTIONS
        last_value = {"work": ["repeat_prev"], "other_locations": ["repeat_prev"]}
        if "school" in root_params["mobility"]["mixing"]:
            last_school_value = root_params["mobility"]["mixing"]["school"]["values"][-1]
        else:
            last_school_value = 1.0
        last_value["school"] = last_school_value
        should_append = {"work": True, "other_locations": True, "school": False}
        for location in ["work", "other_locations", "school"]:
            sc_dict["mobility"]["mixing"][location] = {
                "append": should_append[location],
                "times": [PHASE_2_END, PHASE_2_END + 1],
                "values": [last_value[location], ["close_gap_to_1", gap_close]],
            }

        return sc_dict

    return scenario_func


def make_objective_func():
    def objective_func(run_model, decision_vars):
        """
        Calculates multiple objectives
        :return: list of objectives
        """
        total_deaths = sum(run_model.derived_outputs["infection_deaths"])
        max_hospital = max(run_model.derived_outputs["hospital_occupancy"])
        closing_gap = decision_vars[-1]

        return [total_deaths, max_hospital, closing_gap]

    return objective_func
