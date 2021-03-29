from apps.covid_19.vaccine_optimisation.constants import (
    SCENARIO_START_TIME, OPTIMISED_PHASE_DURATION, SEVERITY_EFFICACY, INFECTION_EFFICACY, TOTAL_DAILY_DOSES
)

APP_NAME = "covid_19"
OPTIMISED_PHASE_END = SCENARIO_START_TIME + OPTIMISED_PHASE_DURATION
SIMULATION_END_TIME = OPTIMISED_PHASE_END + 14
ROOT_MODEL_PARAMS = {
    "time": {
        "end": SCENARIO_START_TIME + 1
    },
    "stratify_by_immunity": True
}


def get_vacc_roll_out_func(decision_vars):
    """
    Build the vaccine roll-out components based on the decision variables
    :param decision_vars: list of 16 floats representing the age-specific allocation proportions
    :return: list of vaccine roll-out components
    """
    roll_out_components = []

    for i, allocation_prop in enumerate(decision_vars):
        daily_doses = allocation_prop * TOTAL_DAILY_DOSES
        component = {
            "age_min": i * 5,
            "supply_timeseries": {
                "times": [SCENARIO_START_TIME - 1, SCENARIO_START_TIME, OPTIMISED_PHASE_END, OPTIMISED_PHASE_END + 1],
                "values": [0., daily_doses, daily_doses, 0.]
            }
        }
        if i < 15:
            component["age_max"] = i * 5 + 4
        roll_out_components.append(component)

    return roll_out_components


def make_scenario_func(root_params):

    def scenario_func(decision_vars):
        """
        The decision vars are the age-specific allocation proportions
        :param decision_vars: a list of length 16, summing to 1
        :return: a dictionary defining the optimisation scenario
        """
        sc_dict = {
            "time": {
                "start": SCENARIO_START_TIME,
                "end": SIMULATION_END_TIME
            },
            # VACCINATION
            "vaccination": {
                "severity_efficacy": SEVERITY_EFFICACY,
                "infection_efficacy": INFECTION_EFFICACY,
                "roll_out_components": get_vacc_roll_out_func(decision_vars)
            },
            # Prepare placeholder for mobility
            "mobility": {
                "mixing": {}
            }
        }

        # LIFTING RESTRICTIONS
        last_value = {"work": ['repeat_prev'], "other_locations": ['repeat_prev'],
                      "school": root_params["default"]["mobility"]["mixing"]["school"]["values"][-1]}
        should_append = {"work": True, "other_locations": True, "school": False}
        for location in ["work", "other_locations", "school"]:
            sc_dict["mobility"]["mixing"][location] = {
                "append": should_append[location],
                "times": [OPTIMISED_PHASE_END, OPTIMISED_PHASE_END + 1],
                "values": [last_value[location], 1.]
            }

        return sc_dict

    return scenario_func


def make_objective_func():

    def objective_func(run_model):
        """
        Calculates modelled incidence 14 days after optimisation window
        :param run_model:
        :return:
        """
        incidence = run_model.derived_outputs["incidence"][-1]
        return [incidence]

    return objective_func
