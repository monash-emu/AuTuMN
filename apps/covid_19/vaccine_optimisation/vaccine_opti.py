from apps.covid_19.vaccine_optimisation.constants import (
    SCENARIO_START_TIME, OPTIMISED_PHASE_DURATION, SEVERITY_EFFICACY, INFECTION_EFFICACY
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
    :param decision_vars:
    :return:
    """
    roll_out_function = {
        "coverage": sum(decision_vars),
        "start_time": 447.,
        "end_time": 797.
    }
    return roll_out_function


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
                "roll_out_function": get_vacc_roll_out_func(decision_vars)
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
