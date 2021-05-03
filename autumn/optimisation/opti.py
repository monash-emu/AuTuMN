import copy
from importlib import import_module

from autumn.utils.params import merge_dicts
from autumn.utils.scenarios import Scenario


class Opti:
    """
    This class is used to define and solve an optimisation problem based on one of the existing AuTuMN apps.
    """
    def __init__(
            self,
            app_name,  # e.g. 'covid_19'  or  'tuberculosis'
            region_name,  # e.g. 'victoria'
            scenario_func=None,  # a function that returns a scenario dictionary based on decision variables
            objective_func=None,  # a function that calculates the objective(s) based on a run model. Should return a list
            root_model_params={},  # a params dictionary to update the baseline params
    ):
        self.app_name = app_name
        self.region_name = region_name
        self.scenario_func = scenario_func
        self.objective_func = objective_func
        self.root_model_params = root_model_params
        self.root_model = None

    def run_root_model(self):
        # Initialise baseline model
        app_module = import_module(f"apps.{self.app_name}")
        app_region = app_module.app.get_region(self.region_name)
        root_params = copy.deepcopy(app_region.params)

        # Update params using root_model_params
        root_params["default"] = merge_dicts(self.root_model_params, root_params["default"])

        # Create Scenario object and run root model
        root_scenario = Scenario(app_region.build_model, idx=0, params=root_params)
        root_scenario.run()

        self.root_model = root_scenario.model

        return root_params

    def evaluate_objective(self, decision_vars):
        """
        Evaluate the objective function(s) for the given decision variables
        :return: a list of objective values
        """
        assert self.scenario_func is not None, "A non-null scenario function is required."
        assert self.objective_func is not None, "A non-null objective function is required."

        sc_dict = self.scenario_func(decision_vars)
        sc_model = self.run_scenario(sc_dict)
        objective = self.objective_func(sc_model, decision_vars)

        return objective

    def run_scenario(self, sc_dict):
        """
        Run a model scenario defined from the scenario dictionary
        :return: a model object
        """
        app_module = import_module(f"apps.{self.app_name}")
        app_region = app_module.app.get_region(self.region_name)

        build_model = app_region.build_model
        params = copy.deepcopy(app_region.params)
        params["default"] = merge_dicts(self.root_model_params, params["default"])

        # Create and run the optimisation scenario
        params["scenarios"][1] = merge_dicts(sc_dict, params["default"])
        opti_scenario = Scenario(build_model, idx=1, params=params)
        opti_scenario.run(base_model=self.root_model)

        return opti_scenario.model

