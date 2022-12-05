import copy
from importlib import import_module

from autumn.core.project import get_project, Project


class Opti:
    """
    This class is used to define and solve an optimisation problem based on one of the existing AuTuMN apps.
    """

    project: Project

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
        self.project = get_project(app_name, region_name)

    def run_root_model(self) -> dict:
        # Update params using root_model_params
        root_params = self.project.param_set.baseline.update(self.root_model_params)
        self.root_model = self.project.run_baseline_model(root_params)
        return root_params.to_dict()

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

        sc_params = self.project.param_set.baseline.update(self.root_model_params).update(sc_dict)
        sc_models = self.project.run_scenario_models(
            baseline_model=self.root_model, scenario_params=[sc_params]
        )
        return sc_models[0]
