from applications.covid_19.mixing_optimisation.mixing_opti import *


def test_optimisation_by_age():
    mixing_mult_by_age = {"a": 1., "b": 1., "c": 1.0, "d": 1., "e": 1., "f": 1.}
    h_immu, obj, models = objective_function(decision_variables=mixing_mult_by_age, mode="by_age")


def test_optimisation_by_location():
    mixing_mult_by_location = {"school": 1., "work": 1., "other_locations": 1.}
    h_immu, obj, models = objective_function(decision_variables=mixing_mult_by_location, mode="by_location")
