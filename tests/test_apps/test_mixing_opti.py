from applications.covid_19.mixing_optimisation.mixing_opti import objective_function


def test_optimisation_by_age():
    mixing_mult_by_age = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0, "f": 1.0}
    h_immu, obj, models = objective_function(decision_variables=mixing_mult_by_age, mode="by_age")


def test_optimisation_by_location():
    mixing_mult_by_location = {"school": 1.0, "work": 1.0, "other_locations": 1.0}
    h_immu, obj, models = objective_function(
        decision_variables=mixing_mult_by_location, mode="by_location"
    )
