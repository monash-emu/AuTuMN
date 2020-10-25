"""
Ensure that the EpiModel model produces the correct infectious multipliers
"""
import numpy as np
from summer.model import EpiModel
from summer.constants import Flow, BirthApproach
from summer.compartment import Compartment

MODEL_KWARGS = {
    "times": np.array([0.0, 1, 2, 3, 4, 5]),
    "compartment_names": ["S", "I"],
    "initial_conditions": {"I": 10},
    "parameters": {},
    "requested_flows": [],
    "starting_population": 1000,
    "infectious_compartments": ["I"],
    "birth_approach": BirthApproach.NO_BIRTH,
    "entry_compartment": "S",
}


def test_get_infection_frequency_multipier():
    model = EpiModel(**MODEL_KWARGS)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    c_src = Compartment("S")
    c_dst = Compartment("I")
    multiplier = model.get_infection_frequency_multipier(c_src, c_dst)
    assert model.population_infectious == 10
    assert model.population_total == 1000
    assert multiplier == 0.01


def test_get_infection_density_multipier():
    model = EpiModel(**MODEL_KWARGS)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    c_src = Compartment("S")
    c_dst = Compartment("I")
    multiplier = model.get_infection_density_multipier(c_src, c_dst)
    assert model.population_infectious == 10
    assert model.population_total == 1000
    assert multiplier == 10
