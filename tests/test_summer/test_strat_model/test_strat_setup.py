"""
Test setup of the stratified model
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from summer.model import StratifiedModel
from summer.constants import (
    Compartment,
    Flow,
    BirthApproach,
    IntegrationType,
)


@pytest.mark.skip
def test_stratify_flows_partial():
    """Ensure flows get stratified properly in partial strat"""
    requested_flows = [
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": "sus",
            "to": "inf",
        },
        {
            "type": Flow.STANDARD,
            "parameter": "recovery_rate",
            "origin": "inf",
            "to": "sus",
        },
        {
            "type": Flow.DEATH,
            "parameter": "infect_death",
            "origin": "inf",
        },
    ]
    parameters = {
        "contact_rate": 1000,
        "recovery_rate": "recovery_rate",
        "infect_death": 10,
    }
    recovery_rate = lambda t: 2 * t
    model = StratifiedModel(
        times=[0, 1, 2, 3, 4, 5],
        compartment_names=["sus", "inf"],
        initial_conditions={"inf": 200},
        parameters=parameters,
        requested_flows=requested_flows,
        starting_population=1000,
        infectious_compartments=["inf"],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment="sus",
    )
    model.time_variants["recovery_rate"] = recovery_rate
    assert model.compartment_names == ["sus", "inf"]
    assert model.flows == requested_flows
    assert model.parameters == parameters
    assert model.time_variants == {"recovery_rate": recovery_rate}
    assert model.overwrite_parameters == []
    assert model.adaptation_functions == {}

    vals = np.array([800, 200])
    assert_array_equal(model.compartment_values, vals)
    model.stratify(
        "location",
        strata_request=["home", "work", "other"],
        compartments_to_stratify=["inf"],
        requested_proportions={"home": 0.6},
    )
    assert model.flows == [
        {
            "origin": "sus",
            "parameter": "contact_rateXlocation_home",
            "to": "infXlocation_home",
            "type": Flow.INFECTION_FREQUENCY,
        },
        {
            "origin": "sus",
            "parameter": "contact_rateXlocation_work",
            "to": "infXlocation_work",
            "type": Flow.INFECTION_FREQUENCY,
        },
        {
            "origin": "sus",
            "parameter": "contact_rateXlocation_other",
            "to": "infXlocation_other",
            "type": Flow.INFECTION_FREQUENCY,
        },
        {
            "origin": "infXlocation_home",
            "parameter": "recovery_rateXlocation_home",
            "to": "sus",
            "type": Flow.STANDARD,
        },
        {
            "origin": "infXlocation_work",
            "parameter": "recovery_rateXlocation_work",
            "to": "sus",
            "type": Flow.STANDARD,
        },
        {
            "origin": "infXlocation_other",
            "parameter": "recovery_rateXlocation_other",
            "to": "sus",
            "type": Flow.STANDARD,
        },
        {
            "origin": "infXlocation_home",
            "parameter": "infect_death",
            "type": Flow.DEATH,
        },
        {
            "origin": "infXlocation_work",
            "parameter": "infect_death",
            "type": Flow.DEATH,
        },
        {
            "origin": "infXlocation_other",
            "parameter": "infect_death",
            "type": Flow.DEATH,
        },
    ]
    assert model.parameters["infect_death"] == 10
    assert model.parameters["contact_rate"] == 1000
    assert model.parameters["contact_rateXlocation_home"] == 1 / 3
    assert model.parameters["contact_rateXlocation_other"] == 1 / 3
    assert model.parameters["contact_rateXlocation_work"] == 1 / 3
    assert model.parameters["recovery_rate"] == "recovery_rate"
    assert model.parameters["recovery_rateXlocation_home"] == 1 / 3
    assert model.parameters["recovery_rateXlocation_other"] == 1 / 3
    assert model.parameters["recovery_rateXlocation_work"] == 1 / 3
    assert model.time_variants == {"recovery_rate": recovery_rate}
    assert model.adaptation_functions["contact_rateXlocation_home"](t=1, v=1) == 1 / 3
    assert model.adaptation_functions["contact_rateXlocation_home"](t=1, v=2) == 2 / 3
    assert model.adaptation_functions["contact_rateXlocation_work"](t=1, v=1) == 1 / 3
    assert model.adaptation_functions["contact_rateXlocation_work"](t=1, v=2) == 2 / 3
    assert model.adaptation_functions["contact_rateXlocation_other"](t=1, v=1) == 1 / 3
    assert model.adaptation_functions["contact_rateXlocation_other"](t=1, v=2) == 2 / 3
    assert model.adaptation_functions["recovery_rateXlocation_home"](t=1, v=1) == 1 / 3
    assert model.adaptation_functions["recovery_rateXlocation_home"](t=1, v=2) == 2 / 3
    assert model.adaptation_functions["recovery_rateXlocation_work"](t=1, v=1) == 1 / 3
    assert model.adaptation_functions["recovery_rateXlocation_work"](t=1, v=2) == 2 / 3
    assert model.adaptation_functions["recovery_rateXlocation_other"](t=1, v=1) == 1 / 3
    assert model.adaptation_functions["recovery_rateXlocation_other"](t=1, v=2) == 2 / 3

    # FIXME: WTF
    # model.prepare_to_run()


@pytest.mark.skip
def test_stratify_flows_full():
    """Ensure flows get stratified properly in full strat"""
    requested_flows = [
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": "sus",
            "to": "inf",
        },
        {
            "type": Flow.STANDARD,
            "parameter": "recovery_rate",
            "origin": "inf",
            "to": "sus",
        },
        {
            "type": Flow.DEATH,
            "parameter": "infect_death",
            "origin": "inf",
        },
    ]
    parameters = {
        "contact_rate": 1000,
        "recovery_rate": "recovery_rate",
        "infect_death": 10,
    }
    recovery_rate = lambda t: 2 * t
    model = StratifiedModel(
        times=[0, 1, 2, 3, 4, 5],
        compartment_names=["sus", "inf"],
        initial_conditions={"inf": 200},
        parameters=parameters,
        requested_flows=requested_flows,
        starting_population=1000,
        infectious_compartments=["inf"],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment="sus",
    )
    model.time_variants["recovery_rate"] = recovery_rate
    assert model.compartment_names == ["sus", "inf"]
    assert model.flows == requested_flows
    assert model.parameters == parameters
    assert model.time_variants == {"recovery_rate": recovery_rate}
    assert model.overwrite_parameters == []
    assert model.adaptation_functions == {}

    vals = np.array([800, 200])
    assert_array_equal(model.compartment_values, vals)
    model.stratify(
        "location",
        strata_request=["home", "work", "other"],
        compartments_to_stratify=[],
        requested_proportions={"home": 0.6},
    )
    assert model.flows == [
        {
            "origin": "susXlocation_home",
            "parameter": "contact_rate",
            "to": "infXlocation_home",
            "type": Flow.INFECTION_FREQUENCY,
        },
        {
            "origin": "susXlocation_work",
            "parameter": "contact_rate",
            "to": "infXlocation_work",
            "type": Flow.INFECTION_FREQUENCY,
        },
        {
            "origin": "susXlocation_other",
            "parameter": "contact_rate",
            "to": "infXlocation_other",
            "type": Flow.INFECTION_FREQUENCY,
        },
        {
            "origin": "infXlocation_home",
            "parameter": "recovery_rate",
            "to": "susXlocation_home",
            "type": Flow.STANDARD,
        },
        {
            "origin": "infXlocation_work",
            "parameter": "recovery_rate",
            "to": "susXlocation_work",
            "type": Flow.STANDARD,
        },
        {
            "origin": "infXlocation_other",
            "parameter": "recovery_rate",
            "to": "susXlocation_other",
            "type": Flow.STANDARD,
        },
        {
            "origin": "infXlocation_home",
            "parameter": "infect_death",
            "type": Flow.DEATH,
        },
        {
            "origin": "infXlocation_work",
            "parameter": "infect_death",
            "type": Flow.DEATH,
        },
        {
            "origin": "infXlocation_other",
            "parameter": "infect_death",
            "type": Flow.DEATH,
        },
    ]
    assert model.parameters["contact_rate"] == 1000
    assert model.parameters["infect_death"] == 10
    assert model.parameters["recovery_rate"] == "recovery_rate"
    assert model.time_variants == {"recovery_rate": recovery_rate}


@pytest.mark.skip
def test_stratify_flows_full__with_adjustment_requests():
    """Ensure flows get stratified properly in full strat"""
    adjustment_requests = {
        "contact_rate": {
            "home": "home_contact_rate",
            "work": 0.5,
        },
        "recovery_rate": {
            "home": 1,
            "work": 2,
        },
        "infect_death": {
            "home": 1,
            "work": 2,
        },
    }
    requested_flows = [
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": "sus",
            "to": "inf",
        },
        {
            "type": Flow.STANDARD,
            "parameter": "recovery_rate",
            "origin": "inf",
            "to": "sus",
        },
        {
            "type": Flow.DEATH,
            "parameter": "infect_death",
            "origin": "inf",
        },
    ]
    parameters = {
        "contact_rate": 1000,
        "recovery_rate": "recovery_rate",
        "infect_death": 10,
    }
    home_contact_rate_func = lambda t: t
    recovery_rate_func = lambda t: 2 * t
    model = StratifiedModel(
        times=[0, 1, 2, 3, 4, 5],
        compartment_names=["sus", "inf"],
        initial_conditions={"inf": 200},
        parameters=parameters,
        requested_flows=requested_flows,
        starting_population=1000,
        infectious_compartments=["inf"],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment="sus",
    )
    model.time_variants["recovery_rate"] = recovery_rate_func
    model.time_variants["home_contact_rate"] = home_contact_rate_func
    assert model.compartment_names == ["sus", "inf"]
    assert model.flows == requested_flows
    assert model.parameters == parameters
    assert model.time_variants == {
        "recovery_rate": recovery_rate_func,
        "home_contact_rate": home_contact_rate_func,
    }
    assert model.overwrite_parameters == []
    assert model.adaptation_functions == {}

    vals = np.array([800, 200])
    assert_array_equal(model.compartment_values, vals)
    model.stratify(
        "location",
        strata_request=["home", "work"],
        compartments_to_stratify=[],
        requested_proportions={"home": 0.6},
        adjustment_requests=adjustment_requests,
    )
    assert model.flows == [
        {
            "origin": "susXlocation_home",
            "parameter": "contact_rateXlocation_home",
            "to": "infXlocation_home",
            "type": Flow.INFECTION_FREQUENCY,
        },
        {
            "origin": "susXlocation_work",
            "parameter": "contact_rateXlocation_work",
            "to": "infXlocation_work",
            "type": Flow.INFECTION_FREQUENCY,
        },
        {
            "origin": "infXlocation_home",
            "parameter": "recovery_rateXlocation_home",
            "to": "susXlocation_home",
            "type": Flow.STANDARD,
        },
        {
            "origin": "infXlocation_work",
            "parameter": "recovery_rateXlocation_work",
            "to": "susXlocation_work",
            "type": Flow.STANDARD,
        },
        {
            "origin": "infXlocation_home",
            "parameter": "infect_deathXlocation_home",
            "type": Flow.DEATH,
        },
        {
            "origin": "infXlocation_work",
            "parameter": "infect_deathXlocation_work",
            "type": Flow.DEATH,
        },
    ]
    assert model.time_variants == {
        "recovery_rate": recovery_rate_func,
        "home_contact_rate": home_contact_rate_func,
    }
    assert model.parameters["contact_rate"] == 1000
    assert model.parameters["infect_death"] == 10
    assert model.parameters["recovery_rate"] == "recovery_rate"
    assert model.parameters["contact_rateXlocation_home"] == "home_contact_rate"
    assert model.parameters["contact_rateXlocation_work"] == 0.5
    assert model.parameters["recovery_rateXlocation_home"] == 1
    assert model.parameters["recovery_rateXlocation_work"] == 2
    assert model.parameters["infect_deathXlocation_home"] == 1
    assert model.parameters["infect_deathXlocation_work"] == 2


@pytest.mark.skip
def test_stratify_compartments_full():
    """Ensure compartment names and values get fully stratified properly"""
    model = StratifiedModel(
        times=[0, 1, 2, 3, 4, 5],
        compartment_names=["sus", "inf"],
        initial_conditions={"inf": 200},
        parameters={},
        requested_flows=[],
        starting_population=1000,
        infectious_compartments=["inf"],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment="sus",
    )
    assert model.compartment_names == ["sus", "inf"]
    vals = np.array([800, 200])
    assert_array_equal(model.compartment_values, vals)
    model.stratify(
        "location",
        strata_request=["home", "work", "other"],
        compartments_to_stratify=[],
        requested_proportions={"home": 0.6},
    )
    assert model.compartment_names == [
        "susXlocation_home",
        "susXlocation_work",
        "susXlocation_other",
        "infXlocation_home",
        "infXlocation_work",
        "infXlocation_other",
    ]
    vals = np.array([480, 160, 160, 120, 40, 40])
    assert_array_equal(model.compartment_values, vals)
    assert model.all_stratifications == {"location": ["home", "work", "other"]}
    assert model.full_stratification_list == ["location"]


@pytest.mark.skip
def test_stratify_compartments_partial():
    """Ensure compartment names and values get partially stratified properly"""
    model = StratifiedModel(
        times=[0, 1, 2, 3, 4, 5],
        compartment_names=["sus", "inf"],
        initial_conditions={"inf": 200},
        parameters={},
        requested_flows=[],
        starting_population=1000,
        infectious_compartments=["inf"],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment="sus",
    )
    assert model.compartment_names == ["sus", "inf"]
    vals = np.array([800, 200])
    assert_array_equal(model.compartment_values, vals)
    model.stratify(
        "location",
        strata_request=["home", "work", "other"],
        compartments_to_stratify=["sus"],
        requested_proportions={"home": 0.6},
    )
    assert model.compartment_names == [
        "susXlocation_home",
        "susXlocation_work",
        "susXlocation_other",
        "inf",
    ]
    vals = np.array([480, 160, 160, 200])
    assert_array_equal(model.compartment_values, vals)
    assert model.all_stratifications == {"location": ["home", "work", "other"]}
    assert model.full_stratification_list == []
