import numpy as np

from summer.flow import AgeingFlow
from summer.compartment import Compartment
from summer.stratification import Stratification


def test_create_ageing_flows():
    strat = Stratification(
        name="age",
        strata=["0", "10", "30", "40"],
        compartments=["S", "I"],
        comp_split_props={},
        flow_adjustments={},
    )
    previous_compartments = [
        Compartment("S", strat_names=["location"], strat_values={"location": "home"}),
        Compartment("S", strat_names=["location"], strat_values={"location": "work"}),
        Compartment("I", strat_names=["location"], strat_values={"location": "home"}),
        Compartment("I", strat_names=["location"], strat_values={"location": "work"}),
        Compartment("R", strat_names=["location"], strat_values={"location": "home"}),
        Compartment("R", strat_names=["location"], strat_values={"location": "work"}),
    ]

    ageing_flows, ageing_params = AgeingFlow.create(strat, previous_compartments, _get_param_value)
    assert ageing_params == {"ageing0to10": 0.1, "ageing10to30": 0.05, "ageing30to40": 0.1}

    flow = ageing_flows[0]
    assert flow.source == Compartment.deserialize("SXlocation_homeXage_0")
    assert flow.dest == Compartment.deserialize("SXlocation_homeXage_10")
    assert flow.param_name == "ageing0to10"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[1]
    assert flow.source == Compartment.deserialize("SXlocation_workXage_0")
    assert flow.dest == Compartment.deserialize("SXlocation_workXage_10")
    assert flow.param_name == "ageing0to10"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[2]
    assert flow.source == Compartment.deserialize("IXlocation_homeXage_0")
    assert flow.dest == Compartment.deserialize("IXlocation_homeXage_10")
    assert flow.param_name == "ageing0to10"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[3]
    assert flow.source == Compartment.deserialize("IXlocation_workXage_0")
    assert flow.dest == Compartment.deserialize("IXlocation_workXage_10")
    assert flow.param_name == "ageing0to10"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[4]
    assert flow.source == Compartment.deserialize("SXlocation_homeXage_10")
    assert flow.dest == Compartment.deserialize("SXlocation_homeXage_30")
    assert flow.param_name == "ageing10to30"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[5]
    assert flow.source == Compartment.deserialize("SXlocation_workXage_10")
    assert flow.dest == Compartment.deserialize("SXlocation_workXage_30")
    assert flow.param_name == "ageing10to30"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[6]
    assert flow.source == Compartment.deserialize("IXlocation_homeXage_10")
    assert flow.dest == Compartment.deserialize("IXlocation_homeXage_30")
    assert flow.param_name == "ageing10to30"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[7]
    assert flow.source == Compartment.deserialize("IXlocation_workXage_10")
    assert flow.dest == Compartment.deserialize("IXlocation_workXage_30")
    assert flow.param_name == "ageing10to30"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[8]
    assert flow.source == Compartment.deserialize("SXlocation_homeXage_30")
    assert flow.dest == Compartment.deserialize("SXlocation_homeXage_40")
    assert flow.param_name == "ageing30to40"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[9]
    assert flow.source == Compartment.deserialize("SXlocation_workXage_30")
    assert flow.dest == Compartment.deserialize("SXlocation_workXage_40")
    assert flow.param_name == "ageing30to40"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[10]
    assert flow.source == Compartment.deserialize("IXlocation_homeXage_30")
    assert flow.dest == Compartment.deserialize("IXlocation_homeXage_40")
    assert flow.param_name == "ageing30to40"
    assert flow.param_func == _get_param_value

    flow = ageing_flows[11]
    assert flow.source == Compartment.deserialize("IXlocation_workXage_30")
    assert flow.dest == Compartment.deserialize("IXlocation_workXage_40")
    assert flow.param_name == "ageing30to40"
    assert flow.param_func == _get_param_value


def _get_param_value(name, time):
    return 13
