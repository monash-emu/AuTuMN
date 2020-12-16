import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from summer2.model import CompartmentalModel


def test_no_derived_outputs():

    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.run()
    # 6 timesteps, 3 compartments.
    assert model.outputs.shape == (6, 3)
    assert model.derived_outputs == {}


def test_compartment_size_derived_outputs():

    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.outputs = np.array(
        [
            [990, 10, 0],
            [980, 15, 5],
            [970, 20, 10],
            [960, 25, 15],
            [950, 30, 20],
            [940, 35, 25],
        ]
    )
    model.request_output_for_compartments("recovered", ["R"])
    model.request_output_for_compartments("not_infected", ["S", "R"])
    model.request_output_for_compartments("total_population", ["S", "I", "R"])
    dos = model._calculate_derived_outputs()
    assert_array_equal(dos["recovered"], np.array([0, 5, 10, 15, 20, 25]))
    assert_array_equal(dos["not_infected"], np.array([990, 985, 980, 975, 970, 965]))
    assert_array_equal(dos["total_population"], np.array([1000, 1000, 1000, 1000, 1000, 1000]))


def test_aggregate_derived_outputs():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.outputs = np.array(
        [
            [990, 10, 0],
            [980, 15, 5],
            [970, 20, 10],
            [960, 25, 15],
            [950, 30, 20],
            [940, 35, 25],
        ]
    )
    model.request_output_for_compartments("recovered", ["R"])
    model.request_output_for_compartments("not_infected", ["S", "R"])
    model.request_output_for_compartments("total_population", ["S", "I", "R"])
    model.request_aggregate_output(name="my_aggregate", sources=["recovered", "total_population"])
    dos = model._calculate_derived_outputs()
    assert_array_equal(dos["my_aggregate"], np.array([1000, 1005, 1010, 1015, 1020, 1025]))


def test_cumulative_derived_outputs():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.outputs = np.array(
        [
            [990, 10, 0],
            [980, 15, 5],
            [970, 20, 10],
            [960, 25, 15],
            [950, 30, 20],
            [940, 35, 25],
        ]
    )
    model.request_output_for_compartments("recovered", ["R"])
    model.request_cumulative_output(name="recoved_cumulative", source="recovered")
    model.request_cumulative_output(name="recoved_cumulative_2", source="recovered", start_time=2)
    dos = model._calculate_derived_outputs()
    assert_array_equal(dos["recoved_cumulative"], np.array([0, 5, 15, 30, 50, 75]))
    assert_array_equal(dos["recoved_cumulative_2"], np.array([0, 0, 10, 25, 45, 70]))


def test_flow_derived_outputs():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})

    # Constant entry.
    model.add_importation_flow("imports", num_imported=2, dest="S")
    model.request_output_for_flow(name="importation", flow_name="imports")
    model.request_output_for_flow(name="importation_raw", flow_name="imports", raw_results=True)

    # Linear entry.
    model.add_importation_flow("imports_land", num_imported=lambda t: 3 * t, dest="S")
    model.request_output_for_flow(name="importation_land", flow_name="imports_land")

    # Quadratic entry.
    model.add_importation_flow("imports_air", num_imported=lambda t: t ** 2, dest="S")
    model.request_output_for_flow(name="importation_air", flow_name="imports_air")

    # Fractional transition flow
    model.add_fractional_flow("recovery", 0.1, "I", "R")
    model.request_output_for_flow(name="recovery_raw", flow_name="recovery", raw_results=True)
    model.request_output_for_flow(name="recovery_delta", flow_name="recovery", raw_results=False)

    model.run()
    dos = model.derived_outputs

    # Raw outputs are the instantaneous flow rate at a given time.
    assert_allclose(dos["recovery_raw"], model.outputs[:, 1] * 0.1, rtol=0.01)

    # Post-processed outputs reflect changes in compartment size.
    recovered_count = np.zeros_like(model.outputs[:, 2])
    recovered_count[1:] = np.diff(model.outputs[:, 2])
    assert_allclose(dos["recovery_delta"], recovered_count, rtol=0.01)

    # Good match for constant
    assert_array_equal(dos["importation"], np.array([0, 2, 2, 2, 2, 2]))
    assert_array_equal(dos["importation_raw"], np.array([2, 2, 2, 2, 2, 2]))

    # Good match for linear
    assert_allclose(dos["importation_land"], np.array([0.0, 1.5, 4.5, 7.5, 10.5, 13.5]))
    # So-so match for quadratic
    assert_allclose(dos["importation_air"], np.array([0.0, 0.6, 2.6, 6.8, 13.0, 20.7]), rtol=0.1)


def test_functional_derived_outputs():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.outputs = np.array(
        [
            [990, 10, 0],
            [980, 15, 5],
            [970, 20, 10],
            [960, 25, 15],
            [950, 30, 20],
            [940, 35, 25],
        ]
    )

    def get_non_infected(rec, pop):
        return rec / pop

    model.request_output_for_compartments("recovered", ["R"])
    model.request_output_for_compartments("population", ["S", "I", "R"])
    model.request_function_output("recovered_prop", get_non_infected, ["recovered", "population"])

    dos = model._calculate_derived_outputs()
    assert_array_equal(dos["recovered_prop"], np.array([0, 0.005, 0.01, 0.015, 0.020, 0.025]))


def test_derived_outputs_with_no_save_results():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_importation_flow("imports", num_imported=2, dest="S")

    # Expect np.array([0, 2, 2, 2, 2, 2]))
    model.request_output_for_flow(name="importation", flow_name="imports", save_results=False)
    # Expect np.array([0, 5, 10, 15, 20, 25]))
    model.request_output_for_compartments("recovered", ["R"], save_results=False)
    # Expect np.array([0, 5, 15, 30, 50, 75]))
    model.request_cumulative_output(
        name="recovered_cumulative", source="recovered", save_results=False
    )
    # Expect np.array([0, 7, 12, 17, 22, 227]))
    model.request_aggregate_output(
        name="some_aggregate", sources=["recovered", "importation"], save_results=False
    )
    # Expect np.array([  0,  12,  27,  47,  72, 102])
    model.request_aggregate_output(
        name="final_aggregate", sources=["some_aggregate", "recovered_cumulative"]
    )

    model.run()
    # Override outputs so the test is easier to write
    model.outputs = np.array(
        [
            [990, 10, 0],
            [980, 15, 5],
            [970, 20, 10],
            [960, 25, 15],
            [950, 30, 20],
            [940, 35, 25],
        ]
    )
    dos = model._calculate_derived_outputs()
    assert_array_equal(dos["final_aggregate"], np.array([0, 12, 27, 47, 72, 102]))
    assert "importation" not in dos
    assert "recovered" not in dos
    assert "recovered_cumulative" not in dos
    assert "some_aggregate" not in dos
