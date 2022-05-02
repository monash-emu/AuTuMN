import pandas as pd
import numpy as np
import pytest

from autumn.tools.dynamic_proportions.solve_transitions import calculate_transition_rates_from_dynamic_props
from summer import CompartmentalModel


@pytest.mark.github_only
def test_dynamic_props_with_single_flow(n_props_to_test=10, error=1e-4):
    """
    Test the dynamic proportion module in the case of a single flow for which we know the exact solution.
    """
    tested_props = np.random.uniform(size=n_props_to_test)
    active_flows = {"progression": ("A", "B")}

    for prop_final_in_a in tested_props:
        props_df = pd.DataFrame(
            data={
                "A": [1., prop_final_in_a],
                "B": [0., 1. - prop_final_in_a]
            },
            index=[0, 100]
        )
        sc_functions = calculate_transition_rates_from_dynamic_props(props_df, active_flows)
        estimated_rate = sc_functions["progression"](50.)
        true_rate = - np.log(prop_final_in_a) / 100.

        assert abs(estimated_rate - true_rate) <= error


@pytest.mark.github_only
def test_dynamic_props_using_ode_model(n_props_to_test=2, error=5e-2):
    """
    Test the dynamic proportion module by solving the transition rates first and then running an ODE model using these
    parameters. The requested proportions are then compared to the model outputs.
    """
    # List of transtion flows
    active_flows = {
        "vaccination": ("unvaccinated", "vaccinated"),
        "boosting": ("vaccinated", "boosted"),
        "waning": ("boosted", "vaccinated")
    }

    tested_p_vacc = np.random.uniform(size=n_props_to_test)
    tested_p_boost = np.random.uniform(size=n_props_to_test)
    tested_p_waned = np.random.uniform(size=n_props_to_test)

    for p_vacc in tested_p_vacc:
        for p_boost in tested_p_boost:
            for p_waned in tested_p_waned:
                test_props_df = pd.DataFrame(
                    data={
                        "unvaccinated": [1., 1. - p_vacc, 1. - p_vacc, 1. - p_vacc],
                        "vaccinated": [0., p_vacc, p_vacc - p_vacc * p_boost, p_vacc - p_vacc * p_boost + p_vacc * p_boost * p_waned],
                        "boosted": [0., .0, p_vacc * p_boost, p_vacc * p_boost - p_vacc * p_boost * p_waned]
                    },
                    index=[0, 100, 150, 200]
                )
                sc_functions = calculate_transition_rates_from_dynamic_props(test_props_df, active_flows)

                # FIXME: Most of the code below should not be repeated within the loop
                # Define a basic compartmental model
                model = CompartmentalModel(
                    times=[0, 200],
                    compartments=["unvaccinated", "vaccinated", "boosted"],
                    infectious_compartments=["unvaccinated"],
                    timestep=0.1,
                )
                model.set_initial_population(distribution={"unvaccinated": 1})
                model.add_transition_flow(name="vaccination", fractional_rate=sc_functions["vaccination"], source="unvaccinated", dest="vaccinated")
                model.add_transition_flow(name="boosting", fractional_rate=sc_functions["boosting"], source="vaccinated", dest="boosted")
                model.add_transition_flow(name="waning", fractional_rate=sc_functions["waning"], source="boosted", dest="vaccinated")

                # Run the model
                model.run()
                output_df = model.get_outputs_df()

                # Check the outputs against the user requests
                for time in test_props_df.index.to_list():
                    for compartment in test_props_df.columns.to_list():
                        requested_prop = test_props_df.loc[time][compartment]
                        modelled_prop = output_df.loc[time][compartment]

                        assert abs(modelled_prop - requested_prop) <= error
