from python_source_code.summer_model import *

my_model = StratifiedModel(
    times=numpy.linspace(1800., 2020.0, 201).tolist(),
    compartment_types=["S", "I"],
    initial_conditions={'I': 1.},
    parameters={"beta": 1.5, "recovery": 1.},
    requested_flows=[
        {"type": "infection_frequency", "parameter": "beta", "origin": "S", "to": "I"},
        {"type": "standard_flows", "parameter": "recovery", "origin": "I", "to": "S"}
    ],
    infectious_compartment='I',
    birth_approach='replace_deaths',
    entry_compartment='S',
    starting_population=100
)

create_flowchart(my_model)

my_model.run_model()

my_model.plot_compartment_size(['I'])
