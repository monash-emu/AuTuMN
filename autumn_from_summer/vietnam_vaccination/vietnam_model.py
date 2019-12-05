from summer_py.summer_model import *
from summer_py.post_processing import *
from summer_py.outputs import *
import os

# A test trial of creating a model that has the same compartments as the one "Data needs for evidence-based decisions: a tuberculosis modeler’s ‘wish list’

my_compartments = ['susceptible', 'recent_latent', 'remote_latent', 'infectious', 'recovered']
my_times = numpy.linspace(0., 100., 101).tolist()
flows = [{'type': 'infection', 'origin': 'susceptible', 'to': 'recent_latent', 'parameter': 'infection_rate'},
         {'type': 'immune_stabilisation', 'origin': 'recent_latent', 'to': 'remote_latent', 'parameter': 'immune_stabilisation_rate'},
         {'type': 'reactivation', 'origin': 'remote_latent', 'to': 'infectious', 'parameter': 'reactivation_rate'},
         {'type': 'reinfection_1', 'origin': 'remote_latent', 'to': 'recent_latent', 'parameter': 'reinfection_rate_1'},
         {'type': 'reinfection_2', 'origin': 'recovered', 'to': 'recent_latent', 'parameter': 'reinfection_rate_2'},
         {'type': 'relapse', 'origin': 'recovered', 'to': 'infectious', 'parameter': 'relapse_rate'},
         {'type': 'cure', 'origin': 'infectious', 'to': 'recovered', 'parameter': 'cure_rate'},
         {'type': 'rapid_progression', 'origin': 'recent_latent', 'to': 'infectious', 'parameter': 'rapid_progression_rate'}]

my_parameters = {'infection_rate': 0.01, 'immune_stabilisation_rate': 0.001, 'reactivation_rate': 0.0004,
                 'reinfection_rate_1': 0.00005, 'reinfection_rate_2': 0.0002, 'cure_rate': 0.1,
                 'rapid_progression_rate': 0.001, 'relapse_rate': 0.002}

my_initial_conditions = {'susceptible': 10000., 'infectious': 500.}

my_model = EpiModel(times=my_times, compartment_types=my_compartments, initial_conditions=my_initial_conditions,
                           parameters=my_parameters, requested_flows=flows, starting_population=10500,
                           infectious_compartment=('infectious',), entry_compartment='susceptible')

create_flowchart(my_model)
print(os.getcwd())

my_model.run_model()

print(my_model.outputs)

pp = PostProcessing(my_model, requested_outputs=['prevXsusceptibleXamong', 'prevXinfectiousXamong'])
out = Outputs([pp])
out.plot_requested_outputs()

my_model.plot_compartment_size(["infectious"])
