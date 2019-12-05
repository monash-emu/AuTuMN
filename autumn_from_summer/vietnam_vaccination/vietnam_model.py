from summer_py.summer_model import *
from summer_py.post_processing import *
from summer_py.outputs import *
import os

# A test trial of creating a model that has the same compartments as the one "Data needs for evidence-based decisions: a tuberculosis modeler’s ‘wish list’
my_times = numpy.linspace(0., 100., 101).tolist()

my_compartments = ["susceptible", "early_latent", "late_latent", "active_TB", "recovered"]

my_flows = [{"type": "infection_density", "parameter": "infection_rate", "origin": "susceptible", "to": "early_latent"},
            {"type": "standard_flows", "parameter": "rapid_progression_rate", "origin": "early_latent", "to": "active_TB"},
            {"type": "standard_flows", "parameter": "immune_stabilisation_rate", "origin": "early_latent", "to": "late_latent"},
            {"type": "standard_flows", "parameter": "reactivation_rate", "origin": "late_latent", "to": "active_TB"},
            {"type": "standard_flows", "parameter": "recovery_rate", "origin": "active_TB", "to": "recovered"},
            {"type": "infection_density", "parameter": "reinfection_from_recovered", "origin": "recovered", "to": "early_latent"},
            {"type": "infection_density", "parameter": "reinfection_from_late_latent", "origin": "late_latent", "to": "early_latent"},
            {"type": "standard_flows", "parameter": "relapse_rate", "origin": "recovered", "to": "active_TB"}]

my_parameters = {'infection_rate': 0.01,
                 'immune_stabilisation_rate': 0.001,
                 'reactivation_rate': 0.0004,
                 'reinfection_from_late_latent': 0.00005,
                 'reinfection_from_recovered': 0.0002,
                 'recovery_rate': 0.1,
                 'rapid_progression_rate': 0.001,
                 'relapse_rate': 0.002}

my_initial_conditions = {"early_latent": 50., "late_latent": 200., "active_TB": 2., "recovered": 0.}

my_model = StratifiedModel(times=my_times, compartment_types=my_compartments, initial_conditions=my_initial_conditions,
                           parameters=my_parameters, requested_flows=my_flows, starting_population=10500,
                           infectious_compartment=('active_TB',), entry_compartment='susceptible')

age_mixing = None # None means homogenous mixing

# Verbose prints out information, does not effect model
# Specify arguments, need to check argument inputs order for my_model.stratify!!!
# default for parameter_adjustment is to give a relative parameter, e.g. original parameter is x, "1":0.5, means new parameter for age 1 is 0.5x
my_model.stratify("age", [0, 1, 10], [], {}, {"recovery_rate": {"1": 0.5, "10": 0.8}}, infectiousness_adjustments={"1": 0.8}, mixing_matrix=age_mixing, verbose=False)

create_flowchart(my_model)
print(os.getcwd())

my_model.run_model()

print(my_model.outputs)

pp = PostProcessing(my_model, requested_outputs=['prevXsusceptibleXamong', 'prevXactive_TBXamong'])
out = Outputs([pp])
out.plot_requested_outputs()

my_model.plot_compartment_size(["active_TB"])