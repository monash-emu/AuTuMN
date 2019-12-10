from summer_py.summer_model import *
from summer_py.post_processing import *
from summer_py.outputs import *
import os
from autumn_from_summer.tb_model import create_multi_scenario_outputs

# A test trial of creating a model that has the same compartments as the one in
# "Data needs for evidence-based decisions: a tuberculosis modeler’s ‘wish list’

my_times = numpy.linspace(0., 100., 101).tolist()

my_compartments = ["susceptible", "early_latent", "late_latent", "active_TB", "recovered"]

my_flows = [{"type": "infection_density", "parameter": "infection_rate", "origin": "susceptible", "to": "early_latent"},
           #{"type": "standard_flows", "parameter": "rapid_progression_rate", "origin": "early_latent", "to": "active_TB"},
            {"type": "standard_flows", "parameter": "immune_stabilisation_rate", "origin": "early_latent", "to": "late_latent"},
            {"type": "standard_flows", "parameter": "reactivation_rate", "origin": "late_latent", "to": "active_TB"},
            {"type": "standard_flows", "parameter": "recovery_rate", "origin": "active_TB", "to": "recovered"},
           #{"type": "infection_density", "parameter": "reinfection_from_recovered", "origin": "recovered", "to": "early_latent"},
           #{"type": "infection_density", "parameter": "reinfection_from_late_latent", "origin": "late_latent", "to": "early_latent"},
           #{"type": "standard_flows", "parameter": "relapse_rate", "origin": "recovered", "to": "active_TB"}]
            {"type":"compartment_death", "parameter": "tb_mortality_rate", "origin":"active_TB"}]

my_parameters = {'infection_rate': 1.,
                 'immune_stabilisation_rate': 0.001,
                 'reactivation_rate': 0.0004,
                 'reinfection_from_late_latent': 0.00005,
                 'reinfection_from_recovered': 0.0002,
                 'recovery_rate': 0.231,
                 'rapid_progression_rate': 0.001,
                 'relapse_rate': 0.002,
                 "tb_mortality_rate": 0.389}

my_initial_conditions = {"early_latent": 50., "late_latent": 200., "active_TB": 200., "recovered": 0.}

my_model = StratifiedModel(times=my_times, compartment_types=my_compartments, initial_conditions=my_initial_conditions,
                           parameters=my_parameters, requested_flows=my_flows, starting_population=10500,
                           infectious_compartment=('active_TB',), entry_compartment='susceptible',
                           birth_approach = "replace_deaths")


# Verbose prints out information, does not effect model
# Specify arguments, need to check argument inputs order for my_model.stratify!!!
# default for parameter_adjustment is to give a relative parameter, e.g. original parameter is x,
# "1":0.5, means new parameter for age 1 is 0.5x

age_mixing = None  # None means homogenous mixing
my_model.stratify("age", [5, 10], [], {}, {}, mixing_matrix=age_mixing, verbose=False)

# props_vaccine = {"none": 0.25, "bcg_only": 0.25, "bcg+novel": 0.25, "novel": 0.25}
# my_model.stratify("vaccine", ["none", "bcg_only", "bcg+novel", "novel"], [], requested_proportions = props_vaccine,
#                   mixing_matrix = None, verbose = False)

proportion_vaccine = {"none": 0.5, "vaccine": 0.5}
my_model.stratify("vaccine", ["none", "vaccine"], [], requested_proportions = proportion_vaccine,
                  infectiousness_adjustments = {"none": 0.9, "vaccine": 0.4},
                  mixing_matrix = None, verbose = False)

# _tb_model.stratify("smear", ["smearpos", "smearneg", "extrapul"], ["infectious"],
#                    infectiousness_adjustments={"smearpos": 1., "smearneg": 0.25, "extrapul": 0.},
#                    verbose=False, requested_proportions=props_smear,
#                    entry_proportions=props_smear)

# stratification_name, strata_request, compartment_types_to_stratify, requested_proportions,
#             entry_proportions={}, adjustment_requests=(), infectiousness_adjustments={}, mixing_matrix=None,
#             verbose=True):


create_flowchart(my_model)
print(os.getcwd())

my_model.run_model()

print(my_model.outputs)

pp = PostProcessing(my_model, requested_outputs=['prevXsusceptibleXamong', 'prevXactive_TBXamong'])
out = Outputs([pp])
out.plot_requested_outputs()

my_model.plot_compartment_size(["susceptible", "vaccine"])

# Output graphs
# req_outputs = ["prevXsusceptibleXamong",
#                "prevXearly_latentXamong",
#                "prevXlate_latentXamong",
#                "prevXrecoveredXamong"]
#
# create_multi_scenario_outputs(models=my_model, req_outputs=req_outputs)