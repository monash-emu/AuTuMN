from summer_py.summer_model import *
from summer_py.post_processing import *
from summer_py.outputs import *
import os
from autumn_from_summer.tb_model import create_multi_scenario_outputs

# A test trial of creating a model that has the same compartments as the one in
# "Data needs for evidence-based decisions: a tuberculosis modeler’s ‘wish list’

# Time steps are given in years
start_time = 1935.
end_time = 2035.
time_step = 1
my_times = numpy.linspace(start_time, end_time, int((end_time-start_time)/time_step) + 1).tolist()

my_compartments = ["susceptible",
                   "early_latent",
                   "late_latent",
                   "active_tb",
                   "recovered"]

my_flows = [{"type": "infection_density", "parameter": "infection_rate", "origin": "susceptible", "to": "early_latent"},
            {"type": "standard_flows", "parameter": "rapid_progression_rate", "origin": "early_latent", "to": "active_tb"},
            {"type": "standard_flows", "parameter": "immune_stabilisation_rate", "origin": "early_latent", "to": "late_latent"},
            {"type": "standard_flows", "parameter": "reactivation_rate", "origin": "late_latent", "to": "active_tb"},
            {"type": "standard_flows", "parameter": "recovery_rate", "origin": "active_tb", "to": "recovered"},
            {"type": "infection_density", "parameter": "reinfection_from_recovered", "origin": "recovered", "to": "early_latent"},
            {"type": "infection_density", "parameter": "reinfection_from_late_latent", "origin": "late_latent", "to": "early_latent"}]
            # {"type": "standard_flows", "parameter": "relapse_rate", "origin": "recovered", "to": "active_tb"},
            # {"type":"compartment_death", "parameter": "tb_mortality_rate", "origin":"active_tb"},

# parameters are in years, except for infection rate. Amount of time a person stays in early latent before going to late latent is 1/immune_stabilisation_rate
# rapid_progression_rate needs to be greater than reactivation_rate
disease_duration = 3.
latent_duration = 3/12

my_parameters = {'infection_rate':  .00013,
                 'immune_stabilisation_rate': 0.01 * 365.25, #0.6,
                 'reactivation_rate': 5.5e-6 * 365.25,
                 'reinfection_from_late_latent': 0.0021,
                 'reinfection_from_recovered': 0.0002,
                 'recovery_rate': 0.2,
                 'rapid_progression_rate': 0.0011 * 365.25,
                 'relapse_rate': 0.002,
                 "tb_mortality_rate": 0.2,
                 "universal_death_rate": 1./50.
                 }

my_initial_conditions = {
                         # "early_latent": 0.,
                         # "late_latent": 0.,
                         "active_tb": 0.000001
                         # "recovered": 0.
                        }

my_model = StratifiedModel(times=my_times,
                           compartment_types=my_compartments,
                           initial_conditions=my_initial_conditions,
                           parameters=my_parameters,
                           requested_flows=my_flows,
                           starting_population=100005,
                           infectious_compartment=('active_tb',),
                           entry_compartment='susceptible',
                           birth_approach="replace_deaths")

my_model.death_flows.to_csv("deaths_flows.csv")

# Verbose prints out information, does not effect model
# Specify arguments, need to check argument inputs order for my_model.stratify!!!
# default for parameter_adjustment is to give a relative parameter, e.g. original parameter is x,
# "1":0.5, means new parameter for age 1 is 0.5x

# Choose what to stratify the model by
stratify_by = ["age"] #["age", "vaccine"]

if "age" in stratify_by:
    # Stratify model by age
    age_mixing = None  # None means homogenous mixing
    my_model.stratify("age", [5, 10], [], {}, {}, mixing_matrix=age_mixing, verbose=False)

if "vaccine" in stratify_by:
    # Stratify model by vaccination status
    # props_vaccine = {"none": 0.25, "bcg_only": 0.25, "bcg+novel": 0.25, "novel": 0.25}
    # my_model.stratify("vaccine", ["none", "bcg_only", "bcg+novel", "novel"], [], requested_proportions = props_vaccine, mixing_matrix = None, verbose = False)
    proportion_vaccine = {"none": 0.5, "vaccine": 0.5}
    my_model.stratify("vaccine", ["none", "vaccine"], [], requested_proportions = proportion_vaccine,
                      infectiousness_adjustments = {"none": 0.9, "vaccine": 0.4},
                      mixing_matrix = None, verbose = False)

# Model outputs
create_flowchart(my_model)
print(os.getcwd())

my_model.run_model()

# print(my_model.outputs)

multiplier = {'prevXactive_tbXamong': 100000}
pp = PostProcessing(my_model, requested_outputs=['prevXsusceptibleXamong', 'prevXearly_latentXamong',
                                                 'prevXlate_latentXamong','prevXactive_tbXamong',
                                                 'prevXrecoveredXamong'], multipliers=multiplier)
out = Outputs([pp])
out.plot_requested_outputs()

# my_model.plot_compartment_size(["susceptible","vaccine"])

# Output graphs test
# req_outputs = ["prevXsusceptibleXamong",
#                "prevXearly_latentXamong",
#                "prevXlate_latentXamong",
#                "prevXrecoveredXamong"]
#
# create_multi_scenario_outputs(models=my_model, req_outputs=req_outputs)

# output_connections = {"TB_deaths":{"origin":"active_tb"}}