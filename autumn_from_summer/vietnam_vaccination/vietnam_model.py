from summer_py.summer_model import *
from summer_py.post_processing import *
from summer_py.outputs import *
import os
from autumn_from_summer.tb_model import create_multi_scenario_outputs
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()

def get_total_popsize(model, time):
    return sum(model.compartment_values)

# A test trial of creating a model that has the same compartments as the one in
# "Data needs for evidence-based decisions: a tuberculosis modeler’s ‘wish list’

# Set up time frame and time steps
start_time = 2000.
end_time = 2100.
time_step = 1 # Time steps are given in years
my_times = numpy.linspace(start_time, end_time, int((end_time-start_time)/time_step) + 1).tolist()

# Set up compartments
my_compartments = ["susceptible",
                   "early_latent",
                   "late_latent",
                   "infectious",
                   "recovered"]

# Set up flows between compartments, births, and deaths
my_flows = [{"type": "infection_density", "parameter": "infection_rate", "origin": "susceptible", "to": "early_latent"},
            {"type": "standard_flows", "parameter": "rapid_progression_rate", "origin": "early_latent", "to": "infectious"},
            {"type": "standard_flows", "parameter": "immune_stabilisation_rate", "origin": "early_latent", "to": "late_latent"},
            {"type": "standard_flows", "parameter": "reactivation_rate", "origin": "late_latent", "to": "infectious"},
            {"type": "standard_flows", "parameter": "recovery_rate", "origin": "infectious", "to": "recovered"},
            {"type": "infection_density", "parameter": "reinfection_from_recovered", "origin": "recovered", "to": "early_latent"},
            # {"type": "infection_density", "parameter": "reinfection_from_late_latent", "origin": "late_latent", "to": "early_latent"},
            # {"type": "standard_flows", "parameter": "relapse_rate", "origin": "recovered", "to": "infectious"},
            {"type": "compartment_death", "parameter": "tb_mortality_rate", "origin": "infectious"}]

# Track incidence rates
out_connections = {
        "incidence_from_early_latent": {"origin": "early_latent", "to": "infectious"},
        "incidence_from_late_latent": {"origin": "late_latent", "to": "infectious"}}

# Set up parameters
my_parameters = {'infection_rate': .00013,
                 'immune_stabilisation_rate': 5.4e-3 * 365.25,
                 'reactivation_rate': 3.3e-6 * 365.25,
                 'reinfection_from_late_latent': 0.0021,
                 'reinfection_from_recovered': 0.00013,
                 'recovery_rate': 0.5,
                 'rapid_progression_rate': 2.7e-4 * 365.25,
                 # 'relapse_rate': 0.002,
                 "tb_mortality_rate": 0.097, # Global tuberculosis report 2018 estimated Viet Nam had 124 000 new TB cases and 12 000 TB-relate deaths
                 "universal_death_rate": 0.00639, #1./50.,
                 "crude_birth_rate": 0.0169
                 }

# If we want the recovered population is the same as the susceptible population in terms of risk of TB infection, uses
my_parameters["reinfection_from_recovered"] = my_parameters["infection_rate"]

# Set up initial condition as a single seed individual with active TB
my_initial_conditions = {"infectious": 1}

my_model = StratifiedModel(times=my_times,
                           compartment_types=my_compartments,
                           initial_conditions=my_initial_conditions,
                           parameters=my_parameters,
                           requested_flows=my_flows,
                           starting_population=100000,
                           infectious_compartment=('infectious',),
                           entry_compartment='susceptible',
                           birth_approach="add_crude_birth_rate",
                           output_connections=out_connections,
                           derived_output_functions={'population': get_total_popsize})

# Choose what to stratify the model by
stratify_by = ["age", "bcg"]

if "age" in stratify_by:
    # Stratify model by age

    # default for parameter_adjustment is to give a relative parameter, e.g. original parameter is x,
    # "1":0.5, means new parameter for age 1 is 0.5x

    infection_rate_adjustment = {"0": 1.0, "5": 1.0, "10": 1.0}
    immune_stabilisation_adjustment = {"0W": 1.2e-2 * 365.25, "5W": 1.2e-2 * 365.25, "10W": 1.2e-2 * 365.25}
    reactivation_rate_adjustment = {"0W": 1.9e-11 * 365.25, "5W": 6.4e-6 * 365.25, "10W": 6.4e-6 * 365.25}
    rapid_progression_rate_adjustment = {"0W": 6.6e-3 * 365.25, "5W": 2.7e-3 * 365.25, "10W": 2.7e-3 * 365.25}

    # immune_stabilisation_adjustment from Romain's epidemic paper, keppa
    # reactivation_rate_adjustment from Romain's epidemic paper, v
    # rapid_progression_rate_adjustment from Romain's epidemic paper, epsilon

    # Matrix of social mixing rates between age groups
    age_mixing_matrix = numpy.array(
                                   [[1.8, 0.3, 0.3, 0.8, 0.4],
                                    [0.3, 2.1, 2.1, 0.7, 0.6],
                                    [0.3, 2.1, 2.1, 0.7, 0.6],
                                    [0.8, 0.7, 0.7, 1.0, 1.15],
                                    [0.4, 0.6, 0.6, 1.15, 1.6]]
                                    )

    # array estimated from Figure 4 in the paper "Social Contact Patterns in Vietnam and Implications for
    # the Control of Infectious Diseases"

    age_mixing = age_mixing_matrix # None means homogenous mixing

    my_model.stratify("age", [0, 5, 10, 15, 60], [], {}, {},
                      infectiousness_adjustments={"0": 0, "5": 0, "10": 0},
                      mixing_matrix=age_mixing,
                      verbose=False,              # Verbose prints out information, does not effect model
                      adjustment_requests={'immune_stabilisation_rate': immune_stabilisation_adjustment,
                                           'reactivation_rate': reactivation_rate_adjustment,
                                           'rapid_progression_rate': rapid_progression_rate_adjustment,
                                           'infection_rate': infection_rate_adjustment}
                      )

if "bcg" in stratify_by:
    # Stratify model by BCG vaccination status

    proportion_bcg = {"bcg_none": 0.05, "bcg_vaccinated": 0.95}
    my_model.stratify("bcg", ["bcg_none", "bcg_vaccinated"], ["susceptible"],
                      requested_proportions=proportion_bcg,
                      entry_proportions={"bcg_none": 0.05, "bcg_vaccinated": 0.95},
                      mixing_matrix=None, verbose=False,
                      adjustment_requests={'infection_rateXage_0': {"bcg_vaccinated": 0.2},
                                           'infection_rateXage_5': {"bcg_vaccinated": 0.2},
                                           'infection_rateXage_10': {"bcg_vaccinated": 0.5},
                                           'infection_rateXage_15': {"bcg_vaccinated": 0.5},
                                           'infection_rateXage_60': {"bcg_vaccinated": 1.0}})

if "novel" in stratify_by:
    # Stratify model by novel vaccination status (parameters are currently arbitrary)

    proportion_novel = {"novel_none": 0.5, "novel_vaccinated": 0.5}
    my_model.stratify("novel", ["novel_none", "novel_vaccinated"], ["susceptible"],
                      requested_proportions=proportion_novel,
                      entry_proportions={"bcg_none": 0.5, "bcg_vaccinated": 0.5},
                      mixing_matrix=None, verbose=False,
                      adjustment_requests={'infection_rateXage_10': {"novel_vaccinated": 0.5},
                                           'infection_rateXage_15': {"novel_vaccinated": 0.5},
                                           'infection_rateXage_60': {"novel_vaccinated": 1.0}})


# Stratification example from Mongolia
    # _tb_model.stratify("organ", ["smearpos", "smearneg", "extrapul"], ["infectious"],
    #                    infectiousness_adjustments={"smearpos": 1., "smearneg": 0.25, "extrapul": 0.},
    #                    verbose=False, requested_proportions=props_smear,
    #                    adjustment_requests={'recovery': recovery_adjustments,
    #                                         'infect_death': mortality_adjustments,
    #                                         'case_detection': diagnostic_sensitivity,
    #                                         'early_progression': props_smear,
    #                                         'late_progression': props_smear
    #                                         },


# Creating plots of model outputs
my_model.transition_flows.to_csv("transition.csv")
my_model.death_flows.to_csv("deaths_flows.csv")

create_flowchart(my_model)
# print(os.getcwd())

my_model.run_model()

my_requested_outputs = [
                     'prevXsusceptibleXamong',
                     'prevXearly_latentXamong',
                     'prevXlate_latentXamong',
                     'prevXinfectiousXamong',
                     'prevXrecoveredXamong',
                     'prevXinfectiousXamongXage_60',
                     'prevXinfectiousXamongXage_15',
                     'prevXinfectiousXamongXage_10',
                     'prevXinfectiousXamongXage_5',
                     'prevXinfectiousXamongXage_0',
                     'prevXlate_latentXamongXage_60',
                     'prevXlate_latentXamongXage_15',
                     'prevXlate_latentXamongXage_10',
                     'prevXlate_latentXamongXage_5',
                     'prevXlate_latentXamongXage_0'
                    ]

my_multiplier = {}
for output in my_requested_outputs:
    # Adds a 100 000 multiplier to all prevalence outputs
    if "prev" in output:
        my_multiplier[output] = 100000

my_translations = {'prevXsusceptibleXamong': "Susceptible prevalence (/100 000)",
                'prevXearly_latentXamong':"Prevalence of early latent TB (/100 000)",
                'prevXlate_latentXamong':"Prevalence of late latent TB (/100 000)",
                'prevXinfectiousXamong': "Prevalence of active TB (/100 000)",
                'prevXrecoveredXamong': "Prevalence of recovered (/100 000)",
                'prevXinfectiousXamongXage_60': "Prevalence of active TB among 60+ year olds (/100 000)",
                'prevXinfectiousXamongXage_15': "Prevalence of active TB among 15-60 year olds (/100 000)",
                'prevXinfectiousXamongXage_10':"Prevalence of active TB among 10-15 year olds (/100 000)",
                'prevXinfectiousXamongXage_5':"Prevalence of active TB among 5-10 year olds (/100 000)",
                'prevXinfectiousXamongXage_0':"Prevalence of active TB among 0-5 year olds (/100 000)",
                'prevXlate_latentXamongXage_60':"Prevalence of late latent TB among 60+ year olds (/100 000)",
                'prevXlate_latentXamongXage_15':"Prevalence of late latent TB among 15-60 year olds (/100 000)",
                'prevXlate_latentXamongXage_10': "Prevalence of late latent TB among 10-15 year olds (/100 000)",
                'prevXlate_latentXamongXage_5':"Prevalence of late latent TB among 5-10 year olds (/100 000)",
                'prevXlate_latentXamongXage_0':"Prevalence of late latent TB among 0-5 year olds (/100 000)"
                }

# pp = PostProcessing(my_model, requested_outputs=my_requested_outputs, multipliers=my_multiplier)
# out = Outputs([pp])
# out.plot_requested_outputs()

models = [my_model]
create_multi_scenario_outputs(models,
                              req_outputs=my_requested_outputs,
                              req_multipliers=my_multiplier,
                              out_dir="outputs_test_17_12_19",
                              translation_dictionary=my_translations,
                              scenario_list=[0]
                              )

want = "" # "population size"
if "population size" in want:
    # Creates plot of total population size

    model_outputs = my_model.outputs
    total_pop = []
    for row in model_outputs:
        total = 0
        for column in row:
            total += column
        total_pop.append(total)

    # plt.style.use("fivethirtyeight")
    plt.plot(my_times, total_pop, label="Total population")
    plt.title("Total population")
    plt.xlabel("years")
    plt.ylabel("Population")
    plt.savefig("My_total_population.png")
    plt.show()