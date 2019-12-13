from summer_py.summer_model import *
from summer_py.post_processing import *
from summer_py.outputs import *
import os
from autumn_from_summer.tb_model import create_multi_scenario_outputs

# A test trial of creating a model that has the same compartments as the one in
# "Data needs for evidence-based decisions: a tuberculosis modeler’s ‘wish list’

# Time steps are given in years
start_time = 2000.
end_time = 2100.
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
            # {"type": "infection_density", "parameter": "reinfection_from_late_latent", "origin": "late_latent", "to": "early_latent"},
            # {"type": "standard_flows", "parameter": "relapse_rate", "origin": "recovered", "to": "active_tb"},
            {"type":"compartment_death", "parameter": "tb_mortality_rate", "origin":"active_tb"}]

my_parameters = {'infection_rate': .00012,
                 'immune_stabilisation_rate': 5.4e-3 * 365.25, #0.6,
                 'reactivation_rate': 3.3e-6 * 365.25,
                 # 'reinfection_from_late_latent': 0.0021,
                 'reinfection_from_recovered': 0.022,
                 'recovery_rate': 0.5,
                 'rapid_progression_rate': 2.7e-4 * 365.25,
                 # 'relapse_rate': 0.002,
                 "tb_mortality_rate": 0.097, # Global tuberculosis report 2018 estimated Viet Nam had 124 000 new TB cases and 12 000 TB-relate deaths
                 "universal_death_rate": 0.00639, #1./50.,
                 "crude_birth_rate": 0.0169
                 }

my_initial_conditions = {"active_tb": 1}

my_model = StratifiedModel(times=my_times, compartment_types=my_compartments, initial_conditions=my_initial_conditions,
                           parameters=my_parameters, requested_flows=my_flows, starting_population=100000,
                           infectious_compartment=('active_tb',), entry_compartment='susceptible',
                           birth_approach = "add_crude_birth_rate")

my_model.death_flows.to_csv("deaths_flows.csv")

# Verbose prints out information, does not effect model
# Specify arguments, need to check argument inputs order for my_model.stratify!!!
# default for parameter_adjustment is to give a relative parameter, e.g. original parameter is x,
# "1":0.5, means new parameter for age 1 is 0.5x

# Choose what to stratify the model by
stratify_by = ["age", "bcg", "novel"]

if "age" in stratify_by:
    # Stratify model by age

    infection_rate_adjustment = {"0": 1.0, "5": 1.0, "10": 1.0}
    immune_stabilisation_adjustment = {"0W": 1.2e-2 * 365.25, "5W": 1.2e-2 * 365.25, "10W": 1.2e-2 * 365.25}
    reactivation_rate_adjustment = {"0W": 1.9e-11 * 365.25, "5W": 6.4e-6 * 365.25, "10W": 6.4e-6 * 365.25}
    rapid_progression_rate_adjustment = {"0W": 6.6e-3 * 365.25, "5W": 2.7e-3 * 365.25, "10W": 2.7e-3 * 365.25}

    # immune_stabilisation_adjustment from Romain's epidemic paper, keppa
    # reactivation_rate_adjustment from Romain's epidemic paper, v
    # rapid_progression_rate_adjustment from Romain's epidemic paper, epsilon

    age_mixing = None  # None means homogenous mixing
    my_model.stratify("age", [0, 5, 10, 15, 60], [], {}, {}, infectiousness_adjustments={"0": 0, "5": 0, "10": 0},
                      mixing_matrix=age_mixing, verbose=False,
                      adjustment_requests={'immune_stabilisation_rate': immune_stabilisation_adjustment,
                                           'reactivation_rate': reactivation_rate_adjustment,
                                           'rapid_progression_rate': rapid_progression_rate_adjustment,
                                           'infection_rate': infection_rate_adjustment})

if "bcg" in stratify_by:
    # Stratify model by bcg vaccination status

    proportion_bcg = {"bcg_none": 0.05, "bcg_vaccinated": 0.95}
    my_model.stratify("bcg", ["bcg_none", "bcg_vaccinated"], ["susceptible"], requested_proportions=proportion_bcg,
                      entry_proportions={"bcg_none": 0.05, "bcg_vaccinated": 0.95},
                      mixing_matrix=None, verbose=False,
                      adjustment_requests={'infection_rateXage_0': {"bcg_vaccinated": 0.2},
                                           'infection_rateXage_5': {"bcg_vaccinated": 0.2},
                                           'infection_rateXage_10': {"bcg_vaccinated": 0.5},
                                           'infection_rateXage_15': {"bcg_vaccinated": 0.5},
                                           'infection_rateXage_60': {"bcg_vaccinated": 1.0}})

    if "novel" in stratify_by:
        # Stratify model by novel vaccination status

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


# Model outputs
create_flowchart(my_model)
print(os.getcwd())

my_model.run_model()

# print(my_model.outputs)

multiplier = {'prevXactive_tbXamong': 100000, 'prevXearly_latentXamong': 100000, 'prevXlate_latentXamong': 100000,
              'prevXsusceptibleXamong': 100000, 'prevXrecoveredXamong': 100000}
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