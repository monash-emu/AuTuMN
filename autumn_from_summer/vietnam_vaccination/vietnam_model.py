from summer_py.summer_model import *
from summer_py.post_processing import *
from summer_py.outputs import *
import os
from autumn_from_summer.tb_model import create_multi_scenario_outputs
import matplotlib.pyplot as plt


def get_total_popsize(model, time):
    return sum(model.compartment_values)


# Set up time frame and time steps in years
start_time = 2000.
end_time = 2500.
time_step = 1
my_times = numpy.linspace(start_time, end_time, int((end_time - start_time) / time_step) + 1).tolist()

# Set up compartments
my_compartments = ["susceptible",
                   "early_latent",
                   "late_latent",
                   "infectious",
                   "recovered"]

# Set up transitions between compartments, births, and deaths
my_flows = [
    {"type": "infection_frequency", "parameter": "infection_rate", "origin": "susceptible", "to": "early_latent"},
    {"type": "standard_flows", "parameter": "rapid_progression_rate", "origin": "early_latent", "to": "infectious"},
    {"type": "standard_flows", "parameter": "immune_stabilisation_rate", "origin": "early_latent", "to": "late_latent"},
    {"type": "standard_flows", "parameter": "reactivation_rate", "origin": "late_latent", "to": "infectious"},
    {"type": "standard_flows", "parameter": "self_recovery_rate", "origin": "infectious", "to": "recovered"},
    # The next few transitions have been commented out because the parameter values are currently unknown
    # {"type": "infection_frequency", "parameter": "reinfection_from_recovered", "origin": "recovered", "to": "early_latent"},
    # {"type": "infection_frequency", "parameter": "reinfection_from_late_latent", "origin": "late_latent", "to": "early_latent"},
    {"type": "standard_flows", "parameter": "relapse_rate", "origin": "recovered", "to": "infectious"},
    {"type": "compartment_death", "parameter": "tb_mortality_rate", "origin": "infectious"},
]

# Track incidence rates
out_connections = {
    "incidence_from_susceptible_to_early_latent": {"origin": "susceptible", "to": "early_latent"},
    "incidence_from_early_latent_to_infectious": {"origin": "early_latent", "to": "infectious"},
    "incidence_from_late_latent_to_infectious": {"origin": "late_latent", "to": "infectious"}}

# Baseline parameter values
my_parameters = {
    'infection_rate': 50,  # further calibration required
    'immune_stabilisation_rate': 5.4e-3 * 365.25,
    'reactivation_rate': 3.3e-6 * 365.25,
    # 'reinfection_from_late_latent': 0.09,  # currently unknown
    # 'reinfection_from_recovered': 0.15,    # currently unknown
    'self_recovery_rate': 0.223,
    'rapid_progression_rate': 2.7e-4 * 365.25,
    'relapse_rate': 0.0139,
    "tb_mortality_rate": 0.11,
    "universal_death_rate": 0.005803,
    "crude_birth_rate": 0.0169,
    "case_detection_rate": 0.57,
    # proportion of cases notified to WHO divided by number of estimated cases for that year
    "treatment_success_rate": 0.92
}
my_parameters['self_recovery_rate'] = 1/3-my_parameters['tb_mortality_rate']

# Add case detection rate (cdr), and hence different rates of recovery without treatment or with detection and treatment
# we assume everyone who gets detected received treatment
cdr = True  # Set to False if you do not want to have case detection
if cdr:
    case_detection = my_parameters['case_detection_rate']

    # Get parameters involved in flows from infectious to recovered
    self_recovery_rate = my_parameters['self_recovery_rate']
    tb_mortality_rate = my_parameters['tb_mortality_rate']
    universal_death_rate = my_parameters['universal_death_rate']

    # Calculate rate of notifications from cdr proportion   # We assume this equals the rate of treatment initiation
    treatment_rate = case_detection * (self_recovery_rate + tb_mortality_rate + universal_death_rate) / (
                1. - case_detection)
    treatment_success_rate = my_parameters['treatment_success_rate']

    # Add new treatment_recovery_rate parameter and recovery with treatment as a flow
    my_parameters['treatment_recovery_rate'] = treatment_rate * treatment_success_rate
    my_flows.append(
        {"type": "standard_flows", "parameter": "treatment_recovery_rate", "origin": "infectious", "to": "recovered"})

# Set up initial condition as a single seed individual with active TB
my_initial_conditions = {"infectious": 1}

my_model = StratifiedModel(
    times=my_times,
    compartment_types=my_compartments,
    initial_conditions=my_initial_conditions,
    parameters=my_parameters,
    requested_flows=my_flows,
    starting_population=1000000,
    infectious_compartment=('infectious',),
    entry_compartment='susceptible',
    birth_approach="add_crude_birth_rate",
    output_connections=out_connections,
    derived_output_functions={'population': get_total_popsize},
    death_output_categories=((), ("age_0",), ("age_5",))  # , ("bcg"))
)

# Choose what to stratify the model by
stratify_by = ["age", "bcg"]  # , "novel"]

if "age" in stratify_by:
    # Stratify model by age

    # default for parameter_adjustment is to give a relative parameter, e.g. original parameter is x,
    # "1":0.5, means new parameter for age 1 is 0.5x

    infection_rate_adjustment = {"0": 1.0, "5": 1.0, "10": 1.0}
    immune_stabilisation_adjustment = {"0W": 1.2e-2 * 365.25, "5W": 1.2e-2 * 365.25, "10W": 1.2e-2 * 365.25}
    reactivation_rate_adjustment = {"0W": 1.9e-11 * 365.25, "5W": 6.4e-6 * 365.25, "10W": 6.4e-6 * 365.25}
    rapid_progression_rate_adjustment = {"0W": 6.6e-3 * 365.25, "5W": 2.7e-3 * 365.25, "10W": 2.7e-3 * 365.25}

    # immune_stabilisation_adjustment from Romain's epidemic paper, Keppa
    # reactivation_rate_adjustment from Romain's epidemic paper, Nu
    # rapid_progression_rate_adjustment from Romain's epidemic paper, Epsilon

    # Matrix of social mixing rates between age groups
    age_mixing_matrix = numpy.array(
        [[1.875, 0.375, 0.375, 0.833, 0.625],
         [0.375, 2.125, 2.125, 0.792, 0.500],
         [0.375, 2.125, 2.125, 0.792, 0.500],
         [0.833, 0.792, 0.792, 0.896, 0.896],
         [0.625, 0.500, 0.500, 0.896, 1.750]]
    )

    # array estimated from Figure 4 in the paper "Social Contact Patterns in Vietnam and Implications for
    # the Control of Infectious Diseases"

    age_mixing = age_mixing_matrix  # None means homogenous mixing
    my_model.stratify("age", [0, 5, 10, 15, 50], [], {}, {},
                      infectiousness_adjustments={"0": 0, "5": 0, "10": 0},
                      mixing_matrix=age_mixing,
                      verbose=False,  # Verbose prints out information, does not effect model
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
                                           'infection_rateXage_10': {"bcg_vaccinated": 0.48},
                                           'infection_rateXage_15': {"bcg_vaccinated": 0.48},
                                           'infection_rateXage_50': {"bcg_vaccinated": 1.0}})

if "novel" in stratify_by:
    # Stratify model by novel vaccination status (parameters are currently arbitrary)

    proportion_novel = {"novel_none": 0.5, "novel_vaccinated": 0.5}
    my_model.stratify("novel", ["novel_none", "novel_vaccinated"], ["susceptible"],
                      requested_proportions=proportion_novel,
                      entry_proportions={"bcg_none": 0.5, "bcg_vaccinated": 0.5},
                      mixing_matrix=None, verbose=False,
                      adjustment_requests={'infection_rateXage_10': {"novel_vaccinated": 0.5},
                                           'infection_rateXage_15': {"novel_vaccinated": 0.5},
                                           'infection_rateXage_50': {"novel_vaccinated": 1.0}})

# Save flows including death to csv files
my_model.transition_flows.to_csv("transition.csv")
my_model.death_flows.to_csv("deaths_flows.csv")

# Creating flow chart and plots of model outputs
create_flowchart(my_model)
print(os.getcwd())

my_model.run_model()

# Specify level of detail for outputs
# 'summary' gives plots of prevalence of each compartment
# 'detailed' gives plots of distribution by stata, prevalence of each compartment for each stratification,
# e.g. prevalence of age 0-5 who are infectious
output_detail = 'summary'  # 'detailed'
if output_detail == 'summary':
    my_requested_outputs = [
        'prevXsusceptibleXamong',
        'prevXearly_latentXamong',
        'prevXlate_latentXamong',
        'prevXinfectiousXamong',
        'prevXrecoveredXamong',
        # 'prevXinfectiousXamongXage_50',
        # 'prevXinfectiousXamongXage_15',
        # 'prevXinfectiousXamongXage_10',
        # 'prevXinfectiousXamongXage_5',
        # 'prevXinfectiousXamongXage_0',
        # 'prevXlate_latentXamongXage_50',
        # 'prevXlate_latentXamongXage_15',
        # 'prevXlate_latentXamongXage_10',
        # 'prevXlate_latentXamongXage_5',
        # 'prevXlate_latentXamongXage_0',
        # "distribution_of_strataXage",
        # "distribution_of_strataXbcg",
        # "distribution_of_stataXnovel"
        # # "prevXsusceptibleBYage",
        # # "prevXearly_latentBYage",
        # # "prevXlate_latentBYage",
        # # "prevXrecoveredBYage"
    ]

    my_multiplier = {}
    for output in my_requested_outputs:
        # Adds a 100 000 multiplier to all prevalence outputs
        if "prev" in output:
            my_multiplier[output] = 100000

    my_translations = {
        'prevXsusceptibleXamong': "Susceptible prevalence (/100 000)",
        'prevXearly_latentXamong': "Prevalence of early latent TB (/100 000)",
        'prevXlate_latentXamong': "Prevalence of late latent TB (/100 000)",
        'prevXinfectiousXamong': "Prevalence of active TB (/100 000)",
        'prevXrecoveredXamong': "Prevalence of recovered (/100 000)",
        # 'prevXinfectiousXamongXage_50': "Prevalence of active TB among 60+ year olds (/100 000)",
        # 'prevXinfectiousXamongXage_15': "Prevalence of active TB among 15-60 year olds (/100 000)",
        # 'prevXinfectiousXamongXage_10':"Prevalence of active TB among 10-15 year olds (/100 000)",
        # 'prevXinfectiousXamongXage_5':"Prevalence of active TB among 5-10 year olds (/100 000)",
        # 'prevXinfectiousXamongXage_0':"Prevalence of active TB among 0-5 year olds (/100 000)",
        # 'prevXlate_latentXamongXage_50':"Prevalence of late latent TB among 60+ year olds (/100 000)",
        # 'prevXlate_latentXamongXage_15':"Prevalence of late latent TB among 15-60 year olds (/100 000)",
        # 'prevXlate_latentXamongXage_10': "Prevalence of late latent TB among 10-15 year olds (/100 000)",
        # 'prevXlate_latentXamongXage_5':"Prevalence of late latent TB among 5-10 year olds (/100 000)",
        # 'prevXlate_latentXamongXage_0':"Prevalence of late latent TB among 0-5 year olds (/100 000)",
    }

    pp = PostProcessing(my_model, requested_outputs=my_requested_outputs, multipliers=my_multiplier)
    out = Outputs([pp], out_dir="outputs_test_20_12_19", translation_dict=my_translations)
    out.plot_requested_outputs()

#####################################################################################################################
# 20/12/19 Unresolved issues with bcg and novel vaccine plots being empty when using the create_outputs function
# Functions is a copy of create_multi_scenario_outputs() with minor changes
#####################################################################################################################
def create_outputs(models, req_outputs, req_times={}, req_multipliers={}, out_dir='outputs_tes', targets_to_plot={},
                   translation_dictionary={}, scenario_list=[]):
    """
    process and generate plots for several scenarios
    :param models: a list of run models
    :param req_outputs. See PostProcessing class
    :param req_times. See PostProcessing class
    :param req_multipliers. See PostProcessing class
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    my_post_proccessing_list = []
    for scenario_index in range(len(models)):

        # automatically add some basic outputs
        if hasattr(models[scenario_index], "all_stratifications"):

            for group in models[scenario_index].all_stratifications.keys():
                # Add distribution of population within each type of stratifications
                req_outputs.append('distribution_of_strataX' + group)

                for stratum in models[scenario_index].all_stratifications[group]:
                    req_outputs.append('prevXinfectiousXamongX' + group + '_' + stratum)
                    req_outputs.append('prevXearly_latentXamongX' + group + '_' + stratum)
                    req_outputs.append('prevXlate_latentXamongX' + group + '_' + stratum)
                    req_outputs.append('prevXrecoveredXamongX' + group + '_' + stratum)

            if "bcg" in models[scenario_index].all_stratifications.keys():
                req_outputs.append('prevXinfectiousXbcg_noneXamongXinfectious')

        for output in req_outputs:
            if output[0:15] == 'prevXinfectious':
                req_multipliers[output] = 1.e5
                # translation_dictionary
            elif output[0:17] == 'prevXearly_latent' or output[0:16] == 'prevXlate_latent':
                req_multipliers[output] = 1.e2

        my_post_proccessing_list.append(post_proc.PostProcessing(
            models[scenario_index],
            requested_outputs=req_outputs,
            scenario_number=scenario_list[scenario_index],
            requested_times=req_times,
            multipliers=req_multipliers))

    outputs = Outputs(my_post_proccessing_list, targets_to_plot, out_dir, translation_dictionary)
    outputs.plot_requested_outputs()

    for req_output in ['prevXinfectious', 'prevXearly_latent', 'prevXlate_latent', 'prevXrecovered']:
        outputs.plot_outputs_by_stratum(req_output)


if output_detail == 'detailed':
    models = [my_model]
    create_outputs(models,
                   my_requested_outputs,
                   req_multipliers=my_multiplier,
                   out_dir="output_test_20_12_19",
                   translation_dictionary=my_translations,
                   scenario_list=[0]
                   )

# This is a practice for plotting using matplotlib
want = ""  # "population size"
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