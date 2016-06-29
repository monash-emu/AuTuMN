

import os
import glob
import datetime
import autumn.model
import autumn.base_analyses
import autumn.plotting
from autumn.spreadsheet import read_and_process_data, read_input_data_xls


# Start timer
start_realtime = datetime.datetime.now()

# Import the data
country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']
data = read_and_process_data(True,
                             ['bcg', 'rate_birth', 'life_expectancy', 'control_panel',
                              'default_parameters',
                              'tb', 'notifications', 'outcomes',
                              'country_constants', 'default_constants',
                              'country_economics', 'default_economics',
                              'country_programs', 'default_programs'],
                             country)

# A few basic preliminaries
out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Note that it takes about one hour to run all of the possible model structures,
# so perhaps don't do that - and longer if running multiple scenarios
scenario = None
is_additional_diagnostics = data['model_constants']['is_additional_diagnostics'][0]
n_organs = data['model_constants']['n_organs'][0]
n_strains = data['model_constants']['n_strains'][0]
n_comorbidities = data['model_constants']['n_comorbidities'][0]
is_quality = data['model_constants']['is_lowquality'][0]
is_amplification = data['model_constants']['is_amplification'][0]
is_misassignment = data['model_constants']['is_misassignment'][0]
if (is_misassignment and not is_amplification) \
        or (n_strains <= 1 and (is_amplification or is_misassignment)):
    pass
else:
    base = os.path.join(out_dir, country + '_baseline')

    model = autumn.model.ConsolidatedModel(
        n_organs,
        n_strains,
        n_comorbidities,
        is_quality,  # Low quality care
        is_amplification,  # Amplification
        is_misassignment,  # Misassignment by strain
        is_additional_diagnostics,
        scenario,  # Scenario to run
        data)
    print(str(n_organs) + " organ(s),   " +
          str(n_strains) + " strain(s),   " +
          str(n_comorbidities) + " comorbidity(ies),   " +
          "Low quality? " + str(is_quality) + ",   " +
          "Amplification? " + str(is_amplification) + ",   " +
          "Misassignment? " + str(is_misassignment) + ".")

    model.integrate()

    # Only make a flow-diagram if the model isn't overly complex
    if n_organs + n_strains + n_comorbidities <= 5:
        model.make_graph(base + '.workflow')

    # Plot over subgroups
    subgroup_solns, subgroup_fractions = autumn.base_analyses.find_fractions(model)
    for i, category in enumerate(subgroup_fractions):
        autumn.plotting.plot_fractions(
            model, subgroup_fractions[category], model.data['model_constants']['recent_time'],
            'strain', base + 'fraction_' + category + '.png', figure_number=30+i)

    autumn.plotting.plot_outputs_against_gtb(
        model, ["incidence", "mortality", "prevalence", "notifications"],
        data['model_constants']['recent_time'],
        'current_time',
        base + '_rate_outputs_gtb.png',
        country,
        scenario=None,
        figure_number=1)

    autumn.plotting.plot_classified_scaleups(model, base)

    # if n_strains >= 2:
    #     autumn.plotting.plot_outputs(
    #         model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #                 "notifications_ds", "notifications_mdr"],
    #         data['model_constants']['recent_time'], base + '.rate_bystrain_outputs_recent.png')
    #     autumn.plotting.plot_outputs(
    #         model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #                 "notifications_ds", "notifications_mdr"],
    #         start_time, base + '.rate_outputs.png')
    #     autumn.plotting.plot_outputs(
    #         model, ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
    #         data['model_constants']['start_time'], base + '.mdr_outputs.png')
    #     autumn.plotting.plot_outputs(
    #         model, ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
    #         data['model_constants']['recent_time'], base + '.mdr_outputs_recent.png')
    #     autumn.plotting.plot_outputs(
    #         model, ["proportion_mdr"],
    #         data['model_constants']['start_time'], base + '.mdr_proportion_recent.png')

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)

print("Time elapsed in running script is " + str(datetime.datetime.now() - start_realtime))


