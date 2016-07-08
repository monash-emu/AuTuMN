

import os
import glob
import datetime
import autumn.model

from autumn.spreadsheet import read_and_process_data, read_input_data_xls
import economics

# Start timer
start_realtime = datetime.datetime.now()

# Import the data
country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']
inputs = read_and_process_data(country, from_test=True)

# A few basic preliminaries
out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Note that it takes about one hour to run all of the possible model structures,
# so perhaps don't do that - and longer if running multiple scenarios
scenario = None
n_organs = inputs['model_constants']['n_organs'][0]
n_strains = inputs['model_constants']['n_strains'][0]
is_quality = inputs['model_constants']['is_lowquality'][0]
is_amplification = inputs['model_constants']['is_amplification'][0]
is_misassignment = inputs['model_constants']['is_misassignment'][0]
if (is_misassignment and not is_amplification) \
        or (n_strains <= 1 and (is_amplification or is_misassignment)):
    pass
else:
    name = 'model%d' % n_organs
    base = os.path.join(out_dir, name)

    model = autumn.model.ConsolidatedModel(
        n_organs,
        n_strains,
        is_quality,  # Low quality care
        is_amplification,  # Amplification
        is_misassignment,  # Misassignment by strain
        scenario,  # Scenario to run
        inputs)
    print(str(n_organs) + " organ(s),   " +
          str(n_strains) + " strain(s),   " +
          "Low quality? " + str(is_quality) + ",   " +
          "Amplification? " + str(is_amplification) + ",   " +
          "Misassignment? " + str(is_misassignment) + ".")

    model.integrate()

    # Only make a flow-diagram if the model isn't overly complex
    if n_organs + n_strains <= 5:
        model.make_graph(base + '.workflow')

    # INDIVIDUAL COMPARTMENTS
    # autumn.plotting.plot_fractions(
    #     model, model.labels, model.compartment_soln, data['attributes']['recent_time'],
    #     "strain", base + '.individ_total_bystrain.png')
    # autumn.plotting.plot_fractions(
    #     model, model.labels, model.compartment_soln, data['attributes']['recent_time'],
    #     "organ", base + '.individ_total_byorgan.png')
    # autumn.plotting.plot_fractions(
    #     model, model.labels, model.fraction_soln, data['attributes']['recent_time'],
    #     "strain", base + '.individ_fraction_bystrain.png')
    # autumn.plotting.plot_fractions(
    #     model, model.labels, model.fraction_soln, data['attributes']['recent_time'],
    #     "organ", base + '.individ_fraction_byorgan.png')
    # COMPARTMENT TYPES
    # autumn.plotting.plot_fractions(
    #     model, model.compartment_types, model.compartment_type_fraction_soln, data['attributes']['recent_time'],
    #     "", base + '.types_fraction.png')
    # autumn.plotting.plot_fractions(
    #     model, model.compartment_types, model.compartment_type_soln, data['attributes']['recent_time'],
    #     "", base + '.types_total.png')
    # autumn.plotting.plot_fractions(
    #     model, model.compartment_types_bystrain, model.compartment_type_bystrain_fraction_soln, data['attributes']['recent_time'],
    #     "strain", base + '.types_fraction_bystrain.png')
    # BROAD COMPARTMENT TYPES
    # autumn.plotting.plot_fractions(
    #     model, model.broad_compartment_types, model.broad_fraction_soln, data['attributes']['recent_time'],
    #     "", base + '.broad_fraction.png')
    # autumn.plotting.plot_fractions(
    #     model, model.broad_compartment_types, model.broad_fraction_soln, data['attributes']['start_time'],
    #     "", base + '.broad_fraction.png')
    # autumn.plotting.plot_fractions(
    #     model, model.broad_compartment_types_bystrain, model.broad_compartment_type_bystrain_fraction_soln, data['attributes']['recent_time'],
    #     "strain", base + '.broadtypes_fraction_bystrain.png')
    # autumn.plotting.plot_fractions(
    #     model, model.broad_compartment_types_byorgan, model.broad_compartment_type_byorgan_fraction_soln, data['attributes']['recent_time'],
    #     "organ", base + '.broadtypes_fraction_byorgan.png')
    # SUBGROUPS
    # autumn.plotting.plot_fractions(
    #     model, model.groups["treatment"], model.treatment_fraction_soln, data['attributes']['recent_time'],
    #     "", base + '.treatment_fraction.png')
    # autumn.plotting.plot_fractions(
    #     model, model.groups["identified"], model.identified_fraction_soln, data['attributes']['recent_time'],
    #     "", base + '.identified.png')
    # OUTPUT RATES
    # if n_strains >= 2:
    #     autumn.plotting.plot_outputs(
    #         model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #                 "notifications_ds", "notifications_mdr"],
    #         data['attributes']['start_time'], base + '.rate_bystrain_outputs.png')




    # if n_strains >= 2:
    #     autumn.plotting.plot_outputs(
    #         model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #                 "notifications_ds", "notifications_mdr"],
    #         data['attributes']['recent_time'], base + '.rate_bystrain_outputs_recent.png')
    #     autumn.plotting.plot_outputs(
    #         model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #                 "notifications_ds", "notifications_mdr"],
    #         start_time, base + '.rate_outputs.png')
    #     autumn.plotting.plot_outputs(
    #         model, ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
    #         data['attributes']['start_time'], base + '.mdr_outputs.png')
    #     autumn.plotting.plot_outputs(
    #         model, ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
    #         data['attributes']['recent_time'], base + '.mdr_outputs_recent.png')
    #     autumn.plotting.plot_outputs(
    #         model, ["proportion_mdr"],
    #         data['attributes']['start_time'], base + '.mdr_proportion_recent.png')

    # Classify scale-up functions for plotting
    classified_scaleups = {'program_prop': [],
                           'program_other': [],
                           'birth': [],
                           'non_program': []}
    for fn in model.scaleup_fns:
        if 'program_prop' in fn:
            classified_scaleups['program_prop'] += [fn]
        elif 'program' in fn:
            classified_scaleups['program_other'] += [fn]
        elif 'demo_rate_birth' in fn:
            classified_scaleups['birth'] += [fn]
        else:
            classified_scaleups['non_program'] += [fn]

    # Plot them from the start of the model and from "recent_time"
    for i, classification in enumerate(classified_scaleups):
        if len(classified_scaleups[classification]) > 0:
            #print(classified_scaleups[classification])

                economics.cost_scaleup_fns(model,
                                            classified_scaleups[classification],
                                            'start_time',
                                            'scenario_end_time',
                                            classification,
                                            country)






