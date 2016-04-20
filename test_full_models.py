

import os
import glob
import datetime
from pprint import pprint
import autumn.model
import numpy
import autumn.plotting
from autumn.spreadsheet import read_input_data_xls

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

start_realtime = datetime.datetime.now()

out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

models_to_run = [[0,  # Organ statuses
                  0,  # Strains
                  0,  # Comorbidities
                  False,  # Add low quality care
                  False,   # Add amplification
                  False]]  # Add misassignment

for running_model in range(len(models_to_run)):
    name = 'model%d' % running_model
    base = os.path.join(out_dir, name)
    model = autumn.model.ConsolidatedModel(
        models_to_run[running_model][0],
        models_to_run[running_model][1],
        models_to_run[running_model][2],
        models_to_run[running_model][3],
        models_to_run[running_model][4],
        models_to_run[running_model][5])
    # print((str(models_to_run[running_model][0]) + " organ(s), " +
    #        str(models_to_run[running_model][1]) + " strain(s), " +
    #        str(models_to_run[running_model][2]) + " comorbidity(ies)"))

    import_params = read_input_data_xls('xls/data_input.xlsx')  #\Github\AuTuMN\xls
    initials = import_params['const']['initials_for_compartments']
    parameters = import_params['const']['model_parameters']
    vac_cover = import_params['costcov']['Cost and coverage']['Vaccination']

    for key, value in parameters.items():
        model.set_parameter(key, value["Best"])
        print(key, value, model.params[key])

    for key, value in initials.items():
        model.set_compartment(key, value["Best"])
        print(key, value)

    start_time = 1850.
    recent_time = 1990.
    model.make_times(start_time, 2015.1, 0.1)
    model.integrate_explicit()
    model.make_graph(base + '.workflow')
    # INDIVIDUAL COMPARTMENTS
    # autumn.plotting.plot_fractions(
    #     model, model.labels, model.compartment_soln, recent_time,
    #     "strain", base + '.individ_total_bystrain.png')
    # autumn.plotting.plot_fractions(
    #     model, model.labels, model.compartment_soln, recent_time,
    #     "organ", base + '.individ_total_byorgan.png')
    # autumn.plotting.plot_fractions(
    #     model, model.labels, model.fraction_soln, recent_time,
    #     "strain", base + '.individ_fraction_bystrain.png')
    # autumn.plotting.plot_fractions(
    #     model, model.labels, model.fraction_soln, recent_time,
    #     "organ", base + '.individ_fraction_byorgan.png')
    # COMPARTMENT TYPES
    # autumn.plotting.plot_fractions(
    #     model, model.compartment_types, model.compartment_type_fraction_soln, recent_time,
    #     "", base + '.types_fraction.png')
    # autumn.plotting.plot_fractions(
    #     model, model.compartment_types, model.compartment_type_soln, recent_time,
    #     "", base + '.types_total.png')
    # autumn.plotting.plot_fractions(
    #     model, model.compartment_types_bystrain, model.compartment_type_bystrain_fraction_soln, recent_time,
    #     "strain", base + '.types_fraction_bystrain.png')
    # BROAD COMPARTMENT TYPES
    # autumn.plotting.plot_fractions(
    #     model, model.broad_compartment_types, model.broad_fraction_soln, recent_time,
    #     "", base + '.broad_fraction.png')
    # autumn.plotting.plot_fractions(
    #     model, model.broad_compartment_types_bystrain, model.broad_compartment_type_bystrain_fraction_soln, recent_time,
    #     "strain", base + '.broadtypes_fraction_bystrain.png')
    # autumn.plotting.plot_fractions(
    #     model, model.broad_compartment_types_byorgan, model.broad_compartment_type_byorgan_fraction_soln, recent_time,
    #     "organ", base + '.broadtypes_fraction_byorgan.png')
    # SUBGROUPS
    # autumn.plotting.plot_fractions(
    #     model, model.groups["treatment"], model.treatment_fraction_soln, recent_time,
    #     "", base + '.treatment_fraction.png')
    # autumn.plotting.plot_fractions(
    #     model, model.groups["identified"], model.identified_fraction_soln, recent_time,
    #     "", base + '.identified.png')
    # OUTPUT RATES
    # autumn.plotting.plot_outputs(
    #     model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #             "notifications_ds", "notifications_mdr"],
    #     start_time, base + '.rate_bystrain_outputs.png')
    autumn.plotting.plot_outputs(
        model, ["incidence", "mortality", "prevalence", "notifications"],
        start_time, base + '.rate_outputs.png')

    # autumn.plotting.plot_outputs(
    #     model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #             "notifications_ds", "notifications_mdr"],
    #     recent_time, base + '.rate_bystrain_outputs_recent.png')
    # autumn.plotting.plot_outputs(
    #     model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #             "notifications_ds", "notifications_mdr"],
    #     start_time, base + '.rate_outputs.png')
    # autumn.plotting.plot_outputs(
    #     model, ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
    #     start_time, base + '.mdr_outputs.png')
    # autumn.plotting.plot_outputs(
    #     model, ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
    #     recent_time, base + '.mdr_outputs_recent.png')
    #
    # autumn.plotting.plot_outputs(
    #     model, ["proportion_mdr"],
    #     start_time, base + '.mdr_proportion_recent.png')

    # Comment out the following 11 lines if running FlexibleModel, as it doesn't have the scale up stuff in it
    # autumn.plotting.plot_scaleup_fns(model,
    #                                  ["program_rate_default_infect_noamplify_ds",
    #                                   "program_rate_default_infect_amplify_ds",
    #                                   "program_rate_default_noninfect_noamplify_ds",
    #                                   "program_rate_default_noninfect_amplify_ds"],
    #                                  base + '.scaleup_amplification.png')
    # autumn.plotting.plot_scaleup_fns(model,
    #                                  ["program_rate_detect_ds_asds",
    #                                   "program_rate_detect_mdr_asmdr",
    #                                   "program_rate_detect_mdr_asds"],
    #                                  base + '.scaleup_detection.png')

    # year = indices(model.times, lambda x: x >= 2015.)[0]
    # print("2015 incidence is: ")
    # print(model.get_var_soln("incidence")[year])
    # print("2015 prevalence is: ")
    # print(model.get_var_soln("prevalence")[year])
    # print("2015 proportion MDR-TB is: ")
    # print(model.get_var_soln("proportion_mdr")[year])
    # print("2015 mortality is: ")
    # print(model.get_var_soln("mortality")[year])


pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)

print("Time elapsed in running script is " + str(datetime.datetime.now() - start_realtime))


# def indices(a, func):
#     return [i for (i, val) in enumerate(a) if func(val)]
#
# inds = indices(model.times, lambda x: x > 2014.)
#
# print(inds[0])
