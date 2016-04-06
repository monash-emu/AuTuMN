

import os
import glob
import datetime

import autumn.model
import autumn.plotting

start_realtime = datetime.datetime.now()

out_dir = 'flexible_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

models_to_run = [[3, 2, 1]]

for running_model in range(len(models_to_run)):
    name = 'model%d' % running_model
    base = os.path.join(out_dir, name)
    model = autumn.model.StratifiedModel()
    print()
    start_time = 1000.
    recent_time = 1990.
    model.make_times(start_time, 2015., 0.05)
    model.integrate_explicit()
    model.make_graph(base + '.workflow')
    # INDIVIDUAL COMPARTMENTS
    autumn.plotting.plot_fractions(
        model, model.labels, model.compartment_soln, recent_time,
        "strain", base + '.individ_total_bystrain.png')
    autumn.plotting.plot_fractions(
        model, model.labels, model.compartment_soln, recent_time,
        "organ", base + '.individ_total_byorgan.png')
    autumn.plotting.plot_fractions(
        model, model.labels, model.fraction_soln, recent_time,
        "strain", base + '.individ_fraction_bystrain.png')
    autumn.plotting.plot_fractions(
        model, model.labels, model.fraction_soln, recent_time,
        "organ", base + '.individ_fraction_byorgan.png')
    # COMPARTMENT TYPES
    autumn.plotting.plot_fractions(
        model, model.compartment_types, model.compartment_type_fraction_soln, recent_time,
        "", base + '.types_fraction.png')
    autumn.plotting.plot_fractions(
        model, model.compartment_types, model.compartment_type_soln, recent_time,
        "", base + '.types_total.png')
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
    #     start_time, base + '.rate_outputs.png')
    # autumn.plotting.plot_outputs(
    #     model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #             "notifications_ds", "notifications_mdr"],
    #     recent_time, base + '.rate_outputs_recent.png')
    # Comment out the following 11 lines if running FlexibleModel, as it doesn't have the scale up stuff in it
    # autumn.plotting.plot_scaleup_fns(model,
    #                                  ["program_rate_default_infect_noamplify_ds",
    #                                   "program_rate_default_infect_amplify_ds",
    #                                   "program_rate_default_noninfect_noamplify_ds",
    #                                   "program_rate_default_noninfect_amplify_ds"],
    #                                  base + '.scaleup_amplification.png')
    # autumn.plotting.plot_scaleup_fns(model,
    #                                  ["program_rate_detect",
    #                                   "program_rate_missed"],
    #                                  base + '.scaleup_detection.png')
#
pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)

print("Time elapsed in running script is " + str(datetime.datetime.now() - start_realtime))
