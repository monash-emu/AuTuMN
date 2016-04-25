

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

# Select the datasets you want to import from the spreadsheet module
import_data = read_input_data_xls(True, ['bcg', 'tb', 'input_data'])  # \Github\AuTuMN\autumn\xls

# To run all possible models
strata_to_run = [0, 2, 3]
true_false = [True, False]

# To select models
strata_to_run = [2]
true_false = [False]

# Note that it takes about one hour to run all of the possible model structures
for n_organs in strata_to_run:
    for n_strains in strata_to_run:
        for n_comorbidities in strata_to_run:
            for is_quality in true_false:
                for is_amplification in true_false:
                    for is_misassignment in true_false:
                        if (is_misassignment and not is_amplification)\
                                or (n_strains <= 1 and (is_amplification or is_misassignment)):
                            pass
                        else:
                            name = 'model%d' % n_organs
                            base = os.path.join(out_dir, name)
                            model = autumn.model.ConsolidatedModel(
                                n_organs,
                                n_strains,
                                n_comorbidities,
                                is_quality,  # Low quality care
                                is_amplification,  # Amplification
                                is_misassignment)  # Misassignment by strain
                            print(str(n_organs) + " organ(s),   " +
                                  str(n_strains) + " strain(s),   " +
                                  str(n_comorbidities) + " comorbidity(ies),   " +
                                  "Low quality? " + str(is_quality) + ",   " +
                                  "Amplification? " + str(is_amplification) + ",   " +
                                  "Misassignment? " + str(is_misassignment) + ".")

                            parameters = import_data['const']['model_parameters']

                            for key, value in parameters.items():
                                model.set_parameter(key, value["Best"])

                            start_time = 1850.
                            recent_time = 1990.
                            model.make_times(start_time, 2015.1, 0.1)
                            model.integrate_explicit()
                            if n_organs + n_strains + n_comorbidities <= 2:
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
                            # if n_strains >= 2:
                            #     autumn.plotting.plot_outputs(
                            #         model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
                            #                 "notifications_ds", "notifications_mdr"],
                            #         start_time, base + '.rate_bystrain_outputs.png')
                            # autumn.plotting.plot_outputs(
                            #     model, ["incidence", "mortality", "prevalence", "notifications"],
                            #     start_time, base + '.rate_outputs.png')
                            # if n_strains >= 2:
                            #     autumn.plotting.plot_outputs(
                            #         model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
                            #                 "notifications_ds", "notifications_mdr"],
                            #         recent_time, base + '.rate_bystrain_outputs_recent.png')
                            #     autumn.plotting.plot_outputs(
                            #         model, ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
                            #                 "notifications_ds", "notifications_mdr"],
                            #         start_time, base + '.rate_outputs.png')
                            #     autumn.plotting.plot_outputs(
                            #         model, ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
                            #         start_time, base + '.mdr_outputs.png')
                            #     autumn.plotting.plot_outputs(
                            #         model, ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
                            #         recent_time, base + '.mdr_outputs_recent.png')
                            #     autumn.plotting.plot_outputs(
                            #         model, ["proportion_mdr"],
                            #         start_time, base + '.mdr_proportion_recent.png')
                            #
                            #     autumn.plotting.plot_scaleup_fns(model,
                            #                                      ["program_rate_default_infect_noamplify_ds",
                            #                                       "program_rate_default_infect_amplify_ds",
                            #                                       "program_rate_default_noninfect_noamplify_ds",
                            #                                       "program_rate_default_noninfect_amplify_ds"],
                            #                                      base + '.scaleup_amplification.png')
                            #     year = indices(model.times, lambda x: x >= 2015.)[0]
                            #     print("2015 incidence is: ")
                            #     print(model.get_var_soln("incidence")[year])
                            #     print("2015 prevalence is: ")
                            #     print(model.get_var_soln("prevalence")[year])
                            #     print("2015 proportion MDR-TB is: ")
                            #     print(model.get_var_soln("proportion_mdr")[year])
                            #     print("2015 mortality is: ")
                            #     print(model.get_var_soln("mortality")[year])

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)

print("Time elapsed in running script is " + str(datetime.datetime.now() - start_realtime))


# def indices(a, func):
#     return [i for (i, val) in enumerate(a) if func(val)]
#
# inds = indices(model.times, lambda x: x > 2014.)
#
# print(inds[0])
