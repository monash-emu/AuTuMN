

import os
import glob
import datetime
from pprint import pprint
import autumn.model
import numpy
import autumn.plotting
from autumn.spreadsheet import read_input_data_xls, get_country_data, calculate_proportion

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

start_realtime = datetime.datetime.now()

out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Select the datasets and country you want to import from the spreadsheet module
fields = ['bcg', 'birth_rate', 'life_expectancy']
gtb_fields = [u'c_cdr', u'c_new_tsr', u'e_inc_100k', u'e_inc_100k_lo', u'e_inc_100k_hi',
              u'e_prev_100k', u'e_prev_100k_lo', u'e_prev_100k_hi',
              u'e_mort_exc_tbhiv_100k', u'e_mort_exc_tbhiv_100k_lo', u'e_mort_exc_tbhiv_100k_hi']
notification_fields = [u'new_sp', u'new_sn', u'new_ep']

country = u'Fiji'
import_data = read_input_data_xls(True, fields + [
    'input_data', 'tb', 'outcomes', 'notifications', 'parameters', 'miscellaneous'])  # \Github\AuTuMN\autumn\xls

fixed_parameters = import_data['params']
miscellaneous_parameters = import_data['miscellaneous']

# You can now list as many fields as you like from the Global TB Report
country_data = {}
for field in fields + gtb_fields:
    country_data[field] = get_country_data('tb', import_data, field, country)
for field in notification_fields:
    country_data[field] = get_country_data('notifications', import_data, field, country)

# Calculate proportions that are smear-positive, smear-negative or extra-pulmonary
organs = [u'new_sp', u'new_sn', u'new_ep']
for organ in organs:
    country_data[u'prop_' + organ] = \
        calculate_proportion(country_data, organ, organs)

# To run all possible models
strata_to_run = [0, 2, 3]
true_false = [True, False]

# To run a single simple example
strata_to_run = [0]
true_false = [False]

# Note that it takes about one hour to run all of the possible model structures
for n_comorbidities in strata_to_run:
    for n_strains in strata_to_run:
        for n_organs in strata_to_run:
            for is_quality in true_false:
                for is_amplification in true_false:
                    for is_misassignment in true_false:
                        if (is_misassignment and not is_amplification) \
                                or (n_strains <= 1 and (is_amplification or is_misassignment)):
                            pass
                        else:
                            name = 'model%d' % n_organs
                            base = os.path.join(out_dir, name)
                            start_time = 1850.
                            recent_time = 1990.

                            model = autumn.model.ConsolidatedModel(
                                [],  # List of breakpoints for age stratification (or empty list for no stratification)
                                n_organs,
                                n_strains,
                                n_comorbidities,
                                is_quality,  # Low quality care
                                is_amplification,  # Amplification
                                is_misassignment,  # Misassignment by strain
                                country_data,
                                start_time)
                            print(str(n_organs) + " organ(s),   " +
                                  str(n_strains) + " strain(s),   " +
                                  str(n_comorbidities) + " comorbidity(ies),   " +
                                  "Low quality? " + str(is_quality) + ",   " +
                                  "Amplification? " + str(is_amplification) + ",   " +
                                  "Misassignment? " + str(is_misassignment) + ".")

                            for key, value in fixed_parameters.items():
                                model.set_parameter(key, value)
                            for key, value in miscellaneous_parameters.items():
                                model.set_parameter(key, value)

                            model.make_times(start_time, 2015.1, 0.1)
                            model.integrate_explicit()
                            if n_organs + n_strains + n_comorbidities <= 5:
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
                            autumn.plotting.plot_outputs(
                                model, ["incidence", "mortality", "prevalence", "notifications"],
                                start_time, base + '.rate_outputs.png')
                            autumn.plotting.plot_outputs_against_gtb(
                                model, "incidence",
                                recent_time, base + '.rate_outputs_gtb.png',
                                country_data)
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
                            autumn.plotting.plot_scaleup_fns(model,
                                                             ["program_prop_algorithm_sensitivity",
                                                              "program_prop_detect",
                                                              "program_prop_vaccination",
                                                              "program_prop_lowquality",
                                                              "program_prop_firstline_dst",
                                                              "program_prop_secondline_dst",
                                                              "program_proportion_success",
                                                              "program_proportion_default",
                                                              "program_proportion_death",
                                                              "tb_proportion_amplification"],
                                                             base + '.scaleups.png', start_time)
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
