

import os
import glob

import autumn.model
import autumn.plotting

out_dir = 'flexible_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

models_to_run = [[3, 1, 1]]

for running_model in range(len(models_to_run)):
    name = 'model%d' % running_model
    base = os.path.join(out_dir, name)
    model = autumn.model.FlexibleModel(models_to_run[running_model][0],
                                       models_to_run[running_model][1],
                                       models_to_run[running_model][2])
    print((str(models_to_run[running_model][0]) + " organ(s), " +
          str(models_to_run[running_model][1]) + " strain(s), " +
          str(models_to_run[running_model][2]) + " comorbidity(ies)"))
    start_time = 1000.
    recent_time = 1990.
    model.make_times(start_time, 2015., 0.05)
    model.integrate_explicit()
    model.make_graph(base + '.workflow')
    # autumn.plotting.plot_fractions(
    #     model, model.labels, model.fraction_soln, base + '.fraction.png')
    # autumn.plotting.plot_fractions(
    #     model, model.compartment_types, model.compartment_type_fraction_soln, base + '.type_fraction.png')
    autumn.plotting.plot_fractions(
        model, model.broad_compartment_types, model.broad_fraction_soln, recent_time, base + '.broad_fraction.png')
    # autumn.plotting.plot_fractions(
    #     model, model.groups["treatment"], model.treatment_fraction_soln, base + '.treatment.png')
    # autumn.plotting.plot_fractions(
    #     model, model.groups["identified"], model.identified_fraction_soln, base + '.identified.png')
    #
    # autumn.plotting.plot_populations(
    #     model, model.labels, model.compartment_soln, recent_time, base + '.population.png')
    autumn.plotting.plot_populations(
        model, model.compartment_types, model.compartment_type_soln, recent_time, base + '.summed_population.png')
    autumn.plotting.plot_populations(
        model, model.broad_compartment_types, model.broad_compartment_soln, recent_time, base + '.broad_population.png')

    autumn.plotting.plot_outputs(
        model, ["incidence", "mortality", "prevalence"], recent_time, base + '.recent_rate_outputs.png')
    autumn.plotting.plot_outputs(
        model, ["incidence", "mortality", "prevalence"], start_time, base + '.rate_outputs.png')

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)
