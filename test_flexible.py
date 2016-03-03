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
    model.make_times(1850, 2050, 0.05)
    model.integrate_explicit()
    model.make_graph(base + '.workflow')
    autumn.plotting.plot_fractions(
        model, model.labels, model.fraction_soln, base + '.fraction.png')
    autumn.plotting.plot_fractions(
        model, model.compartment_list, model.summed_fraction_soln, base + '.summed_fraction.png')
    autumn.plotting.plot_fractions(
        model, model.broad_compartment_types, model.broad_compartments_summed, base + '.population.png')

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)
