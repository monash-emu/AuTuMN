import os
import glob

import autumn.model
import autumn.plotting

out_dir = 'simple_model_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

models_to_run = [[3, 1, 1, "three organs, one strain"],
                 # [2, 2, 1, "two organs, two strains"],
                 [1, 1, 1, "one organ, one strain"]]

for running_model in range(len(models_to_run)):
    name = 'flexible_model'
    base = os.path.join(out_dir, name)
    model = autumn.model.FlexibleModel(models_to_run[running_model][0],
                                       models_to_run[running_model][1],
                                       models_to_run[running_model][2])
    print(models_to_run[running_model][3])
    model.make_times(1900, 2050, 0.05)
    model.integrate_explicit()
    model.make_graph(base + '.workflow')
    autumn.plotting.plot_fractions(
        model, model.labels, base + '.fraction.png')
    autumn.plotting.plot_populations(
        model, model.labels, base + '.population.png')

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)
