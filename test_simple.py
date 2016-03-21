import os
import glob

import autumn.model
import autumn.plotting

out_dir = 'simplified_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

base = os.path.join(out_dir, 'model')
model = autumn.model.SimplifiedModel()
model.make_times(1900, 2050, 0.05)
model.integrate_explicit()
model.make_graph(base + '.workflow')
autumn.plotting.plot_fractions(
    model, model.labels, model.fraction_soln, base + '.fraction.png', '')
autumn.plotting.plot_populations(
    model, model.labels, model.compartment_soln, base + '.population.png', '')

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)
