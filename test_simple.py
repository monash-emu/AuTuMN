import os
import glob

import autumn.base
import autumn.model
import autumn.outputs

out_dir = 'simplified_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

base = os.path.join(out_dir, 'model')

model = autumn.base.SimpleModel()
model.make_times(1600, 2050, 0.05)
model.integrate_explicit()
model.make_flow_diagram(base + '.workflow')

autumn.outputs.plot_fractions(
    model,
    model.labels,
    model.fraction_soln,
    1800,
    '',
    base + '.fraction.png')

autumn.outputs.plot_populations(
    model, model.labels, model.compartment_soln, 1800, '', base + '.population.png')

autumn.outputs.plot_outputs(
    model, ["incidence", "prevalence", "notifications", "mortality"],
    1800, base + '.rate_outputs.png')

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.outputs.open_pngs(pngs)
