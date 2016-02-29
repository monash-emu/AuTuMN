import os
import glob

import autumn.model
import autumn.plotting

out_dir = 'simple_model_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

for name, Model in [
        ('simplified', autumn.model.SimplifiedModel),
        # ('single', autumn.model.SingleStrainModel),
        ('new', autumn.model.NewFullModel)
    ]:
    print 'running', name
    base = os.path.join(out_dir, name)
    model = Model()
    model.make_times(1900, 2050, 0.05)
    model.integrate_explicit()
    model.make_graph(base + '.workflow')
    autumn.plotting.plot_fractions(
        model, model.labels, base + '.fraction.png')
    autumn.plotting.plot_populations(
        model, model.labels, base + '.population.png')

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)