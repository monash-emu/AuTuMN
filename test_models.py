import os
import glob

import autumn.model
import autumn.plotting

out_dir = 'test_model_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

for name, Model in [
        ('simple', autumn.model.SimpleFeedbackModel),
        ('naive', autumn.model.NaiveSingleStrainModel),
        ('basic', autumn.model.SingleStrainTreatmentModel),
    ]:
    print 'running', name
    base = os.path.join(out_dir, name)
    model = Model()
    model.set_flows()
    model.make_graph(base + '.workflow')
    model.make_times(1900, 2050, 0.05)
    model.integrate_explicit()
    autumn.plotting.plot_fractions(
        model, model.labels, base + '.fraction.png')
    autumn.plotting.plot_populations(
        model, model.labels, base + '.population.png')

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)