# -*- coding: utf-8 -*-


"""
# Stage 18 model of the Dynamic Transmission Model

- latent early and latent late,
- fail detect
- two stage treatment
- three pulmonary status
"""


import os
import glob

import pylab


import autumn.model
import autumn.plotting as plotting

out_dir = 'fullmodel_graphs'

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

model = autumn.model.FullModel()
model.make_times_with_n_step(1950., 2015., 20)
model.integrate_explicit()

model.make_graph(os.path.join(out_dir, 'flow_chart.png'))

groups = {
    "ever_infected": ["suscptible_treated", "latent", "active", "missed", "detect", "treatment"],
    "infected": ["latent", "active", "missed", "detect", "treatment"],
    "active": ["active", "missed", "detect", "treatment"],
    "infectious": ["active", "missed", "detect", "treatment_infect"],
    "identified": ["detect", "treatment"],
    "treatment": ["treatment"]
}
for title, tags in groups.items():
    plotting.plot_population_group(model, title, tags)
    pylab.savefig(os.path.join(out_dir, '%s.population.png' % title), dpi=300)
    plotting.plot_fraction_group(model, title, tags)
    pylab.savefig(os.path.join(out_dir, '%s.fraction.png' % title), dpi=300)

plotting.plot_fractions(model, model.labels[:])
pylab.savefig(os.path.join(out_dir, 'fraction.png'), dpi=300)
plotting.plot_populations(model, model.labels[:])
pylab.savefig(os.path.join(out_dir, 'population.png'), dpi=300)

plotting.plot_vars(model, ['incidence', 'notification', 'mortality'])
pylab.savefig(os.path.join(out_dir, 'rates.png'), dpi=300)
# plotting.plot_vars(model, ['prevalence'])
# pylab.savefig(os.path.join(out_dir, 'prevalence.png'), dpi=300)

plotting.plot_flows(model, model.labels[:])
pylab.savefig(os.path.join(out_dir, 'flows.png'), dpi=300)

pngs = glob.glob(os.path.join(out_dir, '*png'))
plotting.open_pngs(pngs)


