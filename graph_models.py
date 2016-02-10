
from autumn.model import \
    SingleStrainSystem, \
    ThreeStrainSystem, \
    Stage6PopulationSystem
from stage4 import ThreeStrainSystem

from autumn.plotting import plot_fractions, plot_populations
import pylab

import os

os.system('rm -rf models')
os.makedirs('models')

for name, ModelClass in [
        ('single.strain', SingleStrainSystem),
        ('three.strain', ThreeStrainSystem),
        # ('stage6', Stage6PopulationSystem),
    ]:
    print 'running', name

    model = ModelClass()
    model.set_flows()

    base = os.path.join('models', name)
    model.make_graph(base + '.workflow')

    model.make_times(1700, 2050, 0.05)
    model.integrate_explicit()

    plot_fractions(model, model.labels, base + '.fraction.png')

    plot_populations(model, model.labels, base + '.population.png')


os.system('open models/*png')