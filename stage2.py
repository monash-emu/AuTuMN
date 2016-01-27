# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:37:33 2015
Testing the Stage 2 model (second slide in Progressive model construction)
Graph results
@author: JTrauer
"""

from autumn.model import Stage2PopulationSystem, make_steps
from autumn.plotting import make_time_plots_color, make_time_plots_one_panel, plot_fractions
import pylab
import os

population = Stage2PopulationSystem()
population.integrate_scipy(make_steps(0, 50, 1))
population.make_graph('stage2.graph.png')
labels = population.labels
plot_fractions(population, population.labels)
pylab.savefig('stage2.fractions.png')
os.system('open stage2.graph.png')


