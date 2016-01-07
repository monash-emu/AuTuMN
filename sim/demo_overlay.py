# -*- coding: utf-8 -*-
"""
RUNMODEL

Possibly temporary code for running the model.

Version: 2015aug05
"""

import optimatb
from optimatb.defaults import meta
from optimatb.parameters import pars

s = optimatb.Sim(meta,pars)
s.run()
s2 = optimatb.Sim(meta,pars)
s2.pars['birth'] = 0.0202
s2.run()

optimatb.plot_overlay([s,s2])

print('Done.')