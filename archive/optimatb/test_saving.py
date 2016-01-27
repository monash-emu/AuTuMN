# -*- coding: utf-8 -*-
"""
RUNMODEL

Possibly temporary code for running the model.

Version: 2015aug05
"""

import optimatb
from optimatb.defaults import meta
from optimatb.parameters import pars

s = optimatb.Sim(meta,pars,'Default')
s.run()

optimatb.save(s,'testfile2')
s2 = optimatb.load('testfile2')

s2.plot()