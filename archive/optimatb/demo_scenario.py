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
s2 = optimatb.SimParameter(meta,pars,'TB got worse')
s3 = optimatb.SimParameter(meta,pars,'People stay indoors')
s2.create_override('tbdeath',1900,2000,0.2,0.5)
s3.create_override('ncontacts',1900,2000,5,10)
[x.run() for x in [s,s2,s3]]
optimatb.plot_overlay([s,s2,s3])
