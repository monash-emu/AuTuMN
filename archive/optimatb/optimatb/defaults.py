# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:28:34 2015

@author: cliffk
"""


from matplotlib.pylab import arange, array
from utils import run
from datetime import datetime

meta = {}

# Housekeeping
meta['version'] = 0.0
meta['commit'] = run('git rev-parse HEAD')
meta['date'] = datetime.today().strftime("%Y-%b-%d %H:%M:%S")

# Define metaparameters
meta['yearstart'] = 1700
meta['yearend'] = 2020
meta['dt'] = 0.05
meta['nstates'] = 5
meta['iS'] = 0
meta['iL'] = 1
meta['iA'] = 2
meta['iD'] = 3
meta['iT'] = 4
meta['statenames'] = ['Susceptible', 'Latent', 'Active', 'Detected', 'Treated']
meta['initial'] = array([1e6, 0, 1e3, 0, 0])

# Calculate additional useful default quantities
meta['tvec'] = arange(meta['yearstart'] , meta['yearend'] , meta['dt'])
meta['npts'] = len(meta['tvec'])
