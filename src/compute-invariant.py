#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:25:45 2018

@author: jbrlod
"""

import sys
import os
sys.path.append('..')
from shalw import SWmodel
import matplotlib.pyplot as plt
#from datatools import cinetic_ener 
import xarray as xr

figdir = '../data/figs'

res = 'mr'
fname = '../data/restartrun_10years_'+res+'.nc'

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def cinetic_ener (uphy, vphy,hphy,H):
    u = uphy.stack(geo=('x','y'))
    v = vphy.stack(geo=('x','y'))
    h = hphy.stack(geo=('x','y'))

    return 0.5*((h+H)*(u**2 + v**2)).mean(axis=1)

def potential_ener(hphy,gstar,H):
    h = hphy.stack(geo=('x','y'))
    return 0.5*gstar*((h+H)**2).mean(axis=1)

ds = xr.open_dataset(fname)

Ec = cinetic_ener(ds.uphy,ds.vphy,ds.hphy,H=500)
Ep = potential_ener(ds.hphy,gstar=0.02,H=500)

fig,ax = plt.subplots(nrows=2,sharex=True)
Ec.plot(ax=ax[0],label='Ec')
ax[0].set_ylabel('Ec')
Ep.plot(ax=ax[1],label='Ep')
ax[1].set_ylabel('Ep')
plt.suptitle('Model '+res)

plt.savefig(os.path.join(figdir,'Energy-'+res+'.png'))

plt.show()
