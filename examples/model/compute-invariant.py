#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:25:45 2018

@author: jbrlod
"""

import sys
import os
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
#from datatools import cinetic_ener 
import xarray as xr
from neuralsw.model.modeltools import cinetic_ener, potential_ener, potential_vor
figdir = '../data/figs'

sres = {'lr','mr','hr'}

#os.chdir(os.path.dirname(os.path.realpath(__file__)))

#Initialize plots
names = {'EC','EP','PV'}
fig=dict()
ax = dict()
for k in names:
    fig[k],ax[k] = plt.subplots()


figm,axm = plt.subplots(ncols=3)

ind = {'hr':0, 'mr':1,'lr':2}

for res in sres:
    fname = '../data/restartrun_20years_'+res+'.nc'
    
    ds = xr.open_dataset(fname)
    ds['time'] = ds.time*ds.dt/(3600*24) #time in days
    Ec = cinetic_ener(ds.uphy,ds.vphy,ds.hphy,H=ds.H)
    Ep = potential_ener(ds.hphy,gstar=ds.gstar,H=ds.H)
    PV = potential_vor(ds.uphy,ds.vphy,ds.hphy,ds.dx,ds.dy,H=ds.H,f0=ds.f0,beta=ds.beta,y=ds.y)
    
    
    Ec.plot(ax=ax['EC'],label=res)
    Ep.plot(ax=ax['EP'],label=res)
    PV.plot(ax=ax['PV'],label=res)

    #plot mean hstate
    val = np.sqrt(ds.vphy**2 + ds.uphy**2)
    #val = ds.uphy*(ds.hphy+ds.H)
    im = axm[ind[res]].imshow(val[ds.time>1000].mean(dim='time'),vmin=-0,vmax=1.5)
    axm[ind[res]].axis('off')
    axm[ind[res]].set_title(res)

#plt.savefig(os.path.join(figdir,'Energy-'+res+'.png'))

for k in names:
    ax[k].set_ylabel(k)
    ax[k].set_xlabel('time[days]')
    ax[k].legend()
    fig[k].savefig(os.path.join(figdir,'spinup_20years_'+k+'.png'))
figm.subplots_adjust(right=0.9)
cbar_ax = figm.add_axes([0.92, 0.25, 0.03, 0.5])
figm.colorbar(im, cax=cbar_ax)
figm.suptitle('mean velocity intensity')
figm.savefig(os.path.join(figdir,'spinup_20years_meanu.png'))
