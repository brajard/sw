#!/usr/bin/env python

# coding: utf-8

# # Script to run a restart
# This script run the model from a rest state during 10 years in order to produce a restart start.

# In[ ]:


## Import package
from neuralsw.model.shalw import SWmodel
import neuralsw
import numpy as np
import xarray as xr
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
#get_ipython()().run_line_magic('matplotlib', 'notebook')


# In[ ]:


## Specify the output
PLOT = True #if plot is wanted

#rootdir
rootdir = os.path.realpath(	os.path.join(os.getcwd(),'../..'))

#directory to store the data
datadir = os.path.realpath(os.path.join(rootdir,'data'))

#suffix (modify default parameters if not empty)
suf = '_low'

#savefile (to check the run)
outname = os.path.join(datadir,'restartrun'+ suf + '.nc')

#restartfile

rstfile = os.path.join(datadir,'restart_10years'+ suf +'.nc')

#Duration of the integration
endtime = 48*30*12*10 #10 years

print('data directory:',datadir)


# In[ ]:


## Init model
if 'low' in rstfile:
    SW = SWmodel(nx=80,ny=80,warg={'taux0':0.1})
else:
    SW = SWmodel(nx=80,ny=80)
SW.initstate_cst(0,0,0)


#Save every month
SW.save(time=np.arange(0,endtime,48*30),name=outname)


# In[ ]:


# run the model
for i in tqdm(range(endtime)):
    SW.next()


# In[ ]:


# Save the restart
SW.save_rst(rstfile)


# In[ ]:


## Plots conservative quantities
if PLOT:
    ## For some unresolved reasons , the keras modul (included in model) should not be called before dealing with nc files
    import neuralsw.model.modeltools as model
    
    ds = xr.open_dataset(outname)

    fig,ax = plt.subplots(nrows=3,sharex=True)
    Ec = model.cinetic_ener(ds=ds)
    Ep = model.potential_ener(ds=ds)
    Pv = model.potential_vor(ds=ds)
    Ec.plot(ax=ax[0])
    Ep.plot(ax=ax[1])
    Pv.plot(ax=ax[2])
    ax[0].set_title('mean kinetic energy')
    ax[0].set_ylabel('Ec')
    ax[0].set_xlabel('')
    ax[1].set_title('mean potential energy')
    ax[1].set_ylabel('Ep')
    ax[1].set_xlabel('')
    ax[2].set_title('mean potential vorticity')
    ax[2].set_ylabel('Pv')
    plt.show()

