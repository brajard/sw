"""
This file makes a simulation over 10 years
and save a sequence of 10 snapshot every 2 months
"""
from shalw import SWmodel
import numpy as np
#import xarray as xr
from tqdm import tqdm
try:
	import matplotlib.pyplot as plt
	PLOT = True
except:
	PLOT = False

rfile = '../data/restart_10years.nc'
outfile = '../data/base_10years-2.nc'
SW = SWmodel(nx=80,ny=80)
SW.inistate_rst(rfile)
SW.set_time(0)

nseq = 10
freq = 12*30*2 #2 months
endtime = 12*30*12*10 #10 years

#Create time vector
starts = np.arange(0,endtime,freq)
time = np.empty(shape=(0,),dtype=int)
for s in starts:
	time = np.concatenate((time,np.arange(s,s+nseq)),axis=0)

SW.save(time=time,name=outfile)
for i in tqdm(range(endtime)):
	SW.next()