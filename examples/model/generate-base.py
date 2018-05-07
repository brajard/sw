"""
This file makes a simulation over 40 years
and save a sequence of 2 snapshots every  month
"""
from shalw import SWmodel
import numpy as np
# import xarray as xr
from tqdm import tqdm
try:
	import matplotlib.pyplot as plt
	PLOT = True
except:
	PLOT = False

rfile = '../data/restart_20years_mr.nc'
outfile = '../data/base_40years_mr.nc'
testfile = '../data/test_image.nc'
para = {'ufil', 'vfil', 'hfil',
	'hphy', 'uphy', 'vphy',
	'hdyn', 'udyn', 'vdyn',
	'uparam', 'vparam', 'hdyn'}
SW = SWmodel(nx=80,ny=80)
SW.inistate_rst(rfile)
SW.set_time(0)

nseq = 2
freq = 12*30*1 #1 month
endtime = 12*30*12*40 #10 years
testime = 12*30*10 #one year after test
# Create time vector
starts = np.arange(0, endtime, freq)
time = np.empty(shape=(0,), dtype=int)
for s in starts:
	time = np.concatenate((time, np.arange(s,s+nseq)), axis=0)

SW.save(time=time, para=para,name=outfile)
SW.save(time=range(endtime+testime-nseq,endtime+testime),
	para=para,name=testfile)
for i in tqdm(range(endtime+testime)):
	SW.next()
