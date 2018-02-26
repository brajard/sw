"""
create a base of image for data assimilation (static), i.e.
A true state
some observations of height
a background state
"""

from shalw import SWmodel
import numpy as np
import os
import xarray as xr
from tqdm import tqdm
from datatools import gkern
from scipy import ndimage
import numpy.ma as ma
try:
	import matplotlib.pyplot as plt

	PLOT = True
except ImportError:
	PLOT = False


# Set to True to run the model
recompute = False


# noise levels
std_b = dict()
std_o = dict()
std_b['hphy'] = 100
std_b['uphy'] = 0.1
std_b['vphy'] = 0.1
std_o['hphy'] = 1

#Convolution kernel for b
k = gkern(kernlen=7,nsig=3)

#number of obs/image nmin/nmax
n = (500,1500)

#input name
fname = '../data/base-assim.nc'

#output name
fout = '../data/base-bck-obs.nc'

if recompute or not os.path.isfile(fname):
	rfile = '../data/restart_10years.nc'

	para = {'hphy', 'uphy', 'vphy'}

	SW = SWmodel(nx=80, ny=80)
	SW.inistate_rst(rfile)
	SW.set_time(0)

	endtime = 12 * 30 * 12 * 40  # 40 years
	time = np.arange(0, endtime, 12 * 7)
	SW.save(time=time, para=para, name=fname)
	for i in tqdm(range(endtime)):
		SW.next()

#load the dataset
ds = xr.open_dataset(fname)

#noise on background
epsb = dict()
for par in std_b:
	eps = np.random.normal(0,std_b[par],ds[par].shape)
	epsb[par] = np.empty_like(ds[par])
	for t in range(ds.time.size):
		epsb[par][t] = ndimage.convolve(eps[t],k,mode='constant')
		epsb[par][t] += ds[par][t]
	ds = ds.assign(**{par+'_b':xr.DataArray(epsb[par],coords=ds[par].coords)})

epso = dict()
depso = dict()
nmin,nmax = n

for par in std_o:
	nt,ny,nx = ds[par].shape
	nobs = np.random.randint(nmin,nmax+1,nt)
	epso[par] = np.empty_like(ds[par]).view(ma.MaskedArray)
	blist = np.empty(shape=ny*nx,dtype=bool)
	for t in range(nt):
		#Constructing the mask :
		# be carreful, the True values correspond to masked ones
		blist[:nobs[t]] = False
		blist[nobs[t]:] = True
		mask = np.random.choice(blist,(ny,nx),replace=False)
		epso[par][t,mask] = ma.masked
		epso[par][t,~mask] = np.random.normal(0,std_o[par],nobs[t])
		epso[par][t,~mask] += ds[par][t].values[~mask]
	ds = ds.assign(**{ par + '_o': xr.DataArray(epso[par], coords=ds[par].coords) })

ds.to_netcdf(fout)

if PLOT:
	t = 1000
	fig, axes = plt.subplots(nrows=3)
	p1 = axes[0].imshow(ds['hphy'][t],label='truth')
	p2 = axes[1].imshow(epsb['hphy'][t],label='background')
	p3 = axes[2].imshow(epso['hphy'][t],label='obs')
	plt.tight_layout()
	plt.show()