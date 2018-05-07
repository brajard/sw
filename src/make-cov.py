from datatools import gkern
import numpy as np
import itertools
import os
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_H(mask1D):
	n = mask1D.shape[0]
	p = np.sum(mask1D)
	H = np.zeros(shape=(p, n))
	j = 0
	for i in range(n):
		if mask1D[i]:
			H[j, i] = 1
			j = j + 1
	return H

def cov(k,std_b,size):


	localB = np.square(std_b * k)


	ny,nx = size
	n = nx*ny
	nedge = k.shape[0]//2

	B4d = np.empty((size+size))
	B4dp = np.pad(B4d, ((nedge,nedge),)*4, 'constant', constant_values=((0,0),)*4)

	for x,y in itertools.product(range(0,nx),range(0,ny)):
		B4dp[y+nedge,x+nedge,y:y+k.shape[0],x:x+k.shape[0]] = localB

	B = B4dp[nedge:-nedge,nedge:-nedge,nedge:-nedge,nedge:-nedge]
	B = np.reshape(B,(ny,nx,n),order='C')
	B = np.transpose(B,(2,0,1))
	B = np.reshape(B,(n,n),order='C')
	return B

# Set to True to recompute the B-matrix
recompute = False

#Name of the file containing the B-matrix
bname = '../data/bmatrix.npy'

#dataset
fname = '../data/base-bck-obs.nc'
fout = '../data/base-kalman.nc'
ds = xr.open_dataset(fname)
ny = ds['y'].size
nx = ds['x'].size
n = ny*nx

#Observational error characteristic
std_o = 1

if recompute or not os.path.isfile(bname):
	# Background error characteristic
	k = gkern(kernlen=7, nsig=3)  # smoothing kernel
	std_b = 100  # standard deviation
	B = cov(k=k,std_b=std_b,size=(ny,nx))
	np.save(bname,B)
else:
	B = np.load(bname)
#To store all the analaysis
dxa = np.empty_like(ds['hphy'])

for t in tqdm(range(ds.time.size)):
#for t in range(10):
	#Perform an analysis
	xt = ds.hphy[t]
	xb = ds.hphy_b[t]
	y = ds.hphy_o[t]
	mask_obs = np.logical_not(np.isnan(y))
	y1D = y.values[mask_obs]
	#Number of obs
	p = y1D.shape[0]

	mask1D = np.reshape(mask_obs.values,-1)
	xb1D = np.reshape(xb.values,-1)
	xt1D = np.reshape(xt.values,-1)

	#Obervation operator
	H = make_H(mask1D)

	#Check observation error
	#yt = np.dot(H,xt1D)
	#rms_o = np.sqrt(np.mean(np.square(yt-y1D)))
	#print('Check obervational error : true=',std_o,' ; estimated=',rms_o)

	#Observation error
	R = std_o**2 * np.eye(p)
	BHT = np.dot(B,np.transpose(H))
	inv = np.linalg.inv(R + np.dot(H,BHT))

	#Gain
	K = np.dot(BHT,inv)

	#Innovation
	d = y1D - np.dot(H,xb1D)

	#Analysis
	xa1D = xb1D + np.dot(K,d)
	xa = xa1D.reshape((ny,nx))
	dxa[t,:,:] = xa

ds = ds.assign(**{'hphy_a':xr.DataArray(dxa,coords=ds['hphy'].coords)})
ds.to_netcdf(fout)
plot = False
if plot:
	fig,ax = plt.subplots(ncols=3)
	ax[0].imshow(xb,vmin=np.min(xt1D),vmax=np.max(xt1D))
	ax[0].set_title('backrground')
	ax[1].imshow(xt,vmin=np.min(xt1D),vmax=np.max(xt1D))
	ax[1].set_title('Truth')
	ax[2].imshow(xa,vmin=np.min(xt1D),vmax=np.max(xt1D))
	ax[2].set_title('Analysis')
	fig.savefig('../data/figs/kalman-result.png')
	plt.show()