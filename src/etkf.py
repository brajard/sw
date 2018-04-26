from datatools import gkern
import numpy as np
import itertools
import os
import xarray as xr
from scipy import ndimage
import scipy
from datatools import make_H
import matplotlib.pyplot as plt
from tqdm import tqdm

#size of the ensemble
m = 10

# noise levels
std_b = dict()
std_o = dict()
std_b['hphy'] = 100
std_b['uphy'] = 0.1
std_b['vphy'] = 0.1
std_o['hphy'] = 1

#Convolution kernel for b
k = gkern(kernlen=7,nsig=3)

#dataset
fname = '../data/base-bck-obs.nc'

#restart file
rfile = '../data/restart_10years.nc'

#load the dataset
ds = xr.open_dataset(fname)

#starting time
t0 = 0

#Generation of the ensemble
epsb = dict()
for par in std_b:
	epsb[par] = np.empty(shape=(m,)+ds[par].shape[1:])

	for im in range(m):
		eps = np.random.normal(0,std_b[par],ds[par][t0].shape)
		epsb[par][im] = ndimage.convolve(eps,k,mode='constant')
		epsb[par][im] += ds[par][t0]


def dict2array(epsb):
	dim = dict()
	ldim = []
	for k in epsb:
		dim[k] = epsb[k].shape[1:]
		ldim.append(k)
		m = epsb[k].shape[0]
	lX = []
	for k in ldim:
		if 'xarray' in str(type(epsb['hphy'])):
			val = epsb[k].values
		else:
			val = epsb[k]
		lX.append(val.reshape(m,-1).T)
	X = np.concatenate(lX,axis=0)
	return np.matrix(X),dim,ldim

def array2dict(X,dim,ldim):
	X = np.array(X)
	m = X.shape[1]
	epsb = dict()
	for k in ldim:
		indmax = np.prod(dim[k])
		epsb[k] = X[:indmax,:].T.reshape((m,)+dim[k])
		X = X[indmax:,:]
	return epsb

#Analysis time
t =0


#Obs:
y = ds.hphy_o[t]

#Compute raw forecast variables
xf,dim,ldim = dict2array(epsb)

#Compute mean of forecast
xf_moy = xf.mean(axis=1)

#Compute normalised anomaly matrix
Xf = (xf - xf_moy)/np.sqrt(m-1)

mask_obs = np.logical_not(np.isnan(y))
y1D = np.matrix(y.values[mask_obs][:,np.newaxis])
mask1D = np.reshape(mask_obs.values,-1)
Hd = dict()
Hd['hphy'] = np.matrix(make_H(mask1D))
#Add the uphy,vphy dim
Hd['uphy'] = np.matrix(np.zeros_like(Hd['hphy']))
Hd['vphy'] = np.matrix(np.zeros_like(Hd['hphy']))
#Concatenate H
H = np.concatenate(tuple(Hd[k] for k in Hd),axis=1)

yf = H*xf

#Compute the mean predicted obs
yf_moy = yf.mean(axis=1)

#Compute the normalised predicted obs anomaly
Yf = (yf - yf_moy)/np.sqrt(m-1)

#Observation error
R = np.matrix(std_o['hphy']*np.identity(yf.shape[0]))
Rinv = R.I

#Compute the ensemble transform matrix
Im = np.matrix(np.identity(m))
Gamma = (Im + (Yf.T)*Rinv*Yf).I

#Anlaysis estimate in the ensemble space
wa = Gamma*(Yf.T)*Rinv*(y1D-yf_moy)

#square root of Gamma
Gamma_sqrt = scipy.linalg.sqrtm(Gamma)

#Generating the posterior ensemble
xa = xf_moy + Xf*(wa + np.sqrt(m-1)*Gamma_sqrt)

xa_moy = xa.mean(axis=1)

if True:
	#plot the background, truth, analyis
	da = array2dict(xa_moy, dim, ldim)

	fig,ax = plt.subplots(ncols=3,nrows=3)
	#plot
	for i,par in enumerate(std_b):
		ax[i,0].imshow(epsb[par].mean(axis=0))
		ax[i,1].imshow(ds[par][t])
		ax[i,2].imshow(da[par].squeeze())
	fig.show()

#TODO : Forecast xa with the model