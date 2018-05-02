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
from shalw import SWmodel



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

def genere_obs(std_o,shape,lobs):
	""" genere a deltao for a field of size shape"""
	nmin,nmax = lobs
	blist = np.empty(shape=np.prod(shape), dtype=bool)
	nobs = np.random.randint(nmin,nmax+1)
	epso = np.empty(shape=shape)
	blist[:nobs] = False
	blist[nobs:] = True
	mask = np.random.choice(blist, shape, replace=False)
	epso[mask] = np.nan
	epso[~mask] = np.random.normal(0, std_o, nobs)
	return epso

def etkf (epsb, y, m):
	#Compute raw forecast variables
	xf,dim,ldim = dict2array(epsb)

	#Compute mean of forecast
	xf_moy = xf.mean(axis=1)

	#Compute normalised anomaly matrix
	Xf = (xf - xf_moy)/np.sqrt(m-1)

	mask_obs = np.logical_not(np.isnan(y))
	y1D = np.matrix(y[mask_obs][:,np.newaxis])
	mask1D = np.reshape(mask_obs,-1)
	Hd = dict()
	Hd['hphy'] = np.matrix(make_H(mask1D))
	#Add the other dims
	for par in epsb:
		if par not in {'hphy'}:
			Hd[par] = np.matrix(np.zeros_like(Hd['hphy']))
	#Concatenate H
	H = np.concatenate(tuple(Hd[k] for k in ldim),axis=1)

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

	epsa = array2dict(xa,dim,ldim)
	return epsa

def forward(SW,epsb,ntime=1):
	"""Forward an ensemble of state durint ntime
	!!!! Not working with the save struct"""
	epsf = dict()
	ntime = 10
	for par in epsb:
		epsf[par] = np.empty_like(epsb[par])
		m = epsb[par].shape[0]
	tstart = SW.t
	for im in range(m):
		SW.set_time (tstart)
		#Initial state of the model
		for parname in SW._restVar:
			SW.set_state(parname, epsb[parname][im])

		# Run the model for ntime step time
		for t in range(ntime):
			SW.next()

		#Get the forecast step
		for parname in SW._restVar:
			epsf[parname][im] = SW.get_state(parname)
	return epsf

if __name__ == "__main__":
	# size of the ensemble
	m = 100

	# noise levels
	std_b = dict()
	std_o = dict()
	std_b['hfil'] = 100
	std_b['hphy'] = 100

	std_b['ufil'] = 0.1
	std_b['uphy'] = 0.1

	std_b['vfil'] = 0.1
	std_b['vphy'] = 0.1

	std_o['hphy'] = 1
	lobs = (500,1500)

	# Convolution kernel for b
	k = gkern(kernlen=7, nsig=3)

	# dataset
	fname = '../data/base-bck-obs.nc'

	# restart file
	rfile = '../data/restart_10years.nc'

	# load the dataset
	ds = xr.open_dataset(fname)

	#load restart
	dr = xr.open_dataset(rfile)

	# starting time
	t0 = 0

	# Generation of the ensemble
	epsb = dict()
	for par in std_b:
		epsb[par] = np.empty(shape=(m,) + dr[par][0].shape)

		for im in range(m):
			eps = np.random.normal(0, std_b[par], dr[par][0].shape)
			epsb[par][im] = ndimage.convolve(eps, k, mode='constant')
			epsb[par][im] += dr[par][0]



	#Model load:
	SW = SWmodel()
	SW.inistate_rst(rfile)
	SW.set_time(t0)

	#Reference model (to produce obs)
	SW0 = SWmodel()
	SW0.inistate_rst(rfile)
	SW0.set_time(t0)

	#parameters in control state variable (not all ?)
	cparam = {'hphy','uphy','vphy','ufil','vfil','hfil'}
	epsa = epsb
	endtime = 3
	for t in tqdm(ds.hphy_o.time[:endtime]):
		#forecast
		if t>t0:
			epsb = forward(SW, epsa, int(t-t0))
			for i in range(int(t0),int(t)):
				SW0.next()

		#Analysis
		epsc = {par:epsb[par] for par in epsb if par in cparam}
		y = SW0.hphy + genere_obs(std_o['hphy'],SW0.hphy.shape,lobs)
		epsa = etkf(epsc,y,m)

		#Add other variables
		for var in epsb:
			if var not in cparam:
				epsa[var] = epsb[var]
		t0 = t

