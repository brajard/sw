#!/usr/bin/env python3
from inspect import signature
import warnings
from collections import defaultdict
import numpy as np
import xarray as xr
import time
from tqdm import tqdm

try:
	import matplotlib.pyplot as plt
	PLOT = True
except:
	PLOT = False

# debug decorator for print
def debug ( func ):
	def func_wrapper ( self ):
		if self._verbose > 1:
			print('call', func.__name__)
		return func(self)

	return func_wrapper



class SWmodel:
	#edge to deal with boarders
	_nedge = 1
	#set of variables mandatory for a restart
	_restVar = {'hphy','uphy','vphy','ufil','vfil','hfil'}
	def __init__ ( self, f0=3.5e-5, beta=2.11e-11, gamma=2e-7, gstar=0.02,rho0=1000, H=500, taux0=0.15, tauy0=0, nu=0.72, dt=1800,
			dx=20e3, dy=20e3, alpha=0.025, nx=80, ny=80 ):

		self._verbose = 1  # debug
		# Numerical scheme
		self._dt = dt
		self._dx = dx
		self._dy = dy
		self._alpha = alpha
		self._nx = nx
		self._ny = ny
		self._endx = nx + type(self)._nedge
		self._endy = ny + type(self)._nedge
		self._t = 0 #init time
		# Compute grid
		# central points
		self._xc = 0
		self._yc = 0

		# Param to define
		# state_
		# dt
		# f
		# parameters
		self._gstar = gstar
		self._f0 = f0
		self._beta = beta
		self._gamma = gamma
		self._rho0 = rho0
		self._taux0 = taux0
		self._tauy0 = tauy0
		self._nu = nu
		self._H = H

		# taux
		# tauy
		# H
		# rho0
		# g

		# model
		# dx,dy,st
		# gridx,gridy

		# variables
		# h,u,v
		# Verbose mode
		#state variables
		self._varname = {'ufil','uphy','vfil','vphy','hfil','hphy','upre','vpre','hpre','vit','vor',
			'hdyn','udyn','uforc','uparam','vdyn','vforc','vparam'}

		# Compute bool
		self._recompute_grid = True

		# State bool
		self._isinit = False

		# compute all grids variables (except state)
		self.compute_grid()

		#save struct
		self._save = []
		return

#Core of model
	def next(self):
		if not self._isinit:
			raise RuntimeError('model not initialised')
		self.updatesave()
		#upre,vpre,hpre
		self._dstate['hpre']=self._dstate['hphy'].copy()
		self._dstate['upre']=self._dstate['uphy'].copy()
		self._dstate['vpre']=self._dstate['vphy'].copy()


		# precomputation VOR et VIT computation
		self.precompute()

		#uphy,vphy,hphy
		#self.hdyn = (self.MCU() / self.dx + self.MCV() / self.dy)
		self.computehdyn()
		self.hphy = self.hfil - 2 * self.dt * self.hdyn

		self.computeuparam()
		self.udyn = self.LAMV()-self.GRADX()/self.dx
		#self.uforc =  self.TAUX()
		self.udyn[:,-1] = 0
		#self.uforc[:,-1] = 0
		self.uphy = self.ufil + 2*self.dt*(
			self.udyn + self.uparam )
		#self.uphy[:,-1] = 0
		self.computevparam()
		self.vdyn = -self.LAMU()-self.GRADY()/self.dy
		#self.vforc = self.TAUY()
		self.vdyn[0,:] = 0
		#self.vforc[0,:] = 0
		self.vphy = self.vfil + 2*self.dt*(
			self.vdyn + self.vparam )
		#self.vphy[0,:] = 0


		#ufil,vfil,hfil =
		self.ufil = self.upre + self.alpha*(self.ufil - 2 * self.upre + self.uphy)
		self.vfil = self.vpre + self.alpha*(self.vfil - 2 * self.vpre + self.vphy)
		self.hfil = self.hpre + self.alpha*(self.hfil - 2 * self.hpre + self.hphy)
		self._t += 1

	def computeuparam( self ):
		self.uparam = - self.DISU() + self.DIFU() + self.TAUX()
		self.uparam[:, -1] = 0

	def computevparam( self ):
		self.vparam = - self.DISV() + self.DIFV()+ self.TAUY()
		self.vparam[0, :] = 0

	def computehdyn( self):
		self.hdyn = (self.MCU() / self.dx + self.MCV() / self.dy)

	def precompute( self ):
		self.VOR()
		self.VIT()
	def TAUX(self):
		hphy = self._dstate['hphy']
		moyh = .5*(hphy[self.ind((0,1))]+hphy[self.ind((0,0))])
		return self.taux/(self.rho0*(self.H+moyh))
	def TAUY(self):
		hphy = self._dstate['hphy']
		moyh = .5*(hphy[self.ind((-1,0))]+hphy[self.ind((0,0))])
		return self.tauy/(self.rho0*(self.H+moyh))
	def DIFU(self):
		upre = self._dstate['upre']
		DX = (upre[self.ind((0,1))]+
		      upre[self.ind((0,-1))]-
		      2*upre[self.ind()])/(self.dx**2)
		DY = (upre[self.ind((1,0))]+
		      upre[self.ind((-1,0))]-
		      2*upre[self.ind()])/(self.dy**2)
		return self.nu*(DX+DY)
	def DIFV(self):
		vpre = self._dstate['vpre']
		DX = (vpre[self.ind((0,1))]+
		      vpre[self.ind((0,-1))]-
		      2*vpre[self.ind()])/(self.dx**2)
		DY = (vpre[self.ind((1,0))]+
		      vpre[self.ind((-1,0))]-
		      2*vpre[self.ind()])/(self.dy**2)
		return self.nu*(DX+DY)

	def VIT(self):
		upre = self._dstate['upre']
		vpre = self._dstate['vpre']
		hpre = self._dstate['hpre']
		vit= .25*((upre[self.ind((0,-1))]+upre[self.ind((0,0))])**2+
		            (vpre[self.ind((0,0))]+vpre[self.ind((1,0))])**2)+\
		       self.gstar*hpre[self.ind((0,0))]
		self.set_state('vit',vit)

	def VOR(self):
		upre = self._dstate['upre']
		vpre = self._dstate['vpre']
		vor = (vpre[self.ind((0,1))]-vpre[self.ind((0,0))])/self.dx -\
		       (upre[self.ind((0,0))]-upre[self.ind((-1,0))])/self.dy +\
		       self.f
		self.set_state('vor',vor)
		#Copy to unbiased the mean computation in LAMU and LAMV
		#other solution: vor = 0 ?????

		#self._dstate['vor'][:,type(self)._nedge-1] = self._dstate['vor'][:,type(self)._nedge]
		#self._dstate['vor'][self._endy,:] = self._dstate['vor'][self._endy-1,:]


	def LAMV(self):
		vpre = self._dstate['vpre']
		vor = self._dstate['vor']
		return (1./8.)*(vor[self.ind((1,0))] + vor[self.ind((0,0))])*\
		       (vpre[self.ind((0,0))]+vpre[self.ind((1,0))]+vpre[self.ind((0,1))]+vpre[self.ind((1,1))])
	def LAMU(self):
		upre = self._dstate['upre']
		vor = self._dstate['vor']
		return (1./8.)*(vor[self.ind((0,0))] + vor[self.ind((0,-1))])*\
		       (upre[self.ind((-1,-1))]+upre[self.ind((0,-1))]+upre[self.ind((-1,0))]+upre[self.ind((0,0))])
	def GRADX(self):
		vit = self._dstate['vit']
		return vit[self.ind((0,1))] - vit[self.ind((0,0))]
	def GRADY(self):
		vit = self._dstate['vit']
		return vit[self.ind((0,0))] - vit[self.ind((-1,0))]
	def DISU(self):
		#remove sign - (error in pdf ?)
		return self.gamma * self.upre
	def DISV(self):
		#remove sign - (error in pdf ?)
		return  self.gamma * self.vpre
	def MCU(self):
		upre = self._dstate['upre']
		hpre = self._dstate['hpre']
		return self.H*(upre[self.ind((0,0))]-upre[self.ind((0,-1))]) +\
			0.5*(upre[self.ind((0,0))]*(hpre[self.ind((0,0))]+hpre[self.ind((0,1))])-
			upre[self.ind((0,-1))]*(hpre[self.ind((0,0))]+hpre[self.ind((0,-1))]))
	def MCV(self):
		vpre = self._dstate['vpre']
		hpre = self._dstate['hpre']
		return self.H*(vpre[self.ind((1,0))]-vpre[self.ind((0,0))]) +\
			0.5*(vpre[self.ind((1,0))]*(hpre[self.ind((1,0))]+hpre[self.ind((0,0))])-
			vpre[self.ind((0,0))]*(hpre[self.ind((0,0))]+hpre[self.ind((-1,0))]))

#Init functions
	def initstate_cst(self,h=0,u=0,v=0,t=0):
		self.hfil = h
		self.hphy = h
		self.hpre = h
		self.vfil = v
		self.ufil = u
		self.vpre = v
		self.upre = u
		self.vphy = v
		self.uphy = u
		self._t = t
		self._isinit = True

	def inistate_rst(self,filename,time=None):
		ds = xr.open_dataset(filename)
		if time is None and not len(ds.time) == 1 :
			raise ValueError('Specify time or having a single time to init with restart')
		elif time is not None and not any(ds.time == time) :
			raise ValueError('Time value ' + str(time) + ' not available in ' + filename)
		if time == None:
			time = 0
		else:
			time = (ds.time == time)
		if not (self)._restVar.issubset(set(ds.variables)):
			raise ValueError(filename + ' should contains fields '+ str((self)._restVar))
		self._t = int(ds.time[time])
		for parname in (self)._restVar:
			self.set_state(parname,ds[parname])
		self._isinit = True


#Save/Load functions
	def save_rst(self,name='restart.nc'):
		self.save(time=np.arange(self.t,self.t+1,1),para=(self)._restVar,name=name)
		self.updatesave()
		self.remove_save(name)


	def remove_save(self,name):
		self._save = [dsave for dsave in self._save if not dsave['name'] == name ]
	def is_savename(self,name):
		return name in {dsave['name'] for dsave in self._save}
	def save(self,time = np.arange(0,12*30*3,1), para={'hphy','vphy','uphy'},name='default.nc',overwrite=False):
		dsave = dict()
		dsave['name'] = name
		dsave['time'] = time

		dset = dict()
		for var in para:
			dset[var]=(['time','y','x'],np.empty(shape=(len(dsave['time']),self.ny,self.nx)))
		dsave['dset'] = dset
		dsave['iter'] = None
		if self.is_savename(name):
			if overwrite:
				self.remove_save(name)
			else:
				raise ValueError('Name '+ name + ' already exist')
		self._save.append(dsave)

	def updatesave(self):
		for dsave in self._save:
			if dsave['iter'] is None and self.t == dsave['time'][0]:
				if self._verbose>1:
					print('Init save ' + dsave['name'] + ' time ' + str(self.t))
				dsave['iter'] = 0
			if dsave['iter'] is not None and dsave['time'][dsave['iter']] == self.t:
				if self._verbose>1:
					print('update save ' + dsave['name'] + ' time ' + str(self.t))
				for var,d in dsave['dset'].items():
					d[1][dsave['iter']] = self.get_state(var)
				dsave['iter'] += 1
			if dsave['iter'] == len(dsave['time']):
				if self._verbose>1:
					print('finish save ' + dsave['name'] + ' time ' + str(self.t))
				dsave['iter'] = None
				ds = xr.Dataset (dsave['dset'],\
				coords = {'y':self.y, 'x':self.x, 'time':dsave['time']},
				attrs=self.get_params())

				ds.to_netcdf(dsave['name'])


	#Properties
	def ind(self,delta=(0,0)):
		(dy,dx)=delta
		return (slice(type(self)._nedge+dy,self._endy+dy),slice(type(self)._nedge+dx,self._endx+dx))
	def get_state(self,name):
		return self._dstate[name][self.ind()]
	def set_state(self,name,value):
		self._dstate[name][self.ind()] = value
	@property
	def hphy(self):
		return self.get_state('hphy')
	@hphy.setter
	def hphy(self,value):
		self.set_state('hphy',value)
	@property
	def hfil(self):
		return self.get_state('hfil')
	@hfil.setter
	def hfil(self,value):
		self.set_state('hfil',value)
	@property
	def hdyn(self):
		return self.get_state('hdyn')
	@hdyn.setter
	def hdyn(self,value):
		self.set_state('hdyn',value)
	@property
	def udyn(self):
		return self.get_state('udyn')
	@udyn.setter
	def udyn(self,value):
		self.set_state('udyn',value)
	@property
	def vdyn(self):
		return self.get_state('vdyn')
	@vdyn.setter
	def vdyn(self,value):
		self.set_state('vdyn',value)
	@property
	def uforc(self):
		return self.get_state('uforc')
	@uforc.setter
	def uforc(self,value):
		self.set_state('uforc',value)
	@property
	def vforc(self):
		return self.get_state('vforc')
	@vforc.setter
	def vforc(self,value):
		self.set_state('vforc',value)
	@property
	def uparam(self):
		return self.get_state('uparam')
	@uparam.setter
	def uparam(self,value):
		self.set_state('uparam',value)
	@property
	def vparam(self):
		return self.get_state('vparam')
	@vparam.setter
	def vparam(self,value):
		self.set_state('vparam',value)
	@property
	def hpre(self):
		return self.get_state('hpre')
	@hpre.setter
	def hpre(self,value):
		self.set_state('hpre',value)
	@property
	def uphy(self):
		return self.get_state('uphy')
	@uphy.setter
	def uphy(self,value):
		self.set_state('uphy',value)
	@property
	def ufil(self):
		return self.get_state('ufil')
	@ufil.setter
	def ufil(self,value):
		self.set_state('ufil',value)
	@property
	def upre(self):
		return self.get_state('upre')
	@upre.setter
	def upre(self,value):
		self.set_state('upre',value)
	@property
	def vphy(self):
		return self.get_state('vphy')
	@vphy.setter
	def vphy(self,value):
		self.set_state('vphy',value)
	@property
	def vfil(self):
		return self.get_state('vfil')
	@vfil.setter
	def vfil(self,value):
		self.set_state('vfil',value)
	@property
	def vpre(self):
		return self.get_state('vpre')
	@vpre.setter
	def vpre(self,value):
		self.set_state('vpre',value)


	@property
	def nx ( self ):
		return self._nx

	@property
	def ny ( self ):
		return self._ny

	@property
	def dx ( self ):
		return self._dx

	@property
	def dy ( self ):
		return self._dy
	@property
	def dt(self):
		return self._dt
	@dx.setter
	def dx ( self, value ):
		self._dx = value
		if self._recompute_grid:
			self.compute_grid()

	@dy.setter
	def dy ( self, value ):
		self._dy = value
		if self._recompute_grid:
			self.compute_grid()

	@nx.setter
	def nx ( self, value ):
		self._nx = value
		self._endx = self._nx + type(self)._nedge
		if self._recompute_grid:
			self.compute_grid()

	@ny.setter
	def ny ( self, value ):
		self._ny = value
		self._endy = self._ny + type(self)._nedge
		if self._recompute_grid:
			self.compute_grid()

	@property
	def xc ( self ):
		return self._xc

	@property
	def yc ( self ):
		return self._yc

	@xc.setter
	def xc ( self, value ):
		self._xc = value
		if self._recompute_grid:
			self.compute_grid_x()

	@yc.setter
	def yc ( self, value ):
		self._yc = value
		if self._recompute_grid:
			self.compute_grid_y()

	@property
	def x ( self ):
		return self._x

	@property
	def y ( self ):
		"""
		:rtype: numpy.ndarray
		"""
		return self._y

	def set_time( self,value ):
		self._t = value

	@property
	def alpha ( self ):
		return self._alpha

	@debug
	def compute_grid ( self ):
		self._save = []
		self.compute_grid_x()
		self.compute_grid_y()
		self.compute_coriolis()
		self.compute_tau()
		self.compute_grid_state()

	@debug
	def compute_grid_state(self):
		self._dstate = dict()
		for k in self._varname:
			self._dstate[k] = np.empty(shape=(self.ny+2*type(self)._nedge,self.nx+2*type(self)._nedge))
			self._dstate[k].fill(0)
		self._isinit = False

	@debug
	def compute_grid_x ( self ):
		self._x = np.linspace(self.xc - self.dx * self.nx / 2, self.xc + self.dy * self.nx / 2, self._nx)

	@debug
	def compute_grid_y ( self ):
		self._y = np.linspace(self.yc - self.dy * self.ny / 2,  self.yc + self.dy * self.ny / 2, self._ny)

	@debug
	def compute_coriolis ( self ):
		self._f = self.f0 + self.beta * (self.y - self.yc)
		#self._f = np.pad(self._f,(type(self)._nedge,type(self)._nedge),'constant')
		self._f = self._f[:,np.newaxis]
	@debug
	def compute_grid_hfil ( self ):
		self._hfil = np.empty(shape=(self.ny,self.nx))


	@debug
	def compute_tau ( self ):
		self._taux = self.taux0 * np.cos(
			2 * np.pi * (np.tile(self.y[:, np.newaxis], (1, self.nx)) - self.yc) / (self.ny * self.dy))
		self._tauy = self.tauy0 * np.ones((self.ny, self.nx))

	@property
	def taux(self):
		return self._taux
	@property
	def tauy(self):
		return self._tauy
	@property
	def f0 ( self ):
		return self._f0
	@property
	def f(self):
		return self._f
	@property
	def gstar(self):
		return self._gstar

	@f0.setter
	def f0 ( self, value ):
		self._f0 = value
		if self._recompute_grid:
			self.compute_coriolis()

	@property
	def H(self):
		return self._H
	@property
	def t(self):
		return self._t
	@property
	def beta ( self ):
		return self._beta

	@beta.setter
	def beta ( self, value ):
		self._beta = value
		if self._recompute_grid:
			self.compute_coriolis()

	@property
	def gamma ( self ):
		return self._gamma
	@property
	def nu(self):
		return self._nu
	@property
	def rho0 ( self ):
		return self._rho0

	@property
	def taux0 ( self ):
		return self._taux0

	@property
	def tauy0 ( self ):
		return self._tauy0

	@tauy0.setter
	def tauy0 ( self, value ):
		self._tauy0 = value
		if self._recompute_grid:
			self.compute_tau()
	@taux0.setter
	def taux0(self,value):
		self._taux0 = value
		if self._recompute_grid:
			self.compute_tau()


	@classmethod
	def _get_param_names ( cls ):
		"""Get parameter names for the estimator"""
		# fetch the constructor or the original constructor before
		# deprecation wrapping if any
		init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
		if init is object.__init__:
			# No explicit constructor to introspect
			return []

		# introspect the constructor arguments to find the model parameters
		# to represent
		init_signature = signature(init)
		# Consider the constructor parameters excluding 'self' and kwargs
		parameters = [p for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
		for p in parameters:
			if p.kind == p.VAR_POSITIONAL:
				raise RuntimeError("scikit-learn estimators should always "
				                   "specify their parameters in the signature"
				                   " of their __init__ (no varargs)."
				                   " %s with constructor %s doesn't "
				                   " follow this convention." % (cls, init_signature))
		# Extract and sort argument names excluding 'self'
		return sorted([p.name for p in parameters])

	def get_params ( self, deep=True ):
		"""Get parameters for this estimator.

		Parameters
		----------
		deep : boolean, optional
			If True, will return the parameters for this estimator and
			contained subobjects that are estimators.

		Returns
		-------
		params : mapping of string to any
			Parameter names mapped to their values.
		"""
		out = dict()
		for key in self._get_param_names():
			# We need deprecation warnings to always be on in order to
			# catch deprecated param values.
			# This is set in utils/__init__.py but it gets overwritten
			# when running under python3 somehow.
			warnings.simplefilter("always", DeprecationWarning)
			try:
				with warnings.catch_warnings(record=True) as w:
					value = getattr(self, key, None)
				if len(w) and w[0].category == DeprecationWarning:
					# if the parameter is deprecated, don't show it
					continue
			finally:
				warnings.filters.pop(0)

			# XXX: should we rather test if instance of estimator?
			if deep and hasattr(value, 'get_params'):
				deep_items = value.get_params().items()
				out.update((key + '__' + k, val) for k, val in deep_items)
			out[key] = value
		return out

	def set_params ( self, **params ):
		"""Set the parameters of this estimator.

		The method works on simple estimators as well as on nested objects
		(such as pipelines). The latter have parameters of the form
		``<component>__<parameter>`` so that it's possible to update each
		component of a nested object.

		Returns
		-------
		self
		"""
		if not params:
			# Simple optimization to gain speed (inspect is slow)
			return self
		valid_params = self.get_params(deep=True)

		nested_params = defaultdict(dict)  # grouped by prefix
		for key, value in params.items():
			key, delim, sub_key = key.partition('__')
			if key not in valid_params:
				raise ValueError('Invalid parameter %s for estimator %s. '
				                 'Check the list of available parameters '
				                 'with `estimator.get_params().keys()`.' % (key, self))

			if delim:
				nested_params[key][sub_key] = value
			else:
				setattr(self, key, value)

		for key, sub_params in nested_params.items():
			valid_params[key].set_params(**sub_params)

		return self

if __name__ == "__main__":
	SW = SWmodel(nx=80,ny=80)
	rfile = '../data/restart_10years.nc'
	SW = SWmodel(nx=80, ny=80)
	SW.inistate_rst(rfile)
	SW.set_time(0)
	#time of the spinup
	#endtime = 12*30*12*10 #10 years
	endtime = 12*30*12*1
	#Declare to save all phy parameters (default) every 12*30 time step(1 month)
	#10000 is approximatively 13 months
	para = {'hphy','hdyn','uphy','udyn','uforc','uparam'}
	SW.save(time=np.arange(1,endtime,12*7),para=para,name='test.nc')

	#Run the model
	start = time.time()

	for i in tqdm(range(endtime)):
		SW.next()
	end = time.time()
	print('run duration',end - start,'seconds')
	#SW.save_rst(name='restart.nc')
	#Plot the final state



	if PLOT:
		ds = xr.open_dataset('test.nc')
		#plt.imshow(SW.get_state('vor'))
		#plt.colorbar()
		#plt.show()
		x,y = 20,41
		fig, axes = plt.subplots(nrows=4,sharex=True)
		ds.uphy.isel(x=x, y=y).plot(ax=axes[0])
		ds.udyn.isel(x=x, y=y).plot(ax=axes[1])
		ds.uparam.isel(x=x, y=y).plot(ax=axes[2])
		ds.uforc.isel(x=x, y=y).plot(ax=axes[3])
		plt.show()