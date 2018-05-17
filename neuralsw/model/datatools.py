"""contains functionality to handle data"""
import numpy as np
import xarray as xr
import os
from os.path import join,isdir
from .shalw import SWmodel
import scipy.stats as st

try:
	import matplotlib.pyplot as plt

	PLOT = True
except:
	PLOT = False

def TimeSeq(endtime,freq,start=0,nseq=2):
	starts = np.arange(start, endtime, freq)
	time = np.empty(shape=(0,), dtype=int)
	for s in starts:
		time = np.concatenate((time, np.arange(s,s+nseq)), axis=0)
	return time

def MakeBigImage(smallimage,ind):
	indy,indx = ind
	ny,nx = max(indy)+1,max(indx)+1
	out = np.empty(shape = (ny,nx))
	for i,(iy,ix) in enumerate(zip(indy,indx)):
		out[iy,ix] = smallimage[i]
	return out

def MakeSmallImages_ind(fieldshape,n=(3,3)):
	Ly,Lx = fieldshape
	nx, ny = n
	ni = Ly*Lx
	indx = np.empty(shape=ni, dtype=int)
	indy = np.empty(shape=ni, dtype=int)
	dy, dx = (ny // 2), (nx // 2)
	i = 0  # index in out
	for ix in range(dx, Lx + dx):
		for iy in range(dy, Ly + dy):
			indy[i] = iy - dy  # because we want index of the orginal field
			indx[i] = ix - dx
			i = i + 1
	return (indy,indx)

def MakeSmallImages(field,n=(3,3)):
	"""

	:param field: image to be divided (shape Ly,Lx)
	:param n: tuple(int,int) size of small images
	:return: array shape [Ly*Lx,n,n]
	"""
	Ly,Lx=field.shape
	nx,ny = n
	#number of small images
	ni = Ly*Lx
	#output
	out = np.empty(shape =(ni,ny,nx))
	indx = np.empty(shape=ni,dtype=int)
	indy = np.empty(shape=ni,dtype=int)
	dy,dx = (ny//2) , (nx//2)

	#add padding for extract in the edges
	datapad = np.pad(field,((dy, dy), (dx,dx)),
		'constant',constant_values=((0,0),(0,0)))

	i = 0 #index in out
	for ix in range(dx,Lx+dx):
		for iy in range(dy,Ly+dy):
			out[i,:,:]=datapad[iy-dy:iy+dy+1,ix-dx:ix+dx+1]
			indy[i] = iy-dy #because we want index of the orginal field
			indx[i] = ix-dx
			i = i+1
	return out,(indy,indx)

def MakeBase1im(infield,outfield,nout=(1,1),delta=(1,1)):
	nyout,nxout = nout
	dy,dx = delta
	nyin,nxin = nyout+2*dy,nxout+2*dx
	npar = len(infield)
	(dataout,indout) = MakeSmallImages(outfield,n=nout)
	ny,nx = infield[0].y.size,infield[0].x.size
	n = ny*nx
	ldatain=[]
	for data in infield:
		out = np.empty(shape=(n,nyin,nxin,1))
		(out[:,:,:,0],indin) = MakeSmallImages(data,n=(nyin,nxin))
		ldatain.append(out)
	datain = np.concatenate(ldatain,axis=3)
	return (datain,indin),(dataout,indout)

class mydata:
	def __init__( self, fname, outfield,infield, forcfield, nout=(1,1),delta=(1,1),dt=1, SW=None):
		self._outfield = outfield
		self._infield = infield
		self._forcfield = forcfield
		self._nout = nout
		self._delta = delta
		self._dt = dt
		if SW is None:
			self._SW = SWmodel()
		else:
			self._SW = SW
		if isinstance(fname,str):
			self._fname = fname
			self._data = xr.open_dataset(self._fname)
		else:
			self._fname = ''
			self._data = fname

		#define times
		self._t = []
		if self.dt == 0:
			tslice = slice(0,None)
		else:
			tslice = slice(0,-self.dt)
		for t in self.data.time[tslice]:
			if int(t+dt) in self.data.time.values:
				self._t.append((t,t+dt))

		self._X = None
		self._y = None
		self._indinx = None
		self._indiny = None
		self._indoutx = None
		self._indouty = None
		self._indint = None
		self._indoutt = None

	def make_base_im (self):
		""""make base without cut into small images"""
		atin = np.array([tin for tin,tout in self.t])
		atout = np.array([tout for tin,tout in self.t])
		#Liste of indata
		Lin = [ (self.data[f].sel(time=atin).values) for f in self.infield]
		Lforc = []
		ny = self.data.y.size
		nx = self.data.x.size
		for name in self.forcfield:
			tmp = getattr(self._SW,name)
			#test if it is constant in time
			if len(tmp.shape) == 2:
				tmp = np.tile(tmp[np.newaxis,:,:],(len(atin),1,1))
			Lforc.append(tmp)
		self._X = np.stack(Lin+Lforc,axis=-1)
		self._y = self.data[self.outfield].sel(time=atout).values[:,:,:,np.newaxis]
		self._indinx = self.data.x
		self._indiny = self.data.y
		self._indoutx = self.data.x
		self._indouty = self.data.y
		self._indint = atin
		self._indoutt = atout

	def make_base( self ):
		ny, nx = len(self.data.y), len(self.data.x)
		nyout, nxout = self._nout
		dy, dx = self._delta
		lX = []
		ly = []
		lindinx = []
		lindiny = []

		lindoutx = []
		lindouty = []
		lindint = []
		lindoutt = []
		for tin,tout in self.t:
			(X1im,(indin1y,indin1x)),(y1im,(indouty,indoutx)) = MakeBase1im(
				[self.data[f].sel(time=tin) for f in self.infield]+\
				[getattr(self._SW,name) for name in self.forcfield],
				self.data[self.outfield].sel(time=tout),
				nout=self._nout,delta=self._delta)
			lX.append(X1im)
			ly.append(y1im)
			lindinx.append(indin1x)
			lindiny.append(indin1y)
			lindint.append(int(tin)*np.ones(indin1x.shape))
			lindoutx.append(indoutx)
			lindouty.append(indouty)
			lindoutt.append(int(tout)*np.ones(indoutx.shape))


		#X = np.concatenate((X,X1im),axis=0)
			#y = np.concatenate((y,y1im),axis=0)
		#self._X = lX
		self._X = np.concatenate(lX,axis=0)
		self._y = np.squeeze(np.concatenate(ly,axis=0),axis=2)
		#self._indinx = lindinx
		self._indinx = np.concatenate(lindinx,axis=0)
		self._indiny = np.concatenate(lindiny,axis=0)
		self._indoutx = np.concatenate(lindoutx,axis=0)
		self._indouty = np.concatenate(lindouty,axis=0)
		self._indint = np.concatenate(lindint,axis=0)
		self._indoutt = np.concatenate(lindoutt,axis=0)

	def save_base( self,indir,pref='data' ,mkdir=True):
		if mkdir and not isdir(indir):
			os.mkdir(indir)
		att2save = {'_X','_y',
			'_indinx','_indiny',
			'_indoutx','_indouty',
			'_indint','_indoutt'}
		for name in att2save:
			np.save(join(indir,pref+name+'.npy'),getattr(self,name))





	@property
	def t( self ):
		return self._t
	@property
	def fname( self ):
		return self._fname
	@property
	def infield( self ):
		return self._infield
	@property
	def forcfield( self ):
		return self._forcfield
	@property
	def data( self ):
		return self._data
	@property
	def dt( self ):
		return self._dt
	@property
	def outfield( self ):
		return self._outfield

def gkern(kernlen=7, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

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

if __name__ == "__main__":
	appfile = '../data/base_10years.nc'
	#data = xr.open_dataset(appfile)
	#out,ind = MakeSmallImages(data.hphy[0,:,:])
	infield = ['uphy','hphy']
	outfield = 'uparam'
	app = mydata(appfile,outfield=outfield,infield=infield,forcfield=['taux'])
	app.make_base()
	#app.save_base(('../data/app-uparam'))

	#test makebigimage
	data = xr.open_dataset(appfile)
	out,ind = MakeSmallImages(data.uparam[10,:,:],n=(1,1))
	rec2D = MakeBigImage(out,ind)
	plt.imshow(data.uparam[10,:,:]-rec2D)
	plt.show()
