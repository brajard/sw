"""contains functionality to handle data"""
import numpy as np
import xarray as xr
from shalw import SWmodel

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
		out = np.empty(shape=(n,1,nyin,nxin))
		(out[:,0,:,:],indin) = MakeSmallImages(data,n=(nyin,nxin))
		ldatain.append(out)
	datain = np.concatenate(ldatain,axis=1)
	return (datain,indin),(dataout,indout)

class mydata:
	def __init__( self, fname, outfield,infield, forcfield, nout=(1,1),delta=(1,1),dt=1, SW=None):
		self._fname = fname
		self._outfield = outfield
		self._infield = infield
		self._forcfield = forcfield
		self._nout = nout
		self._delta = delta
		self._dt = dt
		if SW is None:
			self._SW = SWmodel()

		self._data = xr.open_dataset(self.fname)
		#define times
		self._t = []
		for t in self.data.time[:-self.dt]:
			if int(t+dt) in self.data.time.values:
				self._t.append((t,t+dt))
		ny,nx = len(self.data.y),len(self.data.x)
		nyout,nxout = nout
		dy, dx = delta
		nyin, nxin = nyout + 2 * dy, nxout + 2 * dx

		#X = np.empty(shape=(0,1,nyin,nxin))
		#y = np.empty(shape=(0,1,nyout,nxout))
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
				[self.data[f].sel(time=tin) for f in self.infield],
				self.data[self.outfield].sel(time=tout),
				nout=nout,delta=delta)
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
		self._y = np.concatenate(ly,axis=0)
		#self._indinx = lindinx
		self._indinx = np.concatenate(lindinx,axis=0)
		self._indiny = np.concatenate(lindiny,axis=0)
		self._indoutx = np.concatenate(lindoutx,axis=0)
		self._indouty = np.concatenate(lindouty,axis=0)
		self._indint = np.concatenate(lindint,axis=0)
		self._indoutt = np.concatenate(lindoutt,axis=0)





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

if __name__ == "__main__":
	appfile = '../data/base_10years.nc'
	#data = xr.open_dataset(appfile)
	#out,ind = MakeSmallImages(data.hphy[0,:,:])
	infield = ['uphy','hphy']
	outfield = 'uparam'
	app = mydata(appfile,outfield=outfield,infield=infield,forcfield=[])