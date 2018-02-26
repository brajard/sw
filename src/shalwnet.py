from shalw import SWmodel
from modeltools import loadmymodel
from datatools import MakeSmallImages, MakeBigImage,  MakeSmallImages_ind
import numpy as np

class SWparnn(SWmodel):
	def __init__ (self, nnupar,nnvpar,nout=(1,1),delta=(1,1),
			f0=3.5e-5, beta=2.11e-11, gamma=2e-7, gstar=0.02,rho0=1000, H=500, taux0=0.15, tauy0=0, nu=0.72, dt=1800,
			dx=20e3, dy=20e3, alpha=0.025, nx=80, ny=80 ):
		SWmodel.__init__(self,f0, beta, gamma, gstar,rho0, H, taux0, tauy0, nu, dt,
			dx, dy, alpha, nx, ny )

		if isinstance(nnupar, str):
			self._nnupar = loadmymodel(nnupar)
		else:
			self._nnupar = nnupar
		if isinstance(nnvpar, str):
			self._nnvpar = loadmymodel(nnvpar)
		else:
			self._nnvpar = nnvpar
		self._nout = nout
		self._delta = delta
		self._nin = tuple(n+2*d for (n,d) in zip(self._nout,self._delta))
		#Calculate index of the ouptput image
		self._indout = MakeSmallImages_ind((self.ny,self.nx),
			n=self._nout)

	def computeuparam( self):
		upre,_ = MakeSmallImages(self.upre,n=self._nin)
		hpre,_ = MakeSmallImages(self.hpre,n=self._nin)
		taux,_ = MakeSmallImages(self.taux,n=self._nin)
		x = np.stack((upre,hpre,taux),axis=3)
		y = self._nnupar.predict(x)
		self.uparam = MakeBigImage(y,ind=self._indout)

	def computevparam( self):
		vpre,_ = MakeSmallImages(self.vpre,n=self._nin)
		hpre,_ = MakeSmallImages(self.hpre,n=self._nin)
		tauy,_ = MakeSmallImages(self.tauy,n=self._nin)
		x = np.stack((vpre,hpre,tauy),axis=3)
		y = self._nnvpar.predict(x)
		self.vparam = MakeBigImage(y,ind=self._indout)
