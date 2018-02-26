"""
Run the shallow-water model with a change in the z0 calculation
"""

import sys
sys.path.append('..')
from shalw import SWmodel
import os
import numpy as np
import xarray as xr
from tqdm import tqdm
try:
	import matplotlib.pyplot as plt
	PLOT = True
except:
	PLOT = False

os.chdir(os.path.dirname(os.path.realpath(__file__)))

class SWz0(SWmodel):
	def __init__ (self,
			f0=3.5e-5, beta=2.11e-11, gamma=2e-7, gstar=0.02,rho0=1000, H=500, taux0=0.15, tauy0=0, nu=0.72, dt=1800,
			dx=20e3, dy=20e3, alpha=0.025, nx=80, ny=80 ):
		SWmodel.__init__(self,f0, beta, gamma, gstar,rho0, H, taux0, tauy0, nu, dt,
			dx, dy, alpha, nx, ny )

	def VOR(self):
		upre = self._dstate['upre']
		vpre = self._dstate['vpre']
		vor = (vpre[self.ind((0,1))]-vpre[self.ind((0,0))])/self.dx -\
		       (upre[self.ind((0,0))]-upre[self.ind((-1,0))])/self.dy +\
		       self.f
		self.set_state('vor',vor)
		#Copy to unbiased the mean computation in LAMU and LAMV
		#other solution: vor = 0 ?????
		self._dstate['vor'][:,type(self)._nedge-1] = 0
		self._dstate['vor'][self._endy,:] = 0

if __name__=="__main__":
	rfile = '../../data/restart_10years.nc'
	outfile_chg = '../../data/test-z0-change.nc'
	outfile_nochg = '../../data/test-z0-nochange.nc'

	#Run 2 versions of the model
	# - one with the old z limit condition
	# - one with the changed z limit condition
	files2run = {outfile_chg,outfile_nochg}
	files2run = {} #comment to rerun
	for outfile in files2run:
		para = { 'hphy', 'vphy', 'uphy'}
		if 'nochange' in outfile:
			SW = SWmodel(nx=80,ny=80)
		else:
			SW = SWz0(nx=80, ny=80)
		SW.inistate_rst(rfile)
		SW.set_time(0)

		endtime = 12 * 30 * 12 * 10
		SW.save(time=np.arange(0, endtime, 12 * 7), para=para, name=outfile)
		for i in tqdm(range(endtime+1)):
			SW.next()
	ds_chg = xr.open_dataset(outfile_chg)
	ds_nochg = xr.open_dataset(outfile_nochg)
	out = ds_nochg.assign(dh=ds_chg.hphy - ds_nochg.hphy)
	out = out.assign(du=ds_chg.uphy - ds_nochg.uphy)
	out = out.assign(dv=ds_chg.vphy - ds_nochg.vphy)
	f2plot= {'dh','du','dv'}


	for par in f2plot:
		fig, ax = plt.subplots(nrows=2)
		out[par].isel(time=1).plot(ax=ax[0])
		out[par].isel(time=-1).plot(ax=ax[1])
		plt.tight_layout()
		plt.show()



