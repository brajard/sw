"""
Run the shallow-water model in high rsolution
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
from skimage.util import view_as_blocks

if __name__=="__main__":


	rfile_lr = '../../data/restart_10years_lr.nc'
	rfile_hr = '../../data/restart_10years_hr.nc'
	rfile_mr = '../../data/restart_10years_mr.nc'
	files2run = {rfile_lr,rfile_hr,rfile_mr}

	# First Make restart if it does not exist
	for rfile in files2run:
		if not os.path.isfile(rfile):
			if '_hr' in rfile:
				fact = 8
				outname = '../../data/restartrun_10years_hr.nc'
				#SW = SWmodel(nx=320, ny=320, dx=5e3, dy=5e3,nu=0.18,dt=450)
			elif '_lr' in rfile:
				fact = 1
				outname = '../../data/restartrun_10years_lr.nc'
				#SW = SWmodel(nx=40,ny=40,dx = 40e3, dy=40e3, nu=1.44, dt=3600//fact)
			else:
				fact = 2
				outname = '../../data/restartrun_10years_mr.nc'
			SW = SWmodel(nx = 40*fact, ny = 40*fact, dx = 40e3//fact, dy = 40e3//fact,
			nu = 1.44/fact, dt = 3600//fact)
			SW.initstate_cst(0, 0, 0)
			endtime = (fact*24) * 30 * 12 * 10 #10 years of spinup
			SW.save(time=np.arange(0, endtime, (fact*24)*30), name=outname) #monthly

			#run the model
			for i in tqdm(range(endtime)):
				SW.next()

			#Save the restart
			SW.save_rst(rfile)

	if FALSE:
		ds = xr.open_dataset('../../data/restartrun_10years_hr.nc')
		block_shape = (4, 4)
		spar = {'hphy','uphy','vphy'}
		lr = dict()
		for par in spar:
			lr[par] = np.empty((ds.time.size,80,80))
			for t in range(ds.time.size):
				temp = view_as_blocks(ds[par][t,:,:].values, block_shape)
				flatten = temp.reshape(temp.shape[0],temp.shape[1],-1)
				lr[par][t] = np.mean(flatten,axis=2)
