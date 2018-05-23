from neuralsw.model.shalw import SWmodel
from neuralsw.model.shalwnet import SWparnnim,SWparnnhdyn
import numpy as np
import time
import xarray as xr
from tqdm import tqdm

try:
	import matplotlib.pyplot as plt
	PLOT = True
except:
	PLOT = False

PLOT = False
rfile = '../../data/restart_10years.nc'
nnufile = '../../data/nn0_uparam_nonoise_std/model_uparam.pkl'
nnvfile = '../../data/nn0_vparam_nonoise_std/model_vparam.pkl'
#nhdynfile = '../data/nn-hdynparam-amsgrad-long2/model_hdyn-im.pkl'

SW0 = SWmodel(nx=80, ny=80)
SW = SWparnnim(nnupar=nnufile,nnvpar=nnvfile,nx=80,ny=80)
#SW = SWparnnhdyn(nnupar=nnufile,nnvpar=nnvfile,nnhdyn=nhdynfile, nx=80,ny=80)
SW.inistate_rst(rfile)
SW.set_time(0)
SW0.inistate_rst(rfile)
SW0.set_time(0)
# time of the spinup
# endtime = 12*30*12*10 #10 years
endtime = 48 * 30 * 12 * 1
#endtime = 48
# Declare to save all phy parameters (default) every 12*30 time step(1 month)
# 10000 is approximatively 13 months
para = { 'hphy', 'uphy', 'uparam','vparam','vphy','taux','tauy' }
SW.save(time=np.arange(0, endtime,48*7 ), para=para, name='../../data/test-nn.nc')
SW0.save(time=np.arange(0, endtime,48*7 ), para=para, name='../../data/test-00.nc')

# Run the model
start = time.time()

for i in tqdm(range(endtime)):
	SW.next()
	SW0.next()
end = time.time()
print('run duration', end - start, 'seconds')
# SW.save_rst(name='restart.nc')
# Plot the final state
if PLOT:
	ds = xr.open_dataset('test.nc')
	plt.imshow(SW.get_state('vor'))
	plt.colorbar()
	plt.show()
	x, y = 20, 41
	fig, axes = plt.subplots(nrows=3, sharex=True)
	ds.uphy.isel(x=x, y=y).plot(ax=axes[0])
	ds.udyn.isel(x=x, y=y).plot(ax=axes[1])
	ds.uparam.isel(x=x, y=y).plot(ax=axes[2])
	plt.show()
