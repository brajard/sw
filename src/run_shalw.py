from shalw import SWmodel
from shalwnet import SWparnn
import numpy as np
import time
import xarray as xr
from tqdm import tqdm

try:
	import matplotlib.pyplot as plt
	PLOT = True
except:
	PLOT = False


rfile = '../data/restart_10years.nc'
nnufile = '../data/model_upar.pkl'
nnvfile = '../data/model_vpar.pkl'

SW = SWmodel(nx=80, ny=80)
#SW = SWparnn(nnupar=nnufile,nnvpar=nnvfile,nx=80,ny=80)
SW.inistate_rst(rfile)
SW.set_time(0)
# time of the spinup
# endtime = 12*30*12*10 #10 years
endtime = 12 * 30 * 12 * 1
# Declare to save all phy parameters (default) every 12*30 time step(1 month)
# 10000 is approximatively 13 months
para = { 'hphy', 'hdyn', 'uphy', 'udyn', 'uparam','vparam','vphy' }
SW.save(time=np.arange(1, endtime, 12 * 7), para=para, name='test_tau.nc')

# Run the model
start = time.time()

for i in tqdm(range(endtime)):
	SW.next()
end = time.time()
print('run duration', end - start, 'seconds')
# SW.save_rst(name='restart.nc')
# Plot the final state
if PLOT:
	ds = xr.open_dataset('test.nc')
	# plt.imshow(SW.get_state('vor'))
	# plt.colorbar()
	# plt.show()
	x, y = 20, 41
	fig, axes = plt.subplots(nrows=3, sharex=True)
	ds.uphy.isel(x=x, y=y).plot(ax=axes[0])
	ds.udyn.isel(x=x, y=y).plot(ax=axes[1])
	ds.uparam.isel(x=x, y=y).plot(ax=axes[2])
	plt.show()