"""Use this file to generate the restart file"""

from shalw import SWmodel
import numpy as np
import xarray as xr
try:
	import matplotlib.pyplot as plt
	PLOT = True
except:
	PLOT = False
endtime = 12*30*12*20 #86400 : 20 years
outname = '../data/restartrun.nc'


#Init model
SW = SWmodel(nx=80,ny=80)
SW.initstate_cst(0,0,0)

#Save every month
SW.save(time=np.arange(0,endtime,12*15),name=outname)

#run the model
for i in range(endtime):
	SW.next()

#Save the restart
SW.save_rst('../data/restart_10years.nc')

#Plots
ds = xr.open_dataset(outname)

x = 30
y = 30
t = -1

if PLOT:
	fig, axes = plt.subplots(ncols=2)
	ds.hphy.isel(x=x,y=y).plot(ax=axes[0])
	ds.hphy.isel(time=t).plot(ax=axes[1])
	plt.show()