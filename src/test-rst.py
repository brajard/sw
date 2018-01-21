from shalw import SWmodel
import numpy as np
import xarray as xr
try:
	import matplotlib.pyplot as plt
	PLOT = True
except:
	PLOT = False
endtime = 400
midtime = 200
#Run : one step
SW = SWmodel(nx=80,ny=80)
SW.initstate_cst(0,0,0)
SW.save(time=np.arange(0,endtime,1),name='longrun.nc')
for i in range(endtime):
	SW.next()
SW.remove_save('longrun.nc')

#Run : two steps
SW.initstate_cst(0,0,0)
SW.save(time=np.arange(0,midtime,1),name='firstrun.nc')
for i in range(0,midtime):
	SW.next()
SW.save_rst('restart_'+str(midtime)+'.nc')
SW.remove_save('firstrun.nc')
SW.initstate_cst(0,0,0)
SW.save(time=np.arange(midtime,endtime,1),name='secondrun.nc')
SW.inistate_rst('restart_'+str(midtime)+'.nc')
for i in range(midtime,endtime):
 	SW.next()

#plots
ds_long = xr.open_dataset('longrun.nc')
ds_first = xr.open_dataset('firstrun.nc')
ds_second = xr.open_dataset('secondrun.nc')
x = 0
y = 0

if PLOT:
	fig, axes = plt.subplots(ncols=1)
	ds_long.hphy.isel(x=x,y=y).plot(ax=axes)
	ds_first.hphy.isel(x=x,y=y).plot(ax=axes)
	ds_second.hphy.isel(x=x,y=y).plot(ax=axes)
	plt.show()