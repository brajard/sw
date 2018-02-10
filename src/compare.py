import xarray as xr
import matplotlib.pyplot as plt

phy = xr.open_dataset('test.nc')
nn = xr.open_dataset('test2.nc')
x, y = 20, 41

fig, ax = plt.subplots(nrows=1)
phy.uphy.isel(x=x,y=y).plot(ax=ax,label='physical model')
nn.uphy.isel(x=x,y=y).plot(ax=ax,label='nn parametrization')
ax.legend()
plt.show()

fig,ax = plt.subplots(nrows=1)
phy.hphy.isel(x=x,y=y).plot(ax=ax,label='physical model')
nn.hphy.isel(x=x,y=y).plot(ax=ax,label='nn parametrization')
ax.legend()
plt.show()

fig,ax = plt.subplots(nrows=1)
phy.uparam.isel(x=x,y=y).plot(ax=ax,label='physical model')
nn.uparam.isel(x=x,y=y).plot(ax=ax,label='nn parametrization')
ax.legend()
plt.show()

