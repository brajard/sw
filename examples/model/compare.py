import xarray as xr
import matplotlib.pyplot as plt

phy = xr.open_dataset('test.nc')
nn = xr.open_dataset('test-nn.nc')
x, y = 0, 41
t = 0

out = nn.assign(dh=nn.hphy - phy.hphy)
out = out.assign(du=nn.uphy - phy.uphy)
out = out.assign(duparam=nn.uparam - phy.uparam)
out.to_netcdf('diff.nc')

fig, ax = plt.subplots(nrows=1)
phy.uphy.isel(x=x,y=y).plot(ax=ax,label='u physical model')
nn.uphy.isel(x=x,y=y).plot(ax=ax,label='u nn parametrization')
ax.legend()
plt.show()

fig,ax = plt.subplots(nrows = 1)
phy.uphy.isel(x=x,time=t).plot(ax=ax,label='u phys model')
nn.uphy.isel(x=x,time=t).plot(ax=ax,label='u nn model')
ax.legend()
plt.show()

fig, ax = plt.subplots(nrows=1)
phy.vphy.isel(x=x,y=y).plot(ax=ax,label='v physical model')
nn.vphy.isel(x=x,y=y).plot(ax=ax,label='v nn parametrization')
ax.legend()
plt.show()


fig,ax = plt.subplots(nrows=1)
phy.hphy.isel(x=x,y=y).plot(ax=ax,label='h physical model')
nn.hphy.isel(x=x,y=y).plot(ax=ax,label='h nn parametrization')
ax.legend()
plt.show()

fig,ax = plt.subplots(nrows=1)
phy.uparam.isel(x=x,y=y).plot(ax=ax,label='uparam physical model')
nn.uparam.isel(x=x,y=y).plot(ax=ax,label='nn parametrization')
ax.legend()
plt.show()


