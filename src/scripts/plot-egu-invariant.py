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
figdir = '../../data/figs'

fname0 = '../test0.nc'
fname1 = '../test1.nc'

dsdr=dict()
dsdr[0] = xr.open_dataset(fname0)
dsdr[1] = xr.open_dataset(fname1)
#dsdr[0]['time'] = dsdr[0].time * dsdr[0].dt / (3600 * 24)  # time in days
#dsdr[1]['time'] = dsdr[1].time * dsdr[1].dt / (3600 * 24)  # time in days

mtime = 20
dsd = dict()
dsd[0] = dsdr[0].sel(time=dsdr[0].time<mtime)
dsd[1] = dsdr[1].sel(time=dsdr[1].time<mtime)

def cinetic_ener (uphy, vphy,hphy,H):
    u = uphy.stack(geo=('x','y'))
    v = vphy.stack(geo=('x','y'))
    h = hphy.stack(geo=('x','y'))

    return 0.5*((h+H)*(u**2 + v**2)).mean(axis=1)

def potential_ener(hphy,gstar,H):
    h = hphy.stack(geo=('x','y'))
    return 0.5*gstar*((h+H)**2).mean(axis=1)

def potential_vor(uphy,vphy,hphy,dx,dy,H,f0,beta,y,yc=0):
    u = np.pad(uphy,pad_width=1,mode='constant')
    v = np.pad(vphy,pad_width=1,mode='constant')
    nt,ny,nx = uphy.shape
    dv = v[1:-1,1:-1,2:] - v[1:-1,1:-1:,1:-1]
    du = u[1:-1,1:-1,1:-1] - u[1:-1,0:-2,1:-1]
    zheta = dv/dx - du/dy
    coriolis = f0 + beta*(y-yc)
    coriolis = coriolis.expand_dims('xx',1).expand_dims('tt',0)
    potential_vor = (zheta + coriolis).values/(hphy+H)
    return potential_vor.stack(geo=('x','y')).mean(axis=1)

lab = {0:'num. model',1:'net. model'}

names = {'EC','EP','PV'}
fig=dict()
ax = dict()
for k in names:
    fig[k],ax[k] = plt.subplots()

for k,ds in dsd.items():
    ds['time'] = ds.time * ds.dt / (3600 * 24)  # time in days

    Ec = cinetic_ener(ds.uphy, ds.vphy, ds.hphy, H=ds.H)
    Ep = potential_ener(ds.hphy, gstar=ds.gstar, H=ds.H)
    PV = potential_vor(ds.uphy, ds.vphy, ds.hphy, ds.dx, ds.dy, H=ds.H, f0=ds.f0, beta=ds.beta, y=ds.y)
    Ec.plot(ax=ax['EC'],label=lab[k])
    Ep.plot(ax=ax['EP'],label=lab[k])
    PV.plot(ax=ax['PV'],label=lab[k])

for k in names:
    ax[k].set_ylabel(k)
    ax[k].set_xlabel('time[days]')
    ax[k].legend()
    fig[k].savefig(os.path.join(figdir,'Evol3-'+k+'.png'))
    plt.show()

figu,axu = plt.subplots(ncols=2)
dsd[0].uparam[1].plot(ax=axu[0])
axu[0].set_title('upar num. mod')
axu[0].axis('off')
(dsd[1].uparam[1]-dsd[0].uparam[1]).plot(ax=axu[1])
axu[1].set_title('upar net - num')
axu[1].axis('off')

figu.savefig(os.path.join(figdir,'uparam3.png'))

figv,axv = plt.subplots(ncols=2)
dsd[0].vparam[1].plot(ax=axv[0])
axv[0].set_title('vpar num. mod')
axv[0].axis('off')
(dsd[1].vparam[1]-dsd[0].vparam[1]).plot(ax=axv[1])
axv[1].set_title('vpar net - num')
axv[1].axis('off')

fighd,axhd = plt.subplots(ncols=3)
dsd[0].hdyn[1].plot(ax=axhd[0])
axhd[0].set_title('hdyn num. mod')
axhd[0].axis('off')
dsd[1].hdyn[1].plot(ax=axhd[1])
axhd[0].set_title('hdyn net mod')
axhd[0].axis('off')
(dsd[1].hdyn[1]-dsd[0].hdyn[1]).plot(ax=axhd[2])
axhd[1].set_title('vpar net - num')
axhd[1].axis('off')

fighd.savefig(os.path.join(figdir,'hdyn-t1.png'))
plt.close(fighd)

fighd,axhd = plt.subplots(ncols=3)
dsdr[0].hdyn[20].plot(ax=axhd[0])
axhd[0].set_title('hdyn num. mod')
axhd[0].axis('off')
dsdr[1].hdyn[20].plot(ax=axhd[1])
axhd[0].set_title('hdyn net mod')
axhd[0].axis('off')
(dsdr[1].hdyn[20]-dsdr[0].hdyn[20]).plot(ax=axhd[2])
axhd[1].set_title('vpar net - num')
axhd[1].axis('off')

fighd.savefig(os.path.join(figdir,'hdyn-t20.png'))

figh, axh = plt.subplots()
dsd[0].hphy.stack(geo=('x','y')).mean(axis=1).plot(ax=axh,label=lab[0]);
dsd[1].hphy.stack(geo=('x','y')).mean(axis=1).plot(ax=axh,label=lab[1]);
axh.set_xlabel('time[days]')

figh.savefig(os.path.join(figdir,'evol-h3.png'))