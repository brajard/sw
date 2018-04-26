import sys
sys.path.append('..')
import os
import numpy as np
import xarray as xr
try:
    import matplotlib.pyplot as plt
    PLOT = True
except:
    PLOT = False

os.chdir(os.path.dirname(os.path.realpath(__file__)))
figdir = '../../data/egu'

fname0 = '../../data/egu/test-00-new.nc'
fname1 = '../../data/egu/test-nn-new.nc'

suf = 'new'

dsdr=dict()
dsdr[0] = xr.open_dataset(fname0)
dsdr[1] = xr.open_dataset(fname1)
#dsdr[0]['time'] = dsdr[0].time * dsdr[0].dt / (3600 * 24)  # time in days
#dsdr[1]['time'] = dsdr[1].time * dsdr[1].dt / (3600 * 24)  # time in days

mtime = dsdr[0].time[-1]+1
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

lab = {0:'True model',1:'neural net. model'}

names = {'EC','EP','PV'}
ylabel = {'EC':'kinetic energy','EP':'potential energy','PV':'potential vorticity'}
fig=dict()
ax = dict()
fig0 = dict()
ax0 = dict()
for k in names:
    fig[k],ax[k] = plt.subplots()
    #for plotting numerical model alone:
    fig0[k],ax0[k] = plt.subplots()

for k,ds in dsd.items():
    ds['time'] = ds.time * ds.dt / (3600 * 24)  # time in days

    Ec = cinetic_ener(ds.uphy, ds.vphy, ds.hphy, H=ds.H)
    Ep = potential_ener(ds.hphy, gstar=ds.gstar, H=ds.H)
    PV = potential_vor(ds.uphy, ds.vphy, ds.hphy, ds.dx, ds.dy, H=ds.H, f0=ds.f0, beta=ds.beta, y=ds.y)
    Ec.plot(ax=ax['EC'],label=lab[k])
    Ep.plot(ax=ax['EP'],label=lab[k])
    PV.plot(ax=ax['PV'],label=lab[k])
    if k==0: #plot numerical model alone
        Ec.plot(ax=ax0['EC'],label=lab[k])
        Ep.plot(ax=ax0['EP'],label=lab[k])
        PV.plot(ax=ax0['PV'],label=lab[k])

for k in names:
    ax[k].set_ylabel(ylabel[k],fontsize=20)
    ax[k].set_xlabel('time[days]')
    ax[k].legend()
    ax0[k].set_ylabel(ylabel[k],fontsize=20)
    ax0[k].set_xlabel('time[days]')
    fig[k].savefig(os.path.join(figdir,'evol'+suf+'-'+k+'.png'))
    fig0[k].savefig(os.path.join(figdir, 'evol-0-' + suf + '-' + k + '.png'))



figh, axh = plt.subplots()
dsd[0].hphy.stack(geo=('x','y')).mean(axis=1).plot(ax=axh,label=lab[0])
dsd[1].hphy.stack(geo=('x','y')).mean(axis=1).plot(ax=axh,label=lab[1])
axh.set_xlabel('time[days]')
axh.set_ylabel('height',fontsize=20)
axh.legend()
figh.savefig(os.path.join(figdir,'evol'+suf+'-h.png'))

vlim = {'hphy':-150,'uphy':0.4,'vphy':0.6}

for k,ds in dsd.items():
    svar = {'hphy','uphy','vphy'}
    figm=dict()
    axm=dict()
    for v in svar:
        figm[v], axm[v] = plt.subplots()
        ds[v].mean(dim='time').plot(ax=axm[v],vmin=-vlim[v],vmax=vlim[v])
        axm[v].axis('off')
        figm[v].savefig(os.path.join(figdir,'mean'+suf+'-'+str(v)+str(k)+'.png'))
