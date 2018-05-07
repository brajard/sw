#!/usr/bin/env python
import numpy as np
import os
from os.path import join, isdir
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from modeltools import mymodel, loadmymodel
from keras import regularizers
from keras import optimizers

try:
	import matplotlib.pyplot as plt

	PLOT = True
except:
	PLOT = False

from keras.optimizers import SGD

# Load data
plt.rcParams['agg.path.chunksize'] = 10000

param = 'hdyn'
param = 'v'

#hdyn networks:
#netname = 'nn-' + param + 'param-amsgrad-long5'
#netname = 'nn-' + param + '-noise'
#netname = 'nn-' + param + '-large'
#netname = 'nn-' + param + '-noise1000'
#netname = 'nn-' + param + '-dropout2'
#netname = 'nn-' + param + '-prod'

#u,v networks
netname = 'nn-' +param +'param-im'

outdir = '../data'

#add a extension to the name
ext = ''
if param in {'u','v'}:
        ext = 'param'
Xfile = '../data/app-' + param + ext + '-im/data_X.npy'
yfile = '../data/app-' + param + ext + '-im/data_y.npy'

X = np.load(Xfile)
y = np.load(yfile)


nt, ny, nx, npar = X.shape


nprod = (npar)*(npar-1)//2
Xp = np.empty(shape=(X.shape[:-1])+(nprod,))
k=0
for i in range(npar):
	for j in range(i+1,npar):
		Xp[:,:,:,k] = X[:,:,:,i]*X[:,:,:,j]
		k=k+1
if 'prod' in netname:
        X = np.concatenate((X,Xp),axis=3)

nt,ny,nx,npar = X.shape

#add a extension to the name
ext = ''
if param in {'u','v'}:
        ext = 'par'
nn = loadmymodel(join(outdir, netname, 'model_' + param + ext +'-im.pkl'))

y_predict = nn.predict(X)
corr = np.corrcoef(y.ravel(),y_predict.ravel())[0,1]
mse = np.mean(np.square(y.ravel()-y_predict.ravel()))
title = 'corr='+'{:3.4f}'.format(corr)+', mse='+'{:3.2e}'.format(mse)
print(netname,':',title)
if PLOT:
	plt.plot(y.ravel(), y_predict.ravel(), '.')
	plt.plot(np.array(y.ravel()), y.ravel(), '-r')
	plt.title(title)
	plt.savefig(join(outdir, netname, 'scatter-all.png'))
	plt.show()
	plt.semilogy(nn._history['loss'], color='gray')
	plt.semilogy(nn._history['val_loss'], color='black')
	plt.savefig(join(outdir, netname, 'history-all.png'))
	plt.show()


