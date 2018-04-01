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


param = 'hdyn'
netname = 'nn-' + param + 'param-amsgrad-long2'

# fgnet = 'nn-' +param +'param-amsgrad-long'

outdir = '../data'
Xfile = '../data/app-' + param + '-im/data_X.npy'
yfile = '../data/app-' + param + '-im/data_y.npy'

X = np.load(Xfile)
y = np.load(yfile)


nt, ny, nx, npar = X.shape


nn = loadmymodel(join(outdir, netname, 'model_' + param + '-im.pkl'))

y_predict = nn.predict(X)
corr = np.corrcoef(y.ravel(),y_predict.ravel())[0,1]
mse = np.mean(np.square(y.ravel()-y_predict.ravel()))
title = 'corr='+'{:3.3f}'.format(corr)+', mse='+'{:3.2e}'.format(mse)
print(title)
if PLOT:
	plt.plot(y.ravel(), y_predict.ravel(), '.')
	plt.plot(np.array(y.ravel()), y.ravel(), '-r')
	plt.title(title)
	plt.show()
	plt.savefig(join(outdir, netname, 'scatter-all.png'))


