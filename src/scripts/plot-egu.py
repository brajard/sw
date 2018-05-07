#!/usr/bin/env python
import numpy as np
import sys
import os
sys.path.append('..')

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

os.chdir(os.path.dirname(os.path.realpath(__file__)))

from keras.optimizers import SGD
param = 'uparam'
# Plot input/output
outdir = '../../data/egu'
Xfile = '../../data/app-' + param + '-im/data_X.npy'
yfile = '../../data/app-' + param + '-im/data_y.npy'
X = np.load(Xfile)
y = np.load(yfile)
i = 100

splot = {0:'u',1:'h',2:'tau'}
for k,v in splot.items():
	fig,ax = plt.subplots()
	im = ax.imshow(X[i,:,:,k])
	ax.axis('off')
	fig.colorbar(im)
	fig.savefig(join(outdir,'trainset-'+v+'.png'))

fig,ax = plt.subplots()
im = ax.imshow(y[i,:,:,0])
ax.axis('off')
fig.colorbar(im)
fig.savefig(join(outdir,'trainset-uparam.png'))