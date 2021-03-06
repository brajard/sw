#!/usr/bin/env pythonw
import numpy as np
import os
from os.path import join,isdir
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from modeltools import mymodel,loadmymodel
from keras import regularizers

try:
	import matplotlib.pyplot as plt

	PLOT = True
except:
	PLOT = False

from keras.optimizers import SGD
# Load data



param = 'v'
netname = 'nn-' +param +'param-im2'
fgnet = 'nn-' +param +'param-im'

outdir = '../data'
Xfile = '../data/app-'+ param + 'param-im/data_X.npy'
yfile = '../data/app-'+ param + 'param-im/data_y.npy'

X = np.load(Xfile)
y = np.load(yfile)

X_train, X_test, y_train, y_test = \
	train_test_split(X, y, test_size=0.1)

nt,ny,nx,npar = X.shape
if 'fgnet' in locals():
        nn = loadmymodel(join(outdir,fgnet,'model_'+param+'-im.pkl'))
else:

#Model definition
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu',
                         padding='same',
                         #kernel_regularizer=regularizers.l1(0.001),
                         input_shape=(ny, nx, npar)))
        #model.add(Dropout(0.2))
        model.add(Conv2D(1, (1, 1), activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        #normalization
        #Xtrain_n = np.zeros_like(X_train)
        #Xtest_n = np.zeros_like(X_test)
        moy = np.zeros(npar)
        et = np.zeros(npar)
        for j in range(npar):
                moy[j] = np.mean(X_train[:,:,:,j].ravel())
                et[j] = np.std(X_train[:,:,:,j].ravel())
                #	Xtrain_n[:,:,:,j] = (X_train[:,:,:,j] - moy[j])/et[j]
                #	Xtest_n[:,:,:,j] = (X_test[:,:,:,j] - moy[j])/et[j]
        moy_y = np.mean(y_train.ravel())
        et_y = np.std(y_train.ravel())
        #ytrain_n = (y_train - moy_y)/et_y
        #ytest_n = (y_test - moy_y)/et_y
        nn = mymodel(model,moyX=moy,etX=et,moyY=moy_y,etY=et_y)

#Training
nn.fit(X_train, y_train, epochs=800, batch_size=1,validation_split=0.1)

y_predict = nn.predict(X_test)
if not isdir(join(outdir, netname)):
	os.mkdir(join(outdir, netname))
nn.save(join(outdir,netname,'model_'+param+'par-im.pkl'))

if PLOT:
	plt.plot(y_test.ravel(),y_predict.ravel(),'.')
	plt.plot([min(y_test.ravel())]*2,[max(y_test.ravel())]*2,'-r')
	plt.savefig(join(outdir,netname,'scatter-test.png'))
	plt.show()
	plt.semilogy(nn._history['loss'], color='gray')
	plt.semilogy(nn._history['val_loss'], color='black')
	plt.savefig(join(outdir, netname, 'history.png'))
	plt.show()

