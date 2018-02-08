#!/usr/bin/env python
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from modeltools import mymodel,loadmymodel
try:
	import matplotlib.pyplot as plt

	PLOT = True
except:
	PLOT = False

from keras.optimizers import SGD
# Load data
Xfile = '../data/app-uparam/data_X.npy'
yfile = '../data/app-uparam/data_y.npy'

X = np.load(Xfile)
y = np.load(yfile)

X_train, X_test, y_train, y_test = \
	train_test_split(X, y, test_size=0.33)

ny,nx = 3,3
npar = 3

#Model definition
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(ny, nx, npar)))
#model.add(Dropout(0.2))
model.add(Conv2D(1, (1, 1), activation='linear'))
model.add(Flatten())
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
moy_y = np.mean(y_train)
et_y = np.std(y_train)
#ytrain_n = (y_train - moy_y)/et_y
#ytest_n = (y_test - moy_y)/et_y
nn = mymodel(model,moyX=moy,etX=et,moyY=moy_y,etY=et_y)

#Training
nn.fit(X_train, y_train, epochs=5, batch_size=128)

y_predict = nn.predict(X_test)
if PLOT:
	plt.plot(y_test,y_predict,'.')
	plt.show()
nn.save('../data/model_upar.pkl')

