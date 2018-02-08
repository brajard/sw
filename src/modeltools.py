import numpy as np
import pickle
from keras.models import load_model


def standardization (x,m,et):
	return (x-m)/et
def invstandardization (x,m,et):
	return m + x*et

def normalize(X,moy,et,fnorm=standardization):
	npar = X.shape[-1]
	Xn = np.zeros_like(X)
	for j in range(npar):
		Xn[...,j] = fnorm(X[...,j],moy[j],et[j])
	return Xn
def denormalize(Xn,moy,et,fdenorm=invstandardization):
	npar = Xn.shape[-1]
	X = np.zeros_like(Xn)
	for j in range(npar):
		X[...,j] = fdenorm(Xn[...,j],moy[j],et[j])
	return X

def loadmymodel(fname):
	nn = pickle.load(open(fname, "rb"))
	nn.load_model()
	return nn

class mymodel:
	def __init__( self , model=None,
			moyX=0,etX=1,
			moyY=0,etY=1,
			fnorm=standardization,
			fdenorm=invstandardization):
		self._model = model
		self._moyX = moyX
		self._etX = etX
		self._moyY = moyY
		self._etY = etY
		self._fnorm = fnorm
		self._fdenorm = fdenorm
		if model is not None:
			npar= self._model.input_shape[-1]
			if not hasattr(self._moyX, "__len__"):
				self._moyX = self._moyX*np.ones(shape=(npar))
			if not hasattr(self._etX, "__len__"):
				self._etX = self._etX*np.ones(shape=(npar))
			if not hasattr(self._moyY, "__len__"):
				self._moyY = self._moyY * np.ones(shape=(npar))
			if not hasattr(self._etY, "__len__"):
				self._etY = self._etY * np.ones(shape=(npar))
		#used to load model
		self._modelfile = None

	def fit( self,X,y, **kwargs ):
		npar = X.shape[-1]
		Xn = normalize(X,self._moyX,self._etX,fnorm=self._fnorm)
		yn = normalize(y,self._moyY,self._etY,fnorm=self._fnorm)
		self._model.fit(Xn,yn,**kwargs)

	def predict (self,X,denorm=True,**kwargs):
		Xn = normalize(X,self._moyX,self._etX,fnorm=self._fnorm)
		y_predict = self._model.predict(Xn,**kwargs)
		if denorm:
			y_predict = denormalize(y_predict,
				self._moyY,self._etY,fdenorm=self._fdenorm)
		return y_predict
	def save( self, fname ):

		#to avoid pickling the model
		savemodel = self._model
		self._modelfile =fname + '.h5'
		self._model.save(self._modelfile)
		self._model = None
		with open(fname,'wb') as f:
			pickle.dump(self, f, -1)
		self._model = savemodel
	def load_model( self ):
		self._model = load_model(self._modelfile)



