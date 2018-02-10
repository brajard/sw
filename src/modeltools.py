import numpy as np
import pickle
from keras.models import load_model
from shalw import SWmodel
from datatools import mydata, MakeBigImage
import xarray as xr
try:
	import matplotlib.pyplot as plt

	PLOT = True
except:
	PLOT = False

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
	def predictfield( self,data,outfield,infield, forcfield,
	SW=None):
		if SW is None:
			SW = SWmodel()
		nout = (1,1)
		delta = (1,1)
		field=mydata(data, outfield=outfield,
			infield=infield, forcfield=forcfield, nout=nout, delta=delta,
			dt=1, SW=SW )
		field.make_base()
		y_predict = self.predict(field._X)
		outfield = MakeBigImage(y_predict,
			ind=(field._indouty,field._indoutx))
		return outfield

if __name__ == "__main__":
	name = '../data/model_upar.pkl'
	nn = loadmymodel(name)
	# nn = mymodel(model=nntmp._model,
	# 	moyX = nntmp._moyX,etX=nntmp._etX,
	# 	moyY = nntmp._moyY,etY=nntmp._etY)
	testfile = '../data/test_image.nc'
	tdata = xr.open_dataset(testfile)
	infield = ['uphy', 'hphy']
	outfield = 'uparam'
	forcfield = ['taux']
	y = nn.predictfield(tdata,
			outfield=outfield,infield=infield,forcfield=forcfield)

	fig, (ax1, ax2,ax3) = plt.subplots(ncols=3)
	p1 = ax1.imshow(tdata['uparam'][1,:,:])
	p2 = ax2.imshow(y)
	p3 = ax3.imshow(y-tdata['uparam'][1,:,:],
		cmap=plt.get_cmap('bwr'))
	fig.colorbar(p1,ax=ax1)
	fig.colorbar(p2,ax=ax2)
	fig.colorbar(p3,ax=ax3)
	plt.tight_layout()
	plt.show()
