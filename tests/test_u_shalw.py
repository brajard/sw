import xarray as xr
from shalw import SWmodel
import unittest
import numpy as np

class shalwtest (unittest.TestCase):
	def setUp ( self ):
		"""Initialisation des tests."""
		self.rfile = '../data/restart_10years.nc'
		self.refile = 'test0.nc'
		self.para = {'hphy','hdyn','uphy','udyn','uforc','uparam'}
		self.ds = xr.open_dataset(self.refile)
		SW = SWmodel(nx=80, ny=80)
		SW.inistate_rst(self.rfile)
		SW.set_time(0)
		endtime = 12 * 30 * 12 * 1
		SW.save(time=np.arange(1, endtime, 12 * 7), para=self.para, name='test.nc')
		for i in range(endtime):
			SW.next()
		self.ds2 = xr.open_dataset('test.nc')

	def test_hphy( self ):
		"""
		test hphy on shallow water model for one year
		:return:
		"""
		self.assertAlmostEqual(np.linalg.norm(self.ds.hphy-self.ds2.hphy),0.)

