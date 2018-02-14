"""
create a base of image for data assimilation (static), i.e.
A true state
some observations of height
a background state
"""

from shalw import SWmodel
import numpy as np
import os
# import xarray as xr
from tqdm import tqdm

try:
	import matplotlib.pyplot as plt

	PLOT = True
except ImportError:
	PLOT = False

# Set to True to run the model
recompute = False

fname = '../data/base-assim.nc'

if recompute or not os.path.isfile(fname):
	rfile = '../data/restart_10years.nc'

	para = {'hphy', 'uphy', 'vphy'}

	SW = SWmodel(nx=80, ny=80)
	SW.inistate_rst(rfile)
	SW.set_time(0)

	endtime = 12 * 30 * 12 * 40  # 40 years
	time = np.arange(0, endtime, 12 * 7)
	SW.save(time=time, para=para, name=fname)
	for i in tqdm(range(endtime)):
		SW.next()
