from datatools import mydata

#Base for uparam
appfile = '../data/base_10years.nc'
infield = ['uphy','hphy']
outfield = 'uparam'
app = mydata(appfile,outfield=outfield,infield=infield,forcfield=['taux'])
app.make_base()
app.save_base(('../data/app-uparam'))