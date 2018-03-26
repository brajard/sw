from datatools import mydata

#Base for uparam
appfile = '../data/base_10years_mr.nc'
infield = ['uphy','hphy']
outfield = 'uparam'
app = mydata(appfile,outfield=outfield,infield=infield,forcfield=['taux'],dt=1)
app.make_base()
app.save_base(('../data/app-uparam'))
app.make_base_im()
app.save_base('../data/app-uparam-im')

#Base for vparam
appfile = '../data/base_10years_mr.nc'
infield = ['vphy','hphy']
outfield = 'vparam'
app = mydata(appfile,outfield=outfield,infield=infield,forcfield=['tauy'],dt=1)
app.make_base()
app.save_base(('../data/app-vparam'))
app.make_base_im()
app.save_base('../data/app-vparam-im')