#!/bin/bash

#Generate script
jupyter nbconvert --to script ../notebooks/restart.ipynb  --output ../notebooks/restart
echo '#!/usr/bin/env python' > jrestart.py
cat ../notebooks/restart.py >> jrestart.py
sed 's/get_ipython\(\)/#get_ipython\(\)/g' jrestart.py >jrestarttmp.py
mv jrestarttmp.py jrestart.py
chmod u+x jrestart.py
