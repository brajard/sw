#!/bin/bash

#Generate script
ipython nbconvert --to script ../notebooks/restart.ipynb
echo '#!/usr/bin/env python' > jrestart.py
cat restart.py >> jrestart.py
chmod u+x jrestart.py
