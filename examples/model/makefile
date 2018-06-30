nbdir = ../notebooks
scripts = jrestart.py jmake_simu.py jmake_dataset.py jmake_learn.py jmake_simultest.py
all : $(scripts)
jrestart.py:$(nbdir)/restart.ipynb
jmake_simu.py:$(nbdir)/make_simu.ipynb
jmake_dataset.py:$(nbdir)/make_dataset.ipynb
jmake_learn.py:$(nbdir)/make_learn.ipynb
jmake_simultest.py:$(nbdir)/make_simultest.ipynb

$(scripts):
	jupyter nbconvert --to script $<  --output $(nbdir)/tmp
	mv $(nbdir)/tmp.py tmp
	echo '#!/usr/bin/env python' > header
	cat tmp >> header
	sed 's/get_ipython\(\)/#get_ipython\(\)/g' header >tmp
	mv tmp $@
	rm header
	chmod u+x $@


