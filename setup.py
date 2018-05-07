from setuptools import setup, find_packages

setup(
    name='neuron-sw',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
#    license='MIT',
    description='A python package for testing NN parameterization in a SW model',
    long_description=open('README.md').read(),
    install_requires=['tensorflow','keras','xarray','tqdm','scipy','h5py','sklearn','matplotlib'],
    url='https://github.com/brajard/sw',
    author='Julien Brajard',
    author_email='julien.brajard@locean-ipsl.upmc.fr'
)
