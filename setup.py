# To run this package, please git clone or download the structural-color package
# from https://github.com/manoharan-lab/structural-color into the root directory
# of your infer-structcol package. 

from setuptools import setup

setup(name='infer_structcol',
      version='0.0.1',
      description='Inference of parameters of structural color samples',
      requires=['structural-color','numpy', 'scipy','emcee','matplotlib', 'pint'],
      url='http://github.com/p201-sp2016/infer_structcol',
      author='Mandrill Rumps',
      author_email='vhwang@g.harvard.edu',
      license='GNU GPL v3',
      packages=['infer_structcol'],
      zip_safe=False)
