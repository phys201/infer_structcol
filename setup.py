from setuptools import setup

setup(name='infer_structcol',
      version='0.0.1',
      description='Inference of parameters of structural color samples',
      url='http://github.com/p201-sp2016/infer_structcol',
      author='Mandrill Rumps',
      author_email='vhwang@g.harvard.edu',
      license='GNU GPL v3',
      packages=['infer_structcol'],
      install_requires=['structcol>=0.2','numpy', 'scipy','emcee','matplotlib', 'pint', 'lmfit'],
      dependency_links=['http://github.com/manoharan-lab/structural-color/tarball/master#egg=structcol'],
      zip_safe=False)
