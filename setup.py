from setuptools import setup

setup(name='infer_structcol',
      version='0.0.1',
      description='Inference of parameters of structural color samples',
      url='http://github.com/p201-sp2016/infer_structcol',
      author='Mandrill Rumps',
      author_email='vhwang@g.harvard.edu',
      license='GNU GPL v3',
      packages=['infer_structcol'],
      requires=['structcol','numpy', 'scipy','emcee','matplotlib', 'pint'],
      zip_safe=False)

