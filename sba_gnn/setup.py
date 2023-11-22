from setuptools import setup, find_packages

setup(name='interactions_package',
      packages=find_packages(),
      install_required = ['numpy', 'pandas', 'matplotlib'],
      version ='1.0')