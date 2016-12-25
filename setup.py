from setuptools import setup
from setuptools import find_packages


version = '0.1.2'

setup(name='keras-vis',
      version=version,
      description='Neural Network visualization toolkit for keras',
      author='Raghavendra Kotikalapudi',
      author_email='ragha@outlook.com',
      url='https://github.com/raghakot/keras-vis',
      download_url='https://github.com/raghakot/keras-vis/tarball/{}'.format(version),
      license='MIT',
      install_requires=['keras', 'imageio'],
      packages=find_packages())
