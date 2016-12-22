from setuptools import setup
from setuptools import find_packages


setup(name='keras-vis',
      version='0.1',
      description='Network visualization toolkit for keras',
      author='Raghavendra Kotikalapudi',
      author_email='ragha@outlook.com',
      url='https://github.com/raghakot/keras-vis',
      download_url='https://github.com/raghakot/keras-vis/tarball/0.1',
      license='MIT',
      install_requires=['keras', 'imageio'],
      packages=find_packages())
